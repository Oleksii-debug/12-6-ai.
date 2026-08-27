from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import venv
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/research/mlflow_runtime_bootstrap_stress_v1.json")
EXPECTED_MAIN_SHA = "5020afd671a3885c1b738c8b4eafe7525f630546"
EXPECTED_PARENT_HEAD = "56046888b95f4db35f9ca2f38d13dcc0c1fe11e1"
EXPECTED_UPSTREAM_COMMIT = "0572b16ac9e9c98a02df9df40ad3e48ce3b7c588"
EXPECTED_LICENSE_BLOB = "db7cb10b5e330d56b40370bc178974ccabe71458"
EXPECTED_TRACKING_BLOB = "1a672b170a49b800d420127de63cfff7b394065c"
EXPECTED_SOURCE_VERSION = "3.15.3.dev0"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def identity_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_capture(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return completed.returncode, completed.stdout, completed.stderr


def detect_environment() -> dict[str, Any]:
    managers = {name: shutil.which(name) for name in ("python", "pip", "uv", "poetry", "pdm", "conda", "git")}
    cache = None
    cache_result = run_capture([sys.executable, "-m", "pip", "cache", "dir"])
    if cache_result[0] == 0:
        cache = cache_result[1].strip() or None
    cache_writable = bool(cache and os.access(cache, os.W_OK))
    network: dict[str, Any] = {}
    for url in ("https://pypi.org/simple/", "https://github.com/"):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                network[url] = {"status": "reachable", "http_status": response.status}
        except Exception as exc:  # noqa: BLE001
            network[url] = {"status": "unreachable", "error_type": type(exc).__name__, "error": str(exc)}
    installed = {}
    for package in ("mlflow", "pytest", "torch"):
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
    gpu_probe = shutil.which("nvidia-smi")
    return {
        "python": {"executable": sys.executable, "implementation": sys.implementation.name, "version": platform.python_version()},
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "gpu": {"nvidia_smi": gpu_probe, "cuda_visible": bool(gpu_probe)},
        "package_managers": managers,
        "pip_cache": {"path": cache, "writable": cache_writable, "listing": "NOT_AVAILABLE" if not cache_writable else "AVAILABLE"},
        "network": network,
        "installed_packages": installed,
    }


def validate_contract(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config["project"]["base_main_sha"] != EXPECTED_MAIN_SHA:
        errors.append("project main SHA drift")
    if config["project"]["parent_head_sha"] != EXPECTED_PARENT_HEAD:
        errors.append("parent PR head drift")
    upstream = config["upstream"]
    if upstream["commit_sha"] != EXPECTED_UPSTREAM_COMMIT:
        errors.append("upstream commit drift")
    if upstream["license_blob_sha"] != EXPECTED_LICENSE_BLOB:
        errors.append("license blob drift")
    if upstream["tracking_source_blob_sha"] != EXPECTED_TRACKING_BLOB:
        errors.append("tracking source blob drift")
    if upstream["source_version"] != EXPECTED_SOURCE_VERSION:
        errors.append("source version drift")
    return errors


def classify_install(env: dict[str, Any], install_returncode: int | None, install_stderr: str) -> str:
    if env["installed_packages"].get("mlflow") == EXPECTED_SOURCE_VERSION:
        return "EXECUTABLE_RUNTIME_CANDIDATE"
    if install_returncode is None:
        return "NOT_ATTEMPTED"
    if "Could not resolve host" in install_stderr or "Temporary failure in name resolution" in install_stderr:
        return "RETEST_RUNTIME_REQUIRED"
    return "BLOCKED_ENVIRONMENT"


def validate_tracking_uri(uri: str) -> bool:
    parsed = urllib.parse.urlparse(uri)
    if parsed.username or parsed.password:
        return False
    return parsed.scheme in {"file", "sqlite"}


def validate_metadata(metadata: dict[str, Any]) -> bool:
    forbidden = {"password", "passwd", "secret", "token", "api_key", "access_token", "authorization"}
    return not any(str(key).lower() in forbidden for key in metadata)


def run_real_probe(python_executable: Path) -> dict[str, Any]:
    code = r'''
import json
import tempfile
import time
import mlflow
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="mlflow-runtime-stress-") as td:
    uri = Path(td).as_uri()
    mlflow.set_tracking_uri(uri)
    timings = []
    outputs = []
    for index in range(2):
        start = time.perf_counter()
        with mlflow.start_run(run_name=f"stress-{index}"):
            mlflow.log_param("fixture", "open-source-bootstrap-stress-v1")
            mlflow.log_metric("value", 1.0)
            mlflow.set_tag("environment", "local-free")
            outputs.append({"params": {"fixture": "open-source-bootstrap-stress-v1"}, "metrics": {"value": 1.0}})
        timings.append(time.perf_counter() - start)
    print(json.dumps({"tracking_uri": uri, "outputs": outputs, "timings_seconds": timings}, sort_keys=True))
'''
    rc, stdout, stderr = run_capture([str(python_executable), "-c", code])
    if rc != 0:
        return {"status": "FAILED", "returncode": rc, "stderr": stderr[-4000:]}
    return {"status": "PASS", **json.loads(stdout)}


def attempt_install(root: Path) -> tuple[dict[str, Any], Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="mlflow-runtime-stress-"))
    venv_dir = temp_dir / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    python_bin = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirement = "mlflow @ git+https://github.com/mlflow/mlflow.git@" + EXPECTED_UPSTREAM_COMMIT
    started = time.perf_counter()
    rc, stdout, stderr = run_capture(
        [str(python_bin), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", requirement],
        cwd=root,
    )
    elapsed = time.perf_counter() - started
    freeze_rc, freeze_out, freeze_err = run_capture([str(python_bin), "-m", "pip", "freeze"])
    return (
        {
            "status": "PASS" if rc == 0 else "FAIL",
            "returncode": rc,
            "elapsed_seconds": round(elapsed, 3),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "freeze_returncode": freeze_rc,
            "freeze": freeze_out.splitlines(),
            "freeze_stderr_tail": freeze_err[-1000:],
            "venv_python": str(python_bin),
            "requirement": requirement,
        },
        temp_dir,
    )


def build_evidence(root: Path, attempt: dict[str, Any] | None, runtime: dict[str, Any] | None) -> dict[str, Any]:
    config = load_json(root / CONFIG_PATH)
    environment = detect_environment()
    contract_errors = validate_contract(config)
    install_stderr = attempt["stderr_tail"] if attempt else ""
    install_rc = attempt["returncode"] if attempt else None
    install_classification = classify_install(environment, install_rc, install_stderr)
    final_status = "EXPERIMENTAL_CANDIDATE" if runtime and runtime.get("status") == "PASS" else (
        "RETEST_RUNTIME_REQUIRED" if install_classification == "RETEST_RUNTIME_REQUIRED" else "BLOCKED_ENVIRONMENT"
    )
    evidence = {
        "schema_version": 1,
        "worker_id": "OPEN-SOURCE-BOOTSTRAP-STRESS-V1",
        "worker_issue": 781,
        "component": "MLFLOW_RUNTIME",
        "status": final_status,
        "live_binding": {
            "main_sha_at_claim": EXPECTED_MAIN_SHA,
            "parent_pr": 758,
            "parent_head_sha": EXPECTED_PARENT_HEAD,
            "upstream_commit_sha": EXPECTED_UPSTREAM_COMMIT,
            "license_blob_sha": EXPECTED_LICENSE_BLOB,
            "tracking_source_blob_sha": EXPECTED_TRACKING_BLOB,
        },
        "contract_validation": {"status": "PASS" if not contract_errors else "FAIL", "errors": contract_errors},
        "environment": environment,
        "installation": {
            "attempted": attempt is not None,
            "classification": install_classification,
            "dependency_lock_status": "INCOMPLETE_UPSTREAM_SOURCE_HAS_VERSION_RANGES_ONLY",
            "result": attempt or {},
        },
        "runtime": runtime or {"status": "NOT_EXECUTED", "reason": "exact dependency not installed"},
        "benchmark": {"executed": bool(runtime and runtime.get("status") == "PASS"), "upstream_claims_reused": False},
        "parity": {
            "project_contract_vs_runtime": "NOT_PROVEN" if not runtime else "REAL_RUNTIME_PROBE_EXECUTED",
            "same_inputs_required": True,
            "same_outputs_required": True,
            "unexplained_mismatch_blocks": True,
        },
        "adversarial": {
            "remote_uri_rejected": not validate_tracking_uri("https://example.invalid/mlruns"),
            "credential_uri_rejected": not validate_tracking_uri("sqlite://user:secret@example.invalid/db"),
            "secret_metadata_rejected": not validate_metadata({"api_key": "x"}),
            "file_uri_allowed": validate_tracking_uri("file:///tmp/mlruns"),
            "sqlite_uri_allowed": validate_tracking_uri("sqlite:///tmp/mlflow.db"),
        },
        "truth_boundary": {
            "foreign_weights_used": False,
            "canonical_base_changed": False,
            "model_training_executed": False,
            "paid_compute_used": False,
            "final_test_payload_read": False,
            "network_tracking_used": False,
        },
        "next_action": "Re-run with network/package access, generate and retain a fully pinned dependency lock with artifact hashes, exact commit install, real local-file smoke, deterministic export/reconstruction, and no-network proof before PARITY_PROVEN or ADOPTED.",
    }
    stable_identity = {
        "worker_id": evidence["worker_id"],
        "worker_issue": evidence["worker_issue"],
        "component": evidence["component"],
        "status": evidence["status"],
        "live_binding": evidence["live_binding"],
        "contract_validation": evidence["contract_validation"],
        "installation_policy": {
            "method": config["install"]["method"],
            "requirement": config["install"]["requirement"],
            "dependency_lock_status": evidence["installation"]["dependency_lock_status"],
        },
        "runtime_policy": config["runtime"],
        "adversarial": evidence["adversarial"],
        "truth_boundary": evidence["truth_boundary"],
    }
    evidence["evidence_identity_sha256"] = identity_sha256(stable_identity)
    evidence["evidence_sha256"] = identity_sha256(evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--attempt-install", action="store_true")
    parser.add_argument("--run-runtime", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/mlflow_runtime_bootstrap_stress_v1_evidence.json"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    attempt = None
    temp_dir = None
    runtime = None
    if args.attempt_install:
        attempt, temp_dir = attempt_install(root)
        if args.run_runtime and attempt["returncode"] == 0:
            runtime = run_real_probe(Path(attempt["venv_python"]))
    evidence = build_evidence(root, attempt, runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "evidence_sha256": evidence["evidence_sha256"]}, sort_keys=True))
    if temp_dir is not None and evidence["status"] != "EXPERIMENTAL_CANDIDATE":
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
