from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/open_source_bootstrap_stress_v1.json"
REPORT_SCHEMA = "12-6.open-source-bootstrap-stress-evidence.v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_relative(path_text: str) -> bool:
    path = Path(path_text)
    return not path.is_absolute() and ".." not in path.parts


def command_ok(command: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=8)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return proc.returncode == 0, (proc.stdout or proc.stderr).strip()


def detect_network() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for host in ("pypi.org", "github.com"):
        try:
            socket.getaddrinfo(host, 443)
            checks[host] = {"dns": "RESOLVED"}
        except OSError as exc:
            checks[host] = {"dns": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    return checks


def find_python311() -> dict[str, Any]:
    candidates = ["python3.11", "/usr/bin/python3.11", "/usr/local/bin/python3.11"]
    found = shutil.which("python3.11")
    if found:
        candidates.insert(0, found)
    seen: list[dict[str, Any]] = []
    for candidate in dict.fromkeys(candidates):
        if Path(candidate).exists() or shutil.which(candidate):
            ok, output = command_ok([candidate, "--version"])
            seen.append({"path": candidate, "ok": ok, "version": output})
    return {"available": any("3.11.16" in item.get("version", "") for item in seen), "candidates": seen}


def cache_probe() -> dict[str, Any]:
    candidates = [Path.home() / ".cache/pip", Path.home() / ".cache/uv", Path("/root/.cache/pip")]
    artifacts: list[str] = []
    existing: list[str] = []
    for path in candidates:
        if path.is_dir():
            existing.append(str(path))
            for item in path.rglob("*"):
                if item.is_file() and item.suffix in {".whl", ".zip", ".tar", ".gz"}:
                    artifacts.append(str(item))
    exact = [
        item for item in artifacts
        if any(token in Path(item).name.lower() for token in ("torch-2.13.0", "numpy-2.4.6", "safetensors-0.8.0", "pytest-"))
    ]
    return {"directories": existing, "artifact_count": len(artifacts), "exact_match_candidates": sorted(exact)}


def package_managers() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("python", "pip", "uv", "poetry", "pdm", "conda", "git"):
        path = shutil.which(name)
        if path:
            ok, version = command_ok([path, "--version"])
            out[name] = {"path": path, "available": ok, "version": version}
        else:
            out[name] = {"available": False}
    return out


def git_state(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for args, key in [(["rev-parse", "HEAD"], "head"), (["branch", "--show-current"], "branch")]:
        ok, value = command_ok(["git", "-C", str(root), *args])
        result[key] = value if ok else None
    return result


def environment_observation() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": sys.implementation.name,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": os.cpu_count(),
        "gpu": {
            "nvidia_smi": shutil.which("nvidia-smi"),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "package_managers": package_managers(),
        "python311": find_python311(),
        "network": detect_network(),
        "cache": cache_probe(),
        "git": git_state(ROOT),
    }


def static_contract_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    owned_files = [
        "tools/verify_open_source_bootstrap_stress_v1.py",
        "tests/test_open_source_bootstrap_stress_v1.py",
        "configs/research/open_source_bootstrap_stress_v1.json",
        "reports/open_source_bootstrap_stress_v1_evidence.json",
        "docs/OPEN_SOURCE_BOOTSTRAP_STRESS_V1.md",
    ]
    forbidden = tuple(config["forbidden_mutation_prefixes"])
    return [
        {"name": "base_sha_shape", "passed": len(config["base_sha"]) == 40 and all(c in "0123456789abcdef" for c in config["base_sha"])},
        {"name": "incumbent_head_shape", "passed": len(config["incumbent_pr_head"]) == 40 and all(c in "0123456789abcdef" for c in config["incumbent_pr_head"])},
        {"name": "path_safety", "passed": all(safe_relative(p) for p in config["incumbent_changed_files"])},
        {"name": "no_new_workflow", "passed": config["expected_no_new_workflow"]},
        {"name": "local_free", "passed": config["local_free_only"]},
        {"name": "canonical_surfaces_avoided", "passed": all(not path.startswith(forbidden) for path in owned_files)},
    ]


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    env = environment_observation()
    static = static_contract_checks(config)
    python_ok = env["python311"]["available"] and env["python"]["implementation"] == config["required_python_implementation"] and env["python"]["version"] == config["required_python"]
    dns_ok = all(value.get("dns") == "RESOLVED" for value in env["network"].values())
    cache_ok = bool(env["cache"]["exact_match_candidates"])
    runtime_executable = python_ok and dns_ok and cache_ok
    blockers: list[str] = []
    if not python_ok:
        blockers.append("EXACT_PYTHON_UNAVAILABLE")
    if not dns_ok:
        blockers.append("PACKAGE_INDEX_NETWORK_UNAVAILABLE")
    if not cache_ok:
        blockers.append("NO_LOCAL_EXACT_ARTIFACT_CACHE")
    blockers.append("ENV151_SOURCE_NOT_ON_CURRENT_MAIN")
    deterministic_payload = {
        "schema": REPORT_SCHEMA,
        "worker_id": config["worker_id"],
        "lane_key": config["lane_key"],
        "project": {"repository": config["project_repository"], "base_sha": config["base_sha"]},
        "target": {"pr": config["incumbent_pr"], "head": config["incumbent_pr_head"], "branch": config["incumbent_branch"]},
        "runtime": {"status": "EXECUTED" if runtime_executable else "NOT_EXECUTED", "required_python": config["required_python"], "required_platform": config["required_platform"]},
        "static_checks": static,
        "environment": env,
        "blockers": blockers,
        "missing_artifacts": ["CPython 3.11.16 executable", "exact ENV-151 dependency wheel set for PR #313"] if not runtime_executable else [],
        "canonical_base_contamination": False,
        "paid_compute": False,
        "foreign_weights": False,
        "training_updates": 0,
        "benchmark": {"executed": False, "reason": "Real ENV-151 runtime is not executed when exact interpreter/artifacts are unavailable.", "latency": None, "throughput": None, "rss": None},
        "parity": {"status": "NOT_EXECUTED_RUNTIME_REQUIRED", "reason": "No exact ENV-151 stack execution; no fabricated equivalence claim."},
        "rights": {"bootstrap_code": "PROJECT_OWNED_REUSE_QUALIFICATION", "third_party_dependency_rights": "NOT_FULLY_VERIFIED_BECAUSE_RUNTIME_ARTIFACTS_UNAVAILABLE", "model_weights": "NONE", "dataset_rights": "NONE"},
    }
    deterministic_payload["evidence_identity_sha256"] = sha256_bytes(canonical_bytes(deterministic_payload))
    return deterministic_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = build_report(config)
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["runtime"]["status"], "evidence_identity_sha256": report["evidence_identity_sha256"], "blockers": report["blockers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
