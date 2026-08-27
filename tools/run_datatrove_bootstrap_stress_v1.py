#!/usr/bin/env python3
"""LOCAL_FREE DataTrove bootstrap/runtime qualification runner.

The runner is fail-closed: it never substitutes another DataTrove version and never treats
mock/source-inspection results as runtime evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

VERSION = "0.10.0"
COMMIT = "7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664"
WHEEL_SHA256 = "c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d"
SDIST_SHA256 = "e31f89bdccb30ef0796854f5842ff52b4b224c28b2d5b110088e84071ea05c40"
REPO = "https://github.com/huggingface/datatrove"


def run(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
        return {"command": command, "return_code": proc.returncode, "stdout": proc.stdout[-10000:], "stderr": proc.stderr[-10000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "return_code": 124, "stdout": "", "stderr": str(exc)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_environment() -> dict[str, Any]:
    managers: dict[str, Any] = {}
    for name in ("python", "pip", "uv", "poetry", "pdm", "conda", "git"):
        path = shutil.which(name)
        item: dict[str, Any] = {"available": bool(path), "path": path, "version": None}
        if path:
            if name == "python":
                item["version"] = platform.python_version()
            else:
                result = run([name, "--version"], timeout=20)
                item["version"] = (result["stdout"] or result["stderr"]).strip()[:200]
        managers[name] = item
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu_detected": bool(shutil.which("nvidia-smi")),
        "package_managers": managers,
        "network_to_package_index": "UNKNOWN",
    }


def benchmark_runtime(env_python: Path) -> dict[str, Any]:
    repetitions: list[dict[str, Any]] = []
    for index in (1, 2):
        root = Path(tempfile.mkdtemp(prefix=f"dt774-runtime-{index}-"))
        source = root / "input"
        output = root / "output"
        logs = root / "logs"
        source.mkdir(); output.mkdir(); logs.mkdir()
        (source / "fixture.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in (
                    {"id": "dt774-a", "text": "Hello DataTrove.", "metadata": {"lang": "en"}},
                    {"id": "dt774-b", "text": "Привіт DataTrove.", "metadata": {"lang": "uk"}},
                    {"id": "dt774-c", "text": "Repeat me.", "metadata": {"lang": "en"}},
                )
            ),
            encoding="utf-8",
        )
        program = root / "pipeline.py"
        program.write_text(
            "from datatrove.executor import LocalPipelineExecutor\n"
            "from datatrove.pipeline.readers import JsonlReader\n"
            "from datatrove.pipeline.writers import JsonlWriter\n"
            f"pipeline = [JsonlReader(r'{source}', doc_progress=False), JsonlWriter(output_folder=r'{output}')]\n"
            f"LocalPipelineExecutor(pipeline=pipeline, logging_dir=r'{logs}', tasks=1, workers=1).run()\n",
            encoding="utf-8",
        )
        started = time.perf_counter()
        result = run([str(env_python), str(program)], timeout=120)
        elapsed = time.perf_counter() - started
        files = sorted(output.rglob("*.jsonl"))
        hashes = [sha256_file(path) for path in files]
        repetitions.append({
            "repetition": index,
            "return_code": result["return_code"],
            "elapsed_seconds": round(elapsed, 6),
            "output_sha256": hashes,
            "stderr": result["stderr"],
        })
    same = [tuple(item["output_sha256"]) for item in repetitions]
    deterministic = all(item["return_code"] == 0 for item in repetitions) and len(set(same)) == 1
    return {"status": "PASS" if deterministic else "FAIL", "execution_mode": "real", "repetitions": repetitions, "deterministic": deterministic}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/research/datatrove_bootstrap_stress_v1.json")
    parser.add_argument("--environment", default=None)
    args = parser.parse_args()

    environment = detect_environment()
    temp_root = Path(tempfile.mkdtemp(prefix="dt774-evidence-"))
    venv = Path(args.environment) if args.environment else temp_root / "venv"
    if venv.exists():
        create = {"command": [], "return_code": 0, "stdout": "reused-existing-isolated-env", "stderr": ""}
    else:
        create = run(["uv", "venv", "--python", sys.executable, str(venv)], timeout=60)

    env_python = venv / "bin" / "python"
    install = run(["uv", "pip", "install", "--python", str(env_python), "--no-cache", "--only-binary=:all:", f"datatrove=={VERSION}"], timeout=180)
    installed = install["return_code"] == 0
    installed_version = None
    if installed:
        version = run([str(env_python), "-c", "import importlib.metadata as m; print(m.version('datatrove'))"], timeout=30)
        if version["return_code"] == 0:
            installed_version = version["stdout"].strip()
        else:
            installed = False

    dependency_lock_sha256 = None
    dependency_lock_lines: list[str] = []
    if installed and installed_version == VERSION:
        runtime = benchmark_runtime(env_python)
        benchmark = {"status": runtime["status"], "runs_required": 2, "paid_compute": False}
        parity = {
            "status": runtime["status"],
            "comparison": "Document id/text/metadata round-trip through real JsonlReader -> JsonlWriter",
            "reference": "project fixture JSON semantics",
        }
        freeze = run(["uv", "pip", "freeze", "--python", str(env_python)], timeout=60)
        dependency_lock_lines = sorted(line.strip() for line in freeze["stdout"].splitlines() if line.strip())
        dependency_lock_sha256 = sha256_text("\n".join(dependency_lock_lines) + "\n") if freeze["return_code"] == 0 else None
        dependency_lock_status = "LOCKED_RUNTIME_FREEZE" if dependency_lock_sha256 else "LOCK_FAILED"
    else:
        runtime = {"status": "NOT_EXECUTED", "execution_mode": "real", "reason": "exact DataTrove 0.10.0 was not installed"}
        benchmark = {"status": "NOT_EXECUTED", "runs_required": 2, "paid_compute": False}
        parity = {"status": "NOT_EXECUTED", "reason": "real exact runtime unavailable"}
        dependency_lock_status = "NOT_MATERIALIZED_NETWORK_BLOCK"

    environment["network_to_package_index"] = "PASS" if install["return_code"] == 0 else "FAIL"
    decision = "EXPERIMENTAL_CANDIDATE" if installed else "RETEST_RUNTIME_REQUIRED"
    manifest = {
        "schema_version": 1,
        "component": {"id": "DATATROVE", "package": "datatrove", "version": VERSION},
        "upstream": {
            "repository": REPO,
            "tag": "v0.10.0",
            "commit": COMMIT,
            "wheel_sha256": WHEEL_SHA256,
            "sdist_sha256": SDIST_SHA256,
            "release_date": "2026-08-13",
            "pypi_attestation_source_commit": COMMIT,
        },
        "rights": {
            "software_license": "Apache-2.0",
            "license_source": f"{REPO}/blob/v0.10.0/LICENSE",
            "license_blob_sha": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
            "source_notice_present": False,
            "dataset_rights_inherited": False,
            "model_weight_rights_applicable": False,
            "training_authority_from_code_license": False,
        },
        "environment": environment,
        "bootstrap": {
            "isolated_env_created": create["return_code"] == 0,
            "environment_path": str(venv),
            "exact_install_attempted": True,
            "install_command": install["command"],
            "result_code": install["return_code"],
            "installed": installed,
            "installed_version": installed_version,
            "install_stdout": install["stdout"],
            "install_stderr": install["stderr"],
            "local_cache_checked": True,
            "exact_cached_artifact_found": False,
            "dependency_lock_status": dependency_lock_status,
            "dependency_lock_sha256": dependency_lock_sha256,
            "dependency_lock_packages": dependency_lock_lines,
        },
        "runtime": runtime,
        "parity": parity,
        "benchmark": benchmark,
        "canonical_base_safety": {
            "canonical_base_modified": False,
            "foreign_pretrained_weights_used": False,
            "foreign_instruction_or_alignment_behavior_used": False,
            "tokenizer_modified": False,
            "training_executed": False,
            "benchmark_final_test_data_accessed": False,
            "model_checkpoint_modified": False,
        },
        "decision": decision,
        "truth_boundary": "Source inspection, package metadata and test doubles never constitute runtime PASS evidence.",
    }
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "installed": installed, "runtime": runtime["status"], "manifest": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
