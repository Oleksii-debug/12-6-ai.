from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/linkedin/Liger-Kernel"
UPSTREAM_TAG = "v0.8.2"
UPSTREAM_COMMIT = "000be60929938fd1358e03524c6ab398b6d421bd"
UPSTREAM_LICENSE = "BSD-2-Clause"
UPSTREAM_LICENSE_BLOB = "d2fcc2b1c4384d0bcd1424b7f83db8e48fa753f6"
UPSTREAM_NOTICE_BLOB = "ea2881754f5b3e0eb9926dd9dc6c9d772f962911"
PYPI_SDIST_SHA256 = "387673ed6bf64fc8150cc8315fed578d2fc717ec3450f53489f480880223c1b8"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def command_version(command: str, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            [command, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (proc.stdout or proc.stderr).strip().splitlines()
    return output[0] if output else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def detect_environment() -> dict[str, Any]:
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu": platform.processor(),
        "gpu": {
            "nvidia_smi": command_version("nvidia-smi", "-L"),
            "rocm_smi": command_version("rocm-smi", "--showproductname"),
        },
        "torch": torch_info,
        "packages": {
            "torch": package_version("torch"),
            "triton": package_version("triton"),
            "liger-kernel": package_version("liger-kernel"),
        },
        "package_manager": {
            "pip": command_version(sys.executable, "-m", "pip", "--version"),
            "uv": command_version("uv", "--version"),
            "poetry": command_version("poetry", "--version"),
            "pdm": command_version("pdm", "--version"),
            "conda": command_version("conda", "--version"),
            "git": command_version("git", "--version"),
        },
        "network": {
            "pypi_dns": subprocess.run(
                ["getent", "hosts", "pypi.org"], check=False, capture_output=True, text=True
            ).returncode
            == 0,
            "github_dns": subprocess.run(
                ["getent", "hosts", "github.com"], check=False, capture_output=True, text=True
            ).returncode
            == 0,
        },
        "cache_hints": {
            "local_matches": sorted(
                str(p)
                for root in [Path("/root/.cache"), Path("/tmp"), Path("/opt")]
                if root.exists()
                for p in root.rglob("*")
                if any(token in p.name.lower() for token in ("liger", "triton"))
            )[:50]
        },
    }


def runtime_probe() -> dict[str, Any]:
    env = detect_environment()
    observed = env["packages"]
    cuda_available = bool(env["torch"].get("cuda_available", False))
    if observed.get("liger-kernel") != "0.8.2":
        return {
            "status": "NOT_EXECUTED",
            "reason": "EXACT_LIGER_DEPENDENCY_ABSENT_OR_DRIFTED",
            "observed": {
                "liger-kernel": observed.get("liger-kernel"),
                "triton": observed.get("triton"),
                "cuda": cuda_available,
            },
        }
    if observed.get("triton") is None or not cuda_available:
        return {
            "status": "NOT_EXECUTED",
            "reason": "TRITON_GPU_RUNTIME_UNAVAILABLE",
            "observed": {
                "liger-kernel": observed.get("liger-kernel"),
                "triton": observed.get("triton"),
                "cuda": cuda_available,
            },
        }

    try:
        ops = importlib.import_module("liger_kernel.ops")
    except Exception as exc:
        return {"status": "FAIL", "reason": "LIGER_IMPORT_FAILED", "error": repr(exc)}

    expected = [
        "LigerRMSNormFunction",
        "LigerRopeFunction",
        "LigerSiLUMulFunction",
        "LigerCrossEntropyFunction",
    ]
    missing = [name for name in expected if not hasattr(ops, name)]
    if missing:
        return {"status": "FAIL", "reason": "EXPECTED_OPERATOR_EXPORTS_MISSING", "missing": missing}
    return {"status": "READY_FOR_REAL_GPU_PROBE", "reason": "PREREQUISITES_AVAILABLE"}


def build_evidence(manifest_path: Path, install_attempts: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    env = detect_environment()
    evidence = {
        "schema_version": 1,
        "worker_id": "OPEN-SOURCE-BOOTSTRAP-STRESS-V1",
        "lane_key": "D08|LIGER-KERNEL|OPEN-SOURCE-REUSE-RESEARCH|BOOTSTRAP-STRESS-V1",
        "project_base_sha": manifest["project"]["base_sha"],
        "upstream": manifest["upstream"],
        "rights": manifest["rights"],
        "environment": env,
        "install_attempts": install_attempts,
        "runtime_probe": runtime_probe(),
        "benchmark": {
            "executed": False,
            "runs": 0,
            "reason": "REAL_LIGER_GPU_RUNTIME_NOT_AVAILABLE",
            "test_double_evidence": False,
        },
        "parity": {
            "proven": False,
            "reason": "NO_REAL_UPSTREAM_RUNTIME_EXECUTION",
            "required_ops": manifest["qualification"]["required_ops"],
        },
        "canonical_base_integrity": {
            "foreign_weights_used": False,
            "tokenizer_changed": False,
            "checkpoint_changed": False,
            "training_executed": False,
            "paid_compute_used": False,
        },
        "promotion_state": "RETEST_RUNTIME_REQUIRED",
        "environment_hash": canonical_sha256(env),
    }
    evidence["evidence_identity"] = canonical_sha256(evidence)
    output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return evidence
PY