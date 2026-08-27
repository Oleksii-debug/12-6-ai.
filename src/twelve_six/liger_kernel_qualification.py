"""Fail-closed qualification contract for Liger Kernel.

This module deliberately contains no Liger implementation and imports no foreign
model weights.  Real backend execution is optional and only allowed when the
exact pinned package is installed in the caller's isolated environment and a
compatible accelerator is available.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

UPSTREAM_REPOSITORY = "https://github.com/linkedin/Liger-Kernel"
UPSTREAM_TAG = "v0.8.2"
UPSTREAM_COMMIT = "000be60929938fd1358e03524c6ab398b6d421bd"
PYPI_PACKAGE = "liger-kernel"
PYPI_VERSION = "0.8.2"
PYPI_SDIST_SHA256 = "387673ed6bf64fc8150cc8315fed578d2fc717ec3450f53489f480880223c1b8"
LICENSE = "BSD-2-Clause"
LICENSE_BLOB_SHA = "d2fcc2b1c4384d0bcd1424b7f83db8e48fa753f6"
SUPPORTED_OPERATORS = ("RMSNorm", "RoPE", "SwiGLU", "cross_entropy")


@dataclass(frozen=True)
class Environment:
    python: str
    os: str
    arch: str
    gpu_present: bool
    gpu_description: str | None
    package_manager: str
    exact_dependency_installed: bool
    dependency_version: str | None
    network_pypi: bool
    network_github: bool


@dataclass(frozen=True)
class QualificationResult:
    promotion_state: str
    runtime_state: str
    parity_proven: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "promotion_state": self.promotion_state,
            "runtime_state": self.runtime_state,
            "parity_proven": self.parity_proven,
            "reasons": list(self.reasons),
        }


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "component",
        "upstream",
        "license",
        "artifact",
        "allowed_operators",
        "canonical_base_dependency",
        "promotion_state",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"missing manifest fields: {missing}")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if manifest["component"] != "LIGER_KERNEL":
        raise ValueError("wrong component identity")
    upstream = manifest["upstream"]
    for key in ("repository", "tag", "commit"):
        _require_nonempty(upstream.get(key), f"upstream.{key}")
    if upstream["repository"] != UPSTREAM_REPOSITORY:
        raise ValueError("unexpected upstream repository")
    if upstream["tag"] != UPSTREAM_TAG or upstream["commit"] != UPSTREAM_COMMIT:
        raise ValueError("upstream identity drift")
    if manifest["license"] != LICENSE:
        raise ValueError("license drift")
    artifact = manifest["artifact"]
    _require_nonempty(artifact.get("name"), "artifact.name")
    digest = artifact.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("artifact.sha256 must be lowercase SHA-256")
    allowed = manifest["allowed_operators"]
    if set(allowed) != set(SUPPORTED_OPERATORS):
        raise ValueError("operator allow-list drift")
    if manifest["canonical_base_dependency"] is not False:
        raise ValueError("Liger must remain outside canonical Base lineage")
    if manifest["promotion_state"] not in {"DISCOVERED", "CANDIDATE", "PARITY_PROVEN", "ADOPTED"}:
        raise ValueError("invalid promotion state")
    if manifest["promotion_state"] in {"PARITY_PROVEN", "ADOPTED"}:
        if manifest.get("backend_execution") != "EXECUTED_PASS":
            raise ValueError("promotion beyond CANDIDATE requires real backend execution")
        if manifest.get("parity_proven") is not True:
            raise ValueError("promotion beyond CANDIDATE requires parity_proven=true")
        if manifest.get("gpu_evidence") is not True:
            raise ValueError("Liger kernel parity requires GPU evidence")


def deterministic_probe_input(
    operator: str, shape: Sequence[int], dtype: str = "float32"
) -> dict[str, Any]:
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError("unsupported operator")
    if not shape or any(
        not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape
    ):
        raise ValueError("shape must contain positive integers")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported dtype")
    return {"operator": operator, "shape": list(shape), "dtype": dtype}


def repeatability_digest(probe: Mapping[str, Any]) -> str:
    if not isinstance(probe, Mapping):
        raise TypeError("probe must be a mapping")
    return canonical_sha256(dict(probe))


def compare_numeric_sequences(
    reference: Sequence[float], candidate: Sequence[float], atol: float, rtol: float
) -> bool:
    if len(reference) != len(candidate):
        return False
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol/rtol must be finite and non-negative")
    for a, b in zip(reference, candidate):
        if not math.isfinite(a) or not math.isfinite(b):
            return False
        if abs(a - b) > atol + rtol * abs(a):
            return False
    return True


def detect_environment() -> Environment:
    manager = "none"
    for candidate in ("uv", "poetry", "pdm", "conda", "pip"):
        if shutil.which(candidate):
            manager = candidate
            break
    gpu_present = False
    gpu_description = None
    try:
        import torch  # type: ignore

        gpu_present = bool(
            torch.cuda.is_available()
            or (getattr(torch, "version", None) and torch.version.hip)
        )
        if torch.cuda.is_available():
            gpu_description = torch.cuda.get_device_name(0)
        elif torch.version.hip:
            gpu_description = f"ROCm:{torch.version.hip}"
    except Exception:
        pass
    try:
        import importlib.metadata as metadata

        version = metadata.version(PYPI_PACKAGE)
    except Exception:
        version = None
    return Environment(
        python=sys.version.split()[0],
        os=platform.system(),
        arch=platform.machine(),
        gpu_present=gpu_present,
        gpu_description=gpu_description,
        package_manager=manager,
        exact_dependency_installed=version == PYPI_VERSION,
        dependency_version=version,
        network_pypi=False,
        network_github=False,
    )


def real_runtime_preflight() -> QualificationResult:
    """Return a truthful runtime state without claiming execution when unavailable."""
    available = importlib.util.find_spec("liger_kernel") is not None
    env = detect_environment()
    if not available or not env.exact_dependency_installed:
        return QualificationResult(
            promotion_state="CANDIDATE",
            runtime_state="NOT_EXECUTED_DEPENDENCY_ABSENT",
            parity_proven=False,
            reasons=("exact pinned liger-kernel==0.8.2 is not installed",),
        )
    if not env.gpu_present:
        return QualificationResult(
            promotion_state="CANDIDATE",
            runtime_state="NOT_EXECUTED_GPU_ABSENT",
            parity_proven=False,
            reasons=("compatible accelerator evidence is absent",),
        )
    return QualificationResult(
        promotion_state="CANDIDATE",
        runtime_state="READY_FOR_REAL_PARITY_RUN",
        parity_proven=False,
        reasons=(
            "exact dependency and accelerator are available; "
            "operator-level parity remains unexecuted",
        ),
    )


def write_result(path: Path, result: QualificationResult) -> str:
    payload = result.as_dict()
    payload["result_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload["result_sha256"]
