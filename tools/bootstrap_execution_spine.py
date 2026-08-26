#!/usr/bin/env python3
"""Install one exact research execution environment from committed 12-6 locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv
from typing import Any

CUDA_PREFIXES = ("nvidia-", "cuda-", "triton")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _run(argv: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    cmd = [str(item) for item in argv]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _resolve(repo_root: Path, purpose: str) -> dict[str, Any]:
    canonical_index_path = repo_root / "requirements/locks/index.json"
    canonical_index = _load_json(canonical_index_path)
    canonical_record = canonical_index.get("profiles", {}).get(purpose)
    if canonical_record is not None:
        profile_path = repo_root / canonical_record["path"]
        profile = _load_json(profile_path)
        if _sha256(profile_path) != canonical_record["sha256"]:
            raise RuntimeError(f"canonical profile file hash mismatch: {purpose}")
        return {
            "kind": "canonical",
            "purpose": purpose,
            "profile": profile,
            "profile_path": profile_path,
            "base_profile": profile,
            "base_profile_path": profile_path,
            "overlay_paths": [],
            "canonical_index": canonical_index,
            "canonical_index_path": canonical_index_path,
        }

    purpose_index_path = repo_root / "requirements/profiles/index.json"
    purpose_index = _load_json(purpose_index_path)
    purpose_record = purpose_index.get("profiles", {}).get(purpose)
    if purpose_record is None:
        choices = sorted(
            set(canonical_index.get("profiles", {})) | set(purpose_index.get("profiles", {}))
        )
        raise RuntimeError(f"unknown purpose {purpose!r}; choose one of {choices}")

    profile_path = repo_root / purpose_record["path"]
    profile = _load_json(profile_path)
    if _sha256(profile_path) != purpose_record["sha256"]:
        raise RuntimeError(f"purpose profile file hash mismatch: {purpose}")
    base_ref = profile.get("base_profile")
    if not isinstance(base_ref, dict):
        raise RuntimeError(f"purpose profile {purpose} has no base_profile")
    base_profile_path = repo_root / base_ref["path"]
    base_profile = _load_json(base_profile_path)
    if _sha256(base_profile_path) != base_ref["file_sha256"]:
        raise RuntimeError(f"purpose profile {purpose} base profile hash mismatch")
    if base_profile.get("manifest_sha256") != base_ref["manifest_sha256"]:
        raise RuntimeError(f"purpose profile {purpose} base manifest mismatch")

    overlay_paths: list[Path] = []
    for lock in profile.get("locks", {}).values():
        path = repo_root / lock["path"]
        if _sha256(path) != lock["sha256"]:
            raise RuntimeError(f"purpose overlay hash mismatch: {path}")
        overlay_paths.append(path)

    return {
        "kind": "purpose",
        "purpose": purpose,
        "profile": profile,
        "profile_path": profile_path,
        "base_profile": base_profile,
        "base_profile_path": base_profile_path,
        "overlay_paths": overlay_paths,
        "canonical_index": canonical_index,
        "canonical_index_path": canonical_index_path,
        "purpose_index": purpose_index,
        "purpose_index_path": purpose_index_path,
    }


def _validate(repo_root: Path, resolved: dict[str, Any]) -> None:
    base_profile_id = resolved["base_profile"]["profile_id"]
    _run(
        [
            sys.executable,
            "tools/verify_locked_environment.py",
            "--profile",
            base_profile_id,
            "--validate-only",
        ],
        cwd=repo_root,
    )
    if resolved["kind"] == "purpose":
        _run(
            [
                sys.executable,
                "tools/verify_purpose_environment.py",
                "--profile",
                resolved["purpose"],
                "--root",
                repo_root,
                "--validate-only",
            ],
            cwd=repo_root,
        )


def _lock_paths(repo_root: Path, base_profile: dict[str, Any], with_dev: bool) -> list[Path]:
    keys = ["toolchain", "runtime"]
    if with_dev:
        keys.append("dev")
    paths = []
    for key in keys:
        record = base_profile["locks"].get(key)
        if record is None:
            if key == "dev":
                raise RuntimeError("selected base profile has no dev lock")
            raise RuntimeError(f"selected base profile has no {key} lock")
        path = repo_root / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"{key} lock hash mismatch: {path}")
        paths.append(path)
    return paths


def _install_lock(python: Path, lock: Path, repo_root: Path) -> None:
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--require-hashes",
            "-r",
            lock,
        ],
        cwd=repo_root,
    )


def _install_project(python: Path, repo_root: Path) -> None:
    offline = os.environ.copy()
    offline["PIP_NO_INDEX"] = "1"
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            ".",
        ],
        cwd=repo_root,
        env=offline,
    )


def _probe(
    python: Path,
    repo_root: Path,
    *,
    purpose: str,
    with_dev: bool,
) -> dict[str, Any]:
    code = r"""
import importlib.metadata
import json
import platform

import torch
import twelve_six

rows = []
for dist in importlib.metadata.distributions():
    name = (dist.metadata.get("Name") or "").lower().replace("_", "-").replace(".", "-")
    version = dist.version
    if name:
        rows.append({"name": name, "version": version})
rows.sort(key=lambda item: (item["name"], item["version"]))

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "project_import": "PASS",
    "installed": rows,
}
for module in ("pytest", "ruff", "tokenizers", "transformers"):
    try:
        payload[module] = importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        payload[module] = None
print(json.dumps(payload, sort_keys=True))
"""
    proc = subprocess.run(
        [str(python), "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    if with_dev and (payload["pytest"] is None or payload["ruff"] is None):
        raise RuntimeError("dev bootstrap did not materialize pytest and ruff")
    if purpose.endswith("tokenizer-experiment") and payload["tokenizers"] is None:
        raise RuntimeError("tokenizer purpose did not materialize tokenizers")
    if purpose.endswith("transformers-interop") and payload["transformers"] is None:
        raise RuntimeError("transformers purpose did not materialize transformers")
    if purpose.endswith("cuda-training"):
        if payload["torch_cuda"] != "13.0":
            raise RuntimeError(f"CUDA purpose expected torch CUDA 13.0, got {payload['torch_cuda']!r}")
        payload["gpu_execution"] = (
            "AVAILABLE_NOT_RUN" if payload["cuda_available"] else "NO_GPU_PREFLIGHT_PASS"
        )
    else:
        payload["gpu_execution"] = "NOT_APPLICABLE"

    cuda_payloads = sorted(
        {
            row["name"]
            for row in payload["installed"]
            if row["name"].startswith(CUDA_PREFIXES)
        }
    )
    payload["cuda_payloads"] = cuda_payloads
    return payload


def preflight(
    *,
    repo_root: Path,
    purpose: str,
    with_dev: bool,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved = _resolve(repo_root, purpose)
    expected_python = resolved["profile"]["python"]["version"]
    current_python = platform.python_version()
    if platform.python_implementation().lower() != "cpython":
        raise RuntimeError("execution spine requires CPython")
    if current_python != expected_python:
        raise RuntimeError(f"purpose requires Python {expected_python}, running {current_python}")
    _validate(repo_root, resolved)
    lock_paths = _lock_paths(repo_root, resolved["base_profile"], with_dev)
    lock_paths.extend(resolved["overlay_paths"])
    return {
        "purpose": purpose,
        "purpose_kind": resolved["kind"],
        "python": expected_python,
        "with_dev": with_dev,
        "profile_path": resolved["profile_path"].relative_to(repo_root).as_posix(),
        "profile_sha256": _sha256(resolved["profile_path"]),
        "locks": [
            {"path": path.relative_to(repo_root).as_posix(), "sha256": _sha256(path)}
            for path in lock_paths
        ],
    }


def bootstrap(
    *,
    repo_root: Path,
    purpose: str,
    venv_path: Path,
    with_dev: bool,
    source_sha: str | None,
    evidence_out: Path | None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if platform.python_implementation().lower() != "cpython":
        raise RuntimeError("execution spine requires CPython")
    resolved = _resolve(repo_root, purpose)
    expected_python = resolved["profile"]["python"]["version"]
    current_python = platform.python_version()
    if current_python != expected_python:
        raise RuntimeError(f"purpose requires Python {expected_python}, running {current_python}")

    _validate(repo_root, resolved)

    target = venv_path if venv_path.is_absolute() else repo_root / venv_path
    if target.exists():
        shutil.rmtree(target)
    venv.EnvBuilder(with_pip=True, clear=True).create(target)
    python = _venv_python(target)

    installed_lock_paths = _lock_paths(repo_root, resolved["base_profile"], with_dev)
    for lock in installed_lock_paths:
        _install_lock(python, lock, repo_root)
    for overlay in resolved["overlay_paths"]:
        _install_lock(python, overlay, repo_root)
    _install_project(python, repo_root)
    probe = _probe(python, repo_root, purpose=purpose, with_dev=with_dev)

    lock_records = [
        {"path": path.relative_to(repo_root).as_posix(), "sha256": _sha256(path)}
        for path in [*installed_lock_paths, *resolved["overlay_paths"]]
    ]
    evidence: dict[str, Any] = {
        "schema_version": "12-6.execution-spine-bootstrap.v1",
        "authority": "COMMITTED_HASH_LOCKS",
        "source_sha": source_sha or "UNBOUND_LOCAL",
        "purpose": purpose,
        "purpose_kind": resolved["kind"],
        "python": expected_python,
        "venv": target.relative_to(repo_root).as_posix()
        if target.is_relative_to(repo_root)
        else str(target),
        "with_dev": with_dev,
        "locks": lock_records,
        "profile": {
            "path": resolved["profile_path"].relative_to(repo_root).as_posix(),
            "sha256": _sha256(resolved["profile_path"]),
        },
        "runtime_probe": probe,
        "verification": {
            "committed_metadata_validation": "PASS",
            "exact_hash_install": "PASS",
            "project_editable_install": "PASS",
            "pytest_ruff_available": "PASS" if with_dev else "NOT_REQUESTED",
            "gpu_execution": probe["gpu_execution"],
            "cpu_minimality": (
                "NOT_ASSERTED_CANONICAL_RUNTIME_CONTAINS_CUDA_PAYLOADS"
                if probe["cuda_payloads"] and not purpose.endswith("cuda-training")
                else "NOT_APPLICABLE"
            ),
        },
    }
    evidence["evidence_sha256"] = hashlib.sha256(_canonical_bytes(evidence)).hexdigest()
    if evidence_out is not None:
        path = evidence_out if evidence_out.is_absolute() else repo_root / evidence_out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purpose", default="linux-x86_64")
    parser.add_argument("--venv", type=Path, default=Path(".research-env"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--with-dev", action="store_true")
    parser.add_argument("--source-sha")
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight(repo_root=args.repo_root, purpose=args.purpose, with_dev=args.with_dev)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    evidence = bootstrap(
        repo_root=args.repo_root,
        purpose=args.purpose,
        venv_path=args.venv,
        with_dev=args.with_dev,
        source_sha=args.source_sha,
        evidence_out=args.evidence_out,
    )
    print(f"purpose={evidence['purpose']}")
    print(f"venv={evidence['venv']}")
    print(f"evidence_sha256={evidence['evidence_sha256']}")
    print(f"gpu_execution={evidence['verification']['gpu_execution']}")
    print(f"cpu_minimality={evidence['verification']['cpu_minimality']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
