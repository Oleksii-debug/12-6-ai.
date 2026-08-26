#!/usr/bin/env python3
"""Measure cold exact-lock download/setup versus verified warm wheelhouse setup."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import Any

from twelve_six.integration.hashed_dependency_cache import (
    DependencyCacheError,
    validate_manifest_files,
    verify_wheelhouse,
)

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DependencyCacheError(f"JSON must contain an object: {path}")
    return value


def _venv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def _run(command: list[str | Path], *, cwd: Path) -> None:
    env = dict(os.environ)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)


def _lock_paths(root: Path, manifest: dict[str, Any]) -> list[Path]:
    result = []
    for record in manifest["component_locks"]:
        path = (root / str(record["path"])).resolve()
        path.relative_to(root.resolve())
        result.append(path)
    return result


def _uses_cpu_torch(lock: Path) -> bool:
    return any(
        line.strip().lower().startswith("torch==") and "+cpu" in line.lower()
        for line in lock.read_text(encoding="utf-8").splitlines()
    )


def populate_wheelhouse(root: Path, manifest: dict[str, Any], wheelhouse: Path) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    for lock in _lock_paths(root, manifest):
        command: list[str | Path] = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--require-hashes",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            wheelhouse,
        ]
        if _uses_cpu_torch(lock):
            command.extend(
                ["--index-url", CPU_INDEX, "--extra-index-url", PYPI_INDEX]
            )
        command.extend(["-r", lock])
        _run(command, cwd=root)


def install_from_wheelhouse(
    root: Path,
    manifest: dict[str, Any],
    wheelhouse: Path,
    environment: Path,
) -> None:
    if environment.exists():
        shutil.rmtree(environment)
    venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    python = _venv_python(environment)
    for lock in _lock_paths(root, manifest):
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "--no-deps",
                "--no-index",
                "--find-links",
                wheelhouse,
                "-r",
                lock,
            ],
            cwd=root,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = _load_json(manifest_path)
    validate_manifest_files(root, manifest)

    pre = verify_wheelhouse(root, manifest, args.wheelhouse)
    cache_hit = pre["status"] == "VERIFIED" and pre["wheel_count"] > 0
    download_seconds: float | None = None
    if not cache_hit:
        started = time.perf_counter()
        populate_wheelhouse(root, manifest, args.wheelhouse)
        download_seconds = time.perf_counter() - started

    verified = verify_wheelhouse(root, manifest, args.wheelhouse)
    if verified["status"] != "VERIFIED":
        raise DependencyCacheError("wheelhouse population produced no verified wheels")

    first_env = root / ".ci164-benchmark-first"
    started = time.perf_counter()
    install_from_wheelhouse(root, manifest, args.wheelhouse, first_env)
    first_install_seconds = time.perf_counter() - started
    shutil.rmtree(first_env, ignore_errors=True)

    warm_env = root / ".ci164-benchmark-warm"
    started = time.perf_counter()
    install_from_wheelhouse(root, manifest, args.wheelhouse, warm_env)
    warm_setup_seconds = time.perf_counter() - started
    shutil.rmtree(warm_env, ignore_errors=True)

    cold_setup_seconds = (
        download_seconds + first_install_seconds if download_seconds is not None else None
    )
    speedup = (
        cold_setup_seconds / warm_setup_seconds
        if cold_setup_seconds is not None and warm_setup_seconds > 0
        else None
    )
    evidence = {
        "schema_version": "12-6.ci164-cache-timing.v1",
        "cache_key": manifest["cache_key"],
        "profile_id": manifest["purpose_profile"]["profile_id"],
        "cache_hit_at_start": cache_hit,
        "download_seconds": download_seconds,
        "first_offline_install_seconds": first_install_seconds,
        "cold_setup_seconds": cold_setup_seconds,
        "warm_setup_seconds": warm_setup_seconds,
        "cold_to_warm_speedup": speedup,
        "verified_wheel_count": verified["wheel_count"],
        "verified_wheel_bytes": verified["verified_bytes"],
        "correctness": {
            "manifest_files_rehashed": True,
            "cached_wheels_rehashed_against_exact_locks": True,
            "install_uses_require_hashes": True,
            "install_uses_no_index": True,
            "cache_required_for_correctness": False,
        },
    }
    destination = args.evidence_out if args.evidence_out.is_absolute() else root / args.evidence_out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
