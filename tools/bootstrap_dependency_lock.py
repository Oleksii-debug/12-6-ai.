#!/usr/bin/env python3
"""Generate one platform-specific, hash-locked dependency profile from pip reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_MODULE = ROOT / "src" / "twelve_six" / "integration" / "dependency_lock.py"
_spec = importlib.util.spec_from_file_location("twelve_six_dependency_lock", LOCK_MODULE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load dependency lock module")
_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lock)

EXACT_PYTHON_VERSION = _lock.EXACT_PYTHON_VERSION
PROJECT_DISTRIBUTION = _lock.PROJECT_DISTRIBUTION
assert_exact_python = _lock.assert_exact_python
build_profile_manifest = _lock.build_profile_manifest
canonical_distribution_name = _lock.canonical_distribution_name
current_profile_id = _lock.current_profile_id
write_manifest = _lock.write_manifest
LOCK_ROOT = Path("requirements/locks")


def _run_report(requirements: list[str]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        report_path = Path(handle.name)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--dry-run",
        "--ignore-installed",
        "--report",
        str(report_path),
        *requirements,
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)
    if not isinstance(report, dict):
        raise RuntimeError("pip report must contain an object")
    return report


def _packages(report: dict[str, Any]) -> dict[str, tuple[str, tuple[str, ...]]]:
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    installations = report.get("install")
    if not isinstance(installations, list):
        raise RuntimeError("pip report is missing install list")
    for item in installations:
        if not isinstance(item, dict):
            raise RuntimeError("pip report install entry must be an object")
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("pip report install entry is missing metadata")
        name = canonical_distribution_name(str(metadata.get("name", "")))
        if name == PROJECT_DISTRIBUTION:
            continue
        version = str(metadata.get("version", "")).strip()
        download = item.get("download_info")
        if not version or not isinstance(download, dict):
            raise RuntimeError(f"pip report lacks exact artifact evidence for {name}")
        archive = download.get("archive_info")
        if not isinstance(archive, dict):
            raise RuntimeError(f"pip report lacks archive info for {name}")
        hashes = archive.get("hashes")
        if not isinstance(hashes, dict) or not hashes.get("sha256"):
            raise RuntimeError(f"pip report lacks SHA-256 for {name}=={version}")
        sha_values = hashes["sha256"]
        if isinstance(sha_values, str):
            digest_values = (sha_values,)
        elif isinstance(sha_values, list):
            digest_values = tuple(str(value) for value in sha_values)
        else:
            raise RuntimeError(f"pip report SHA-256 is malformed for {name}")
        normalized_hashes = tuple(sorted(set(digest_values)))
        previous = result.get(name)
        current = (version, normalized_hashes)
        if previous is not None and previous != current:
            raise RuntimeError(f"conflicting resolver evidence for {name}")
        result[name] = current
    return result


def _write_lock(path: Path, packages: dict[str, tuple[str, tuple[str, ...]]], *, group: str) -> None:
    lines = [
        f"# 12-6 AI {group} lock; generated with CPython {EXACT_PYTHON_VERSION}.",
        "# Install with: python -m pip install --require-hashes --no-deps -r <this-file>",
    ]
    for name, (version, hashes) in sorted(packages.items()):
        if not hashes:
            raise RuntimeError(f"no artifact hashes for {name}")
        line = f"{name}=={version}"
        for digest in hashes:
            line += f" --hash=sha256:{digest}"
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _declared_toolchain() -> list[str]:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build = document.get("build-system", {})
    requires = build.get("requires", []) if isinstance(build, dict) else []
    if not isinstance(requires, list):
        raise RuntimeError("build-system.requires must be a list")
    return ["pip==26.2.1", *[str(item) for item in requires]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()

    assert_exact_python()
    profile = current_profile_id()
    output_root = args.output_root.resolve()
    profile_dir = LOCK_ROOT / profile

    runtime = _packages(_run_report(["."]))
    combined_dev = _packages(_run_report([".[dev]"]))
    toolchain = _packages(_run_report(_declared_toolchain()))

    for name, runtime_record in runtime.items():
        if name not in combined_dev or combined_dev[name] != runtime_record:
            raise RuntimeError(f"dev resolution drifted runtime package {name}")
    dev_only = {name: record for name, record in combined_dev.items() if name not in runtime}

    lock_files = {
        "toolchain": profile_dir / "toolchain.lock.txt",
        "runtime": profile_dir / "runtime.lock.txt",
        "dev": profile_dir / "dev.lock.txt",
    }
    groups = {"toolchain": toolchain, "runtime": runtime, "dev": dev_only}
    for group, relative in lock_files.items():
        _write_lock(output_root / relative, groups[group], group=group)

    manifest = build_profile_manifest(
        root=output_root,
        profile_id=profile,
        lock_files=lock_files,
        package_counts={group: len(packages) for group, packages in groups.items()},
    )
    manifest_path = output_root / profile_dir / "profile.json"
    write_manifest(manifest_path, manifest)
    print(f"profile={profile}")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    for group, packages in groups.items():
        print(f"{group}_packages={len(packages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
