"""Generate a non-authoritative Windows dependency-lock proposal from pip reports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "windows-x86_64"
EXACT_PYTHON = "3.11.9"
PROJECT = "twelve-six-ai"
_NAME_NORMALIZER = re.compile(r"[-_.]+")


def _canonical_name(name: str) -> str:
    return _NAME_NORMALIZER.sub("-", name.strip()).lower()


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
        value = json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)
    if not isinstance(value, dict):
        raise TypeError("pip report must contain an object")
    return value


def _packages(report: dict[str, Any]) -> dict[str, tuple[str, tuple[str, ...]]]:
    installs = report.get("install")
    if not isinstance(installs, list):
        raise TypeError("pip report is missing install list")
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    for item in installs:
        if not isinstance(item, dict):
            raise TypeError("pip report install entry must be an object")
        metadata = item.get("metadata")
        download = item.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise TypeError("pip report entry is missing metadata or download_info")
        name = _canonical_name(str(metadata.get("name", "")))
        if not name or name == PROJECT:
            continue
        version = str(metadata.get("version", "")).strip()
        archive = download.get("archive_info")
        if not version or not isinstance(archive, dict):
            raise RuntimeError(f"pip report lacks exact artifact evidence for {name}")
        hashes = archive.get("hashes")
        if not isinstance(hashes, dict):
            raise RuntimeError(f"pip report lacks artifact hashes for {name}")
        sha = hashes.get("sha256")
        if isinstance(sha, str):
            digests = (sha,)
        elif isinstance(sha, list):
            digests = tuple(str(value) for value in sha)
        else:
            raise RuntimeError(f"pip report lacks SHA-256 for {name}=={version}")
        current = (version, tuple(sorted(set(digests))))
        previous = result.get(name)
        if previous is not None and previous != current:
            raise RuntimeError(f"conflicting resolver evidence for {name}")
        result[name] = current
    return result


def _write_lock(
    path: Path,
    group: str,
    packages: dict[str, tuple[str, tuple[str, ...]]],
) -> None:
    lines = [
        f"# 12-6 AI {group} proposal; generated with CPython {EXACT_PYTHON}.",
        "# PROPOSAL ONLY until committed profile/index validation and exact-head CI pass.",
        "# Install with: python -m pip install --require-hashes --no-deps -r <this-file>",
    ]
    for name, (version, hashes) in sorted(packages.items()):
        if not hashes:
            raise RuntimeError(f"no hashes for {name}")
        line = f"{name}=={version}"
        for digest in hashes:
            line += f" --hash=sha256:{digest}"
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _toolchain_requirements() -> list[str]:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build = document.get("build-system", {})
    requires = build.get("requires", []) if isinstance(build, dict) else []
    if not isinstance(requires, list):
        raise TypeError("build-system.requires must be a list")
    return ["pip==26.2.1", *[str(item) for item in requires]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual = ".".join(str(value) for value in sys.version_info[:3])
    if sys.implementation.name.lower() != "cpython" or actual != EXACT_PYTHON:
        raise RuntimeError(f"requires exact CPython {EXACT_PYTHON}; got {actual}")
    if sys.platform != "win32":
        raise RuntimeError("Windows lock proposal must run on Windows")

    runtime = _packages(_run_report(["."]))
    combined_dev = _packages(_run_report([".[dev]"]))
    toolchain = _packages(_run_report(_toolchain_requirements()))
    for name, runtime_record in runtime.items():
        if combined_dev.get(name) != runtime_record:
            raise RuntimeError(f"dev resolution drifted runtime package {name}")
    dev_only = {name: record for name, record in combined_dev.items() if name not in runtime}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = {"runtime": runtime, "dev": dev_only, "toolchain": toolchain}
    for group, packages in groups.items():
        _write_lock(output / f"{group}.lock.txt", group, packages)
    proposal = {
        "schema_version": "12-6.windows-lock-proposal.v1",
        "profile_id": PROFILE,
        "python_version": EXACT_PYTHON,
        "package_counts": {group: len(packages) for group, packages in groups.items()},
    }
    (output / "proposal.json").write_text(
        json.dumps(proposal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(proposal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
