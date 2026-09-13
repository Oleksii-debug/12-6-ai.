"""Cheap dependency/tool preflight before any expensive CI environment install."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "requirements" / "locks" / "index.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return value


def _require_command(name: str) -> str:
    value = shutil.which(name)
    if value is None:
        raise SystemExit(f"required command not found: {name}")
    return value


def _validate_profile(profile_id: str) -> dict[str, Any]:
    index = _read_json(INDEX)
    if index.get("python_version") != "3.11.16":
        raise SystemExit("lock index Python identity drift")
    profiles = index.get("profiles")
    if not isinstance(profiles, dict) or profile_id not in profiles:
        raise SystemExit(f"unknown lock profile: {profile_id}")
    record = profiles[profile_id]
    if not isinstance(record, dict):
        raise SystemExit(f"invalid profile record: {profile_id}")
    manifest_path = ROOT / str(record["path"])
    if _sha256(manifest_path) != record["sha256"]:
        raise SystemExit(f"profile file SHA mismatch: {profile_id}")
    profile = _read_json(manifest_path)
    if profile.get("manifest_sha256") != record["manifest_sha256"]:
        raise SystemExit(f"profile semantic identity drift: {profile_id}")
    locks = profile.get("locks")
    if not isinstance(locks, dict) or not locks:
        raise SystemExit(f"profile has no lock groups: {profile_id}")
    checked: dict[str, dict[str, Any]] = {}
    for group, lock in sorted(locks.items()):
        if not isinstance(lock, dict):
            raise SystemExit(f"invalid lock record: {profile_id}/{group}")
        path = ROOT / str(lock["path"])
        actual = _sha256(path)
        expected = lock.get("sha256")
        if actual != expected:
            raise SystemExit(f"lock SHA mismatch: {profile_id}/{group}")
        checked[group] = {"path": str(lock["path"]), "sha256": actual}
    return {
        "profile_id": profile_id,
        "profile_path": str(record["path"]),
        "profile_file_sha256": record["sha256"],
        "profile_manifest_sha256": record["manifest_sha256"],
        "locks": checked,
    }


def run(profile_id: str, source_sha: str, output: Path | None) -> dict[str, Any]:
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise SystemExit("--source-sha must be a lowercase 40-character Git SHA")
    if sys.version_info[:3] != (3, 11, 16):
        raise SystemExit(f"Python drift: expected 3.11.16, got {sys.version.split()[0]}")
    git = _require_command("git")
    current = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != source_sha:
        raise SystemExit(f"source checkout mismatch: {current} != {source_sha}")
    profile = _validate_profile(profile_id)
    evidence = {
        "schema": "12-6.ci-dependency-preflight.v1",
        "source_sha": source_sha,
        "python": sys.version.split()[0],
        "git": subprocess.check_output([git, "--version"], text=True).strip(),
        "lock_index_path": str(INDEX.relative_to(ROOT)),
        "lock_index_file_sha256": _sha256(INDEX),
        "profile": profile,
        "network_install_performed": False,
        "tests_performed": False,
        "status": "PASS",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="linux-x86_64")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.profile, args.source_sha, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
