"""Generate, verify and offline-restore Plan-1 OS/ABI wheelhouse locks."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from twelve_six.environment_lock import (
    EnvironmentLockError, build_lock, verify_or_restore,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify", "restore"))
    parser.add_argument("--platform", choices=("linux_x86_64", "win_amd64"), required=True)
    parser.add_argument("--python", choices=("3.11", "3.12", "3.13"), required=True)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--expected-lock-sha256")
    args = parser.parse_args(argv)
    try:
        project = args.project.read_bytes()
        if args.command == "generate":
            lock = build_lock(args.wheelhouse, platform=args.platform,
                              python=args.python, pyproject=project)
            if args.lock.exists() or args.lock.is_symlink():
                raise EnvironmentLockError("immutable lock destination already exists")
            args.lock.parent.mkdir(parents=True, exist_ok=True)
            # Replacing a previously approved lock without impact qualification
            # is forbidden. Generate only to a new named candidate pathname.
            with args.lock.open("xb") as handle:
                handle.write(lock)
            print(json.dumps({"status": "GENERATED_UNAPPROVED_CANDIDATE",
                              "sha256": hashlib.sha256(lock).hexdigest()}, sort_keys=True))
            return 0
        if not args.expected_lock_sha256:
            raise EnvironmentLockError("independently pinned --expected-lock-sha256 required")
        lock = args.lock.read_bytes()
        if hashlib.sha256(lock).hexdigest() != args.expected_lock_sha256:
            raise EnvironmentLockError("independent lock pin drift")
        if args.command == "restore" and args.target is None:
            raise EnvironmentLockError("--target required for offline restore")
        result = verify_or_restore(lock, pyproject=project, wheelhouse=args.wheelhouse,
                                   platform=args.platform, python=args.python,
                                   target=args.target or Path("not-used"),
                                   restore=args.command == "restore")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (EnvironmentLockError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"DENIED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
