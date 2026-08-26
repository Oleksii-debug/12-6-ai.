#!/usr/bin/env python3
"""Create a deterministic project environment from committed hash locks.

This bootstrap intentionally performs no project imports or tests.  It only
validates lock syntax, creates a clean venv, and installs the requested lock
groups with --require-hashes --no-deps in declared order.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import venv
from hashlib import sha256
from pathlib import Path

LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;@/\\]+(?: --hash=sha256:[0-9a-f]{64})+$"
)


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate_lock(path: Path) -> int:
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if LOCK_LINE.fullmatch(line) is None:
            raise SystemExit(f"non-exact or unhashed lock line {path}:{number}")
        count += 1
    if count == 0:
        raise SystemExit(f"empty lock: {path}")
    return count


def venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--lock", type=Path, action="append", required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    args = parser.parse_args()

    locks = [path.resolve() for path in args.lock]
    lock_evidence = []
    for path in locks:
        if not path.is_file():
            raise SystemExit(f"missing lock: {path}")
        lock_evidence.append(
            {"path": path.as_posix(), "sha256": file_sha(path), "package_count": validate_lock(path)}
        )

    environment = args.venv.resolve()
    if environment.exists():
        shutil.rmtree(environment)
    venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    python = venv_python(environment)
    for path in locks:
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "--no-deps",
                "-r",
                str(path),
            ],
            check=True,
        )

    freeze = subprocess.check_output(
        [str(python), "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    evidence = {
        "schema": "12-6.universal-environment-bootstrap.v1",
        "python_executable": str(python),
        "lock_order": lock_evidence,
        "installed": sorted(freeze),
        "floating_resolution": False,
        "require_hashes": True,
        "no_deps": True,
    }
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["evidence_sha256"] = sha256(raw).hexdigest()
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"python={python}")
    print(f"evidence_sha256={evidence['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
