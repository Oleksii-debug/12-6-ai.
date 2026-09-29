#!/usr/bin/env python3
"""Assess an immutable learned-20M launch manifest and optional TRAINING_RUN lease."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from twelve_six.learned20m_training_lease import assess_training_run_lease


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return payload


def main(argv: list[str]) -> int:
    if len(argv) not in {2, 3, 5}:
        print(
            "usage: assess_learned20m_training_lease.py MANIFEST [LEASE] "
            "[--now YYYY-MM-DDTHH:MM:SSZ]",
            file=sys.stderr,
        )
        return 2

    manifest_path = Path(argv[1])
    lease_path: Path | None = None
    now = None
    if len(argv) >= 3:
        lease_path = Path(argv[2])
    if len(argv) == 5:
        if argv[3] != "--now":
            print("expected --now before timestamp", file=sys.stderr)
            return 2
        try:
            now = datetime.strptime(argv[4], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            print(json.dumps({"contract_valid": False, "error": str(exc)}, sort_keys=True))
            return 2

    try:
        manifest = _read_object(manifest_path)
        lease = _read_object(lease_path) if lease_path is not None else None
        result = assess_training_run_lease(manifest, lease, now=now).as_dict()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"contract_valid": False, "error": str(exc)}, sort_keys=True))
        return 2

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["local_duplicate_guard_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
