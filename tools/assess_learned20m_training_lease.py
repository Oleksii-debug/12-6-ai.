#!/usr/bin/env python3
"""Assess an immutable learned-20M launch manifest and optional TRAINING_RUN lease."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from twelve_six.learned20m_training_lease import assess_training_run_lease

MAX_INPUT_BYTES = 1_048_576


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object member")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is not finite")
    return parsed


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        raw = source.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("training lease input exceeds byte limit")
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_object,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return payload


def _print_contract_error(message: str) -> int:
    print(json.dumps({"contract_valid": False, "error": message}, sort_keys=True))
    return 2


def main(argv: list[str]) -> int:
    if len(argv) not in {2, 3, 5}:
        return _print_contract_error(
            "invalid arguments: expected MANIFEST [LEASE] "
            "[--now YYYY-MM-DDTHH:MM:SSZ]"
        )

    manifest_path = Path(argv[1])
    lease_path: Path | None = None
    now = None
    if len(argv) >= 3:
        lease_path = Path(argv[2])
    if len(argv) == 5:
        if argv[3] != "--now":
            return _print_contract_error(
                "invalid arguments: expected --now before timestamp"
            )
        try:
            now = datetime.strptime(argv[4], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return _print_contract_error(
                "invalid --now timestamp; expected YYYY-MM-DDTHH:MM:SSZ"
            )

    try:
        manifest = _read_object(manifest_path)
        lease = _read_object(lease_path) if lease_path is not None else None
        result = assess_training_run_lease(manifest, lease, now=now).as_dict()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return _print_contract_error(str(exc))

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["local_duplicate_guard_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
