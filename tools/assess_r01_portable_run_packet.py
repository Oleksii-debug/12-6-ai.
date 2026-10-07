#!/usr/bin/env python3
"""Validate and assess a provider-neutral 12-6 training run packet."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from twelve_six.portable_run_packet import assess_portable_run_packet

DEFAULT_PATH = Path("configs/research/r01_portable_local_free_run_packet_v1.json")


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object member: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is not finite")
    return parsed


def _load_packet(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds decoder limit") from exc
    if not isinstance(payload, dict):
        raise ValueError("run packet root must be an object")
    return payload


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    try:
        payload = _load_packet(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"contract_valid": False, "error": str(exc)}, sort_keys=True))
        return 2

    result = assess_portable_run_packet(payload).as_dict()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
