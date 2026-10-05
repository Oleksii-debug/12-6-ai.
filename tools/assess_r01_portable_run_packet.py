#!/usr/bin/env python3
"""Validate and assess a provider-neutral 12-6 training run packet."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from twelve_six.portable_run_packet import assess_portable_run_packet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
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


def _load_packet(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("run packet exceeds input byte limit")
        payload = json.loads(
            raw.decode("utf-8"),
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
    if len(argv) > 2:
        print(
            json.dumps(
                {"contract_valid": False, "error": "invalid arguments: expected at most one packet path"},
                sort_keys=True,
            )
        )
        return 2
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
