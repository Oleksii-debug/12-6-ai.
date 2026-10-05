#!/usr/bin/env python3
"""Assess the fail-closed learned-20M launch packet."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from twelve_six.learned20m_readiness import assess_learned20m_readiness

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
MAX_INPUT_BYTES = 1_048_576
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 10_000


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Never echo an untrusted member name into automation/log output.
            raise ValueError("duplicate object member")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is not finite")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(char in "123456789" for char in significand):
        raise ValueError("nonzero JSON number underflowed to zero")
    return parsed


def _load_packet(path: Path) -> dict[str, Any]:
    # Bound each untrusted read before parsing, including on Windows network drives.
    with path.open("rb") as source:
        raw = source.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("launch packet exceeds input byte limit")
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_object,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
    )
    if not isinstance(payload, dict):
        raise ValueError("launch packet root must be an object")

    # Iterative traversal keeps the structural limit independent of Python recursion.
    pending: list[tuple[Any, int]] = [(payload, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ValueError("launch packet exceeds JSON structure limit")
        if isinstance(current, dict):
            for key, value in current.items():
                key.encode("utf-8")  # Reject unpaired JSON-escaped surrogates.
                pending.append((value, depth + 1))
        elif isinstance(current, list):
            pending.extend((value, depth + 1) for value in current)
        elif isinstance(current, str):
            current.encode("utf-8")
    return payload


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(
            json.dumps(
                {"error": "invalid arguments: expected at most one packet path"},
                sort_keys=True,
            )
        )
        return 2
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    try:
        payload = _load_packet(path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        print(json.dumps({"error": f"invalid launch packet: {exc}"}, sort_keys=True))
        return 2

    result = assess_learned20m_readiness(payload).as_dict()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["material_training_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
