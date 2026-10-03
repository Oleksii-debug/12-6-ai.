#!/usr/bin/env python3
"""Assess the fail-closed learned-20M launch packet."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from twelve_six.learned20m_readiness import assess_learned20m_readiness

DEFAULT_PATH = Path("configs/research/r01_learned20m_launch_readiness_v1.json")


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
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_object,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
    )
    if not isinstance(payload, dict):
        raise ValueError("launch packet root must be an object")
    return payload


def main(argv: list[str]) -> int:
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
