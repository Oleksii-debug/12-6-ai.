#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from twelve_six.learned20m_evaluation_firewall import EvaluationFirewallError, validate_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = Path("configs/evaluation/learned20m_evaluation_firewall_v1.json")
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


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("evaluation firewall policy exceeds input byte limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("evaluation firewall policy JSON nesting limit exceeded") from exc
    if not isinstance(value, dict):
        raise TypeError("evaluation firewall policy root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    path = args.policy if args.policy.is_absolute() else args.repo_root / args.policy
    try:
        policy = _load_policy(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    try:
        result = validate_policy(policy)
    except EvaluationFirewallError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
