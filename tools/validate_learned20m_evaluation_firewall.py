#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from twelve_six.learned20m_evaluation_firewall import EvaluationFirewallError, validate_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = Path("configs/evaluation/learned20m_evaluation_firewall_v1.json")
MAX_INPUT_BYTES = 1_048_576
MAX_JSON_INTEGER_DIGITS = 64


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
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ValueError("nonzero JSON number underflowed to zero")
    return parsed


def _parse_bounded_int(value: str) -> int:
    if len(value.removeprefix("-")) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError("JSON integer exceeds 64 digits")
    return int(value)


def _file_stamp(info: os.stat_result) -> tuple[int, int, int]:
    # Atime may change during the read; content-bearing metadata must not.
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("evaluation firewall policy must be a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("evaluation firewall policy must be a regular file")
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("evaluation firewall policy changed between check and open")
            if _file_stamp(before) != _file_stamp(opened):
                raise ValueError("evaluation firewall policy changed before open")
            raw = source.read(MAX_INPUT_BYTES + 1)
            if _file_stamp(os.fstat(source.fileno())) != _file_stamp(opened):
                raise ValueError("evaluation firewall policy changed during read")
    except OSError:
        raise ValueError("cannot read evaluation firewall policy") from None

    try:
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("evaluation firewall policy exceeds input byte limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
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
