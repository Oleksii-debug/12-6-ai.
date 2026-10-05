#!/usr/bin/env python3
"""Validate and assess a provider-neutral 12-6 training run packet."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any

from twelve_six.portable_run_packet import assess_portable_run_packet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
MAX_INPUT_BYTES = 1_048_576
MAX_JSON_INTEGER_DIGITS = 64


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object member")
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> Any:
    raise ValueError("non-finite JSON constant")


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
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _load_packet(path: Path) -> dict[str, Any]:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("run packet must be a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise ValueError("run packet must be a regular file")
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ValueError("run packet changed between check and open")
                if _file_stamp(before) != _file_stamp(opened):
                    raise ValueError("run packet changed before open")
                raw = source.read(MAX_INPUT_BYTES + 1)
                if _file_stamp(os.fstat(source.fileno())) != _file_stamp(opened):
                    raise ValueError("run packet changed during read")
        finally:
            if descriptor != -1:
                os.close(descriptor)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("run packet exceeds input byte limit")
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
    except OSError:
        raise ValueError("cannot read run packet") from None
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds decoder limit") from exc
    if not isinstance(payload, dict):
        raise ValueError("run packet root must be an object")
    return payload


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(
            json.dumps(
                {
                    "contract_valid": False,
                    "error": "invalid arguments: expected at most one packet path",
                },
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
