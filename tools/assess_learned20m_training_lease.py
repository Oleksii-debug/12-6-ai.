#!/usr/bin/env python3
"""Assess an immutable learned-20M launch manifest and optional TRAINING_RUN lease."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from twelve_six.learned20m_training_lease import assess_training_run_lease

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


def _read_object(
    path: Path,
    *,
    label: str = "training lease input",
) -> dict[str, Any]:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise ValueError(f"{label} must be a regular file")
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ValueError(f"{label} changed between check and open")
                if _file_stamp(before) != _file_stamp(opened):
                    raise ValueError(f"{label} changed before open")
                raw = source.read(MAX_INPUT_BYTES + 1)
                if _file_stamp(os.fstat(source.fileno())) != _file_stamp(opened):
                    raise ValueError(f"{label} changed during read")
        finally:
            if descriptor != -1:
                os.close(descriptor)
    except OSError:
        raise ValueError(f"cannot read {label}") from None
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError(f"{label} exceeds byte limit")
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_object,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
        parse_int=_parse_bounded_int,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
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
        manifest = _read_object(manifest_path, label="manifest")
        lease = (
            _read_object(lease_path, label="lease")
            if lease_path is not None
            else None
        )
        result = assess_training_run_lease(manifest, lease, now=now).as_dict()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return _print_contract_error(str(exc))

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["local_duplicate_guard_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
