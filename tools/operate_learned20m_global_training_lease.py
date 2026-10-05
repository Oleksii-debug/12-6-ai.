#!/usr/bin/env python3
"""Machine-readable operator for the learned-20M global Git-ref lease."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.learned20m_global_training_lease import (
    acquire_global_training_run_lease,
    inspect_global_training_run_lease,
    renew_global_training_run_lease,
    terminate_global_training_run_lease,
)
from twelve_six.learned20m_training_lease import build_authorized_training_run_lease


MAX_MANIFEST_BYTES = 1_048_576
MAX_JSON_INTEGER_DIGITS = 64


class _DuplicateKey(ValueError):
    pass


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Never echo attacker-controlled member names into operator logs.
            raise _DuplicateKey("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_constant:{value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("json_number_not_finite")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ValueError("nonzero_json_number_underflowed_to_zero")
    return parsed


def _parse_bounded_int(value: str) -> int:
    if len(value.removeprefix("-")) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError("json_integer_exceeds_64_digits")
    return int(value)


def _file_stamp(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    # Open attacker-controlled input nonblocking and bind the read to one regular
    # descriptor before allocation or JSON decoding.
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_BINARY", 0)
        )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("manifest_not_regular_file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
            if _file_stamp(os.fstat(handle.fileno())) != _file_stamp(opened):
                raise ValueError("manifest_changed_during_read")
    except OSError:
        raise ValueError("manifest_read_failed") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest_exceeds_byte_limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest_utf8_invalid") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
    except RecursionError as exc:
        raise ValueError("manifest_json_invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("manifest_not_object")
    return value


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    subparsers.add_parser("inspect", allow_abbrev=False)

    acquire = subparsers.add_parser("acquire", allow_abbrev=False)
    acquire.add_argument("--run-id", required=True)
    acquire.add_argument("--holder-id", required=True)
    acquire.add_argument("--ttl-seconds", type=int, required=True)
    acquire.add_argument("--expected-terminal-authority-sha256", required=True)

    renew = subparsers.add_parser("renew", allow_abbrev=False)
    renew.add_argument("--expected-remote-tip", required=True)
    renew.add_argument("--ttl-seconds", type=int, required=True)

    terminate = subparsers.add_parser("terminate", allow_abbrev=False)
    terminate.add_argument("--expected-remote-tip", required=True)
    terminate.add_argument(
        "--status",
        choices=("COMPLETED", "FAILED", "ABORTED"),
        required=True,
    )
    return parser


def _reject_repeated_known_options(
    parser: argparse.ArgumentParser,
    raw_args: list[str],
) -> None:
    known: set[str] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        for action in current._actions:
            known.update(
                option
                for option in action.option_strings
                if option.startswith("--")
            )
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                pending.extend(
                    choice
                    for choice in choices.values()
                    if isinstance(choice, argparse.ArgumentParser)
                )

    seen: set[str] = set()
    for raw in raw_args:
        option = raw.split("=", 1)[0]
        if option not in known:
            continue
        if option in seen:
            parser.error(f"argument {option}: may not be repeated")
        seen.add(option)


def main() -> int:
    parser = _parser()
    _reject_repeated_known_options(parser, sys.argv[1:])
    args = parser.parse_args()
    try:
        manifest = _load_mapping(args.manifest)
        if args.operation == "inspect":
            result = inspect_global_training_run_lease(
                args.repo_root, args.remote, manifest
            )
            _emit(result.as_dict())
            return 0 if result.present and result.valid else 3
        if args.operation == "acquire":
            lease = build_authorized_training_run_lease(
                manifest,
                expected_terminal_authority_sha256=(
                    args.expected_terminal_authority_sha256
                ),
                run_id=args.run_id,
                holder_id=args.holder_id,
                ttl_seconds=args.ttl_seconds,
            )
            result = acquire_global_training_run_lease(
                args.repo_root,
                args.remote,
                manifest,
                lease.as_dict(),
                expected_terminal_authority_sha256=(
                    args.expected_terminal_authority_sha256
                ),
            )
        elif args.operation == "renew":
            result = renew_global_training_run_lease(
                args.repo_root,
                args.remote,
                manifest,
                expected_remote_tip=args.expected_remote_tip,
                ttl_seconds=args.ttl_seconds,
            )
        else:
            result = terminate_global_training_run_lease(
                args.repo_root,
                args.remote,
                manifest,
                expected_remote_tip=args.expected_remote_tip,
                status=args.status,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"operation": args.operation, "ok": False, "blockers": [str(exc)]})
        return 2
    _emit(result.as_dict())
    return 0 if result.committed and result.post_write_reread_verified else 3


if __name__ == "__main__":
    raise SystemExit(main())
