#!/usr/bin/env python3
"""Bounded LOCAL_FREE current-main materializer for the qualified Lesia 1892 Wikisource edition."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any

from twelve_six.data.wikisource_pd_contract import (
    WikisourceIntakeError,
    validate_control_contract,
)
from twelve_six.data.wikisource_pd_edition import materialize_live


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/data/d03_wikisource_lesia1892_current_main_v1.json"
MAX_CONTROL_BYTES = 1_048_576
MAX_JSON_INTEGER_DIGITS = 64


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WikisourceIntakeError("control contract has duplicate JSON object member")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise WikisourceIntakeError(f"control contract has non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise WikisourceIntakeError("control contract has non-finite JSON number")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise WikisourceIntakeError("control contract JSON number underflowed to zero")
    return parsed


def _parse_bounded_int(value: str) -> int:
    if len(value.removeprefix("-")) > MAX_JSON_INTEGER_DIGITS:
        raise WikisourceIntakeError("control contract JSON integer exceeds 64 digits")
    return int(value)


def load_strict_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
    except json.JSONDecodeError as exc:
        raise WikisourceIntakeError(f"invalid control contract JSON: {exc.msg}") from exc
    except RecursionError as exc:
        raise WikisourceIntakeError("control contract JSON nesting limit exceeded") from exc
    if not isinstance(value, dict):
        raise WikisourceIntakeError("control contract must be a JSON object")
    return value


def _file_stamp(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_control_contract(path: Path) -> dict[str, Any]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise WikisourceIntakeError("control contract must be a regular non-symlink file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise WikisourceIntakeError("control contract changed between check and open")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            raw = source.read(MAX_CONTROL_BYTES + 1)
            after = os.fstat(source.fileno())
        if _file_stamp(after) != _file_stamp(opened):
            raise WikisourceIntakeError("control contract changed during read")
    except FileNotFoundError as exc:
        raise WikisourceIntakeError("cannot read control contract") from exc
    except OSError as exc:
        raise WikisourceIntakeError("cannot safely read control contract") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if len(raw) > MAX_CONTROL_BYTES:
        raise WikisourceIntakeError("control contract exceeds byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WikisourceIntakeError("control contract is not valid UTF-8") from exc
    return load_strict_json_object(text)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_identity(path: Path, *, label: str) -> tuple[int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise WikisourceIntakeError(f"cannot inspect {label}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise WikisourceIntakeError(f"{label} is not a regular file")
    return (info.st_dev, info.st_ino)


def _unlink_owned(
    path: Path,
    identity: tuple[int, int],
    *,
    label: str,
    missing_ok: bool = False,
) -> None:
    try:
        observed = _regular_identity(path, label=label)
    except WikisourceIntakeError:
        if missing_ok and not os.path.lexists(path):
            return
        raise
    if observed != identity:
        raise WikisourceIntakeError(f"{label} ownership changed before rollback")
    try:
        path.unlink()
    except OSError as exc:
        raise WikisourceIntakeError(f"cannot remove owned {label}") from exc
    _fsync_directory(path.parent)


def _write_create_only_durable(path: Path, payload: bytes) -> tuple[int, int]:
    identity: tuple[int, int] | None = None
    try:
        with path.open("xb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise WikisourceIntakeError("created materialization output is not regular")
            identity = (info.st_dev, info.st_ino)
            if handle.write(payload) != len(payload):
                raise OSError("incomplete materialization output write")
            handle.flush()
            os.fsync(handle.fileno())
        if _regular_identity(path, label="materialization output") != identity:
            raise WikisourceIntakeError("materialization output ownership changed after write")
        _fsync_directory(path.parent)
        return identity
    except BaseException as exc:
        if identity is not None:
            try:
                _unlink_owned(
                    path,
                    identity,
                    label="failed materialization output",
                    missing_ok=True,
                )
            except WikisourceIntakeError as cleanup_exc:
                raise WikisourceIntakeError(
                    "materialization output write failed and owned cleanup could not be proven"
                ) from cleanup_exc
        raise exc


def _prepare_output_pair(candidate: Path, report: Path) -> tuple[Path, Path]:
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    report = Path(os.path.abspath(os.fspath(report)))
    for output in (candidate, report):
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
    candidate_identity = os.path.normcase(str(candidate.resolve(strict=False)))
    report_identity = os.path.normcase(str(report.resolve(strict=False)))
    if candidate_identity == report_identity:
        raise WikisourceIntakeError("candidate and report outputs must be distinct")
    for output in (candidate, report):
        output.parent.mkdir(parents=True, exist_ok=True)
    return candidate, report


def _publish_output_pair(
    candidate_path: Path,
    candidate_payload: bytes,
    report_path: Path,
    report_payload: bytes,
) -> None:
    candidate_path, report_path = _prepare_output_pair(candidate_path, report_path)
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        candidate_identity = _write_create_only_durable(candidate_path, candidate_payload)
        created.append((candidate_path, candidate_identity))
        report_identity = _write_create_only_durable(report_path, report_payload)
        created.append((report_path, report_identity))
    except BaseException as exc:
        rollback_errors: list[str] = []
        for output, identity in reversed(created):
            try:
                _unlink_owned(
                    output,
                    identity,
                    label="partial materialization output",
                    missing_ok=True,
                )
            except WikisourceIntakeError as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise WikisourceIntakeError(
                "partial materialization publication rollback failed: "
                + "; ".join(rollback_errors)
            ) from exc
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=112)
    return parser


def _reject_duplicate_options(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> None:
    known = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    seen: set[str] = set()
    for raw in argv:
        option = raw.split("=", 1)[0]
        if option not in known:
            continue
        if option in seen:
            parser.error(f"argument {option}: may not be repeated")
        seen.add(option)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = sys.argv[1:] if argv is None else list(argv)
    _reject_duplicate_options(parser, arguments)
    args = parser.parse_args(arguments)
    try:
        candidate_path, report_path = _prepare_output_pair(
            args.candidate_out,
            args.report_out,
        )
        contract = _read_control_contract(args.contract)
        validate_control_contract(contract)
        result = materialize_live(max_pages=args.max_pages)
        report_payload = (
            json.dumps(result.report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        _publish_output_pair(
            candidate_path,
            result.candidate_jsonl,
            report_path,
            report_payload,
        )
    except (OSError, ValueError, WikisourceIntakeError) as exc:
        parser.error(str(exc))
    print(result.report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
