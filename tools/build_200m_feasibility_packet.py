#!/usr/bin/env python3
"""Build or verify the non-authorizing R01 ~200M feasibility packet."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from twelve_six.feasibility_200m import (
    FeasibilityPacketError,
    build_200m_feasibility_packet,
    retained_identities_for_built_packet,
    validate_200m_feasibility_packet,
)

_BUILD_FIELDS = {
    "source_git_sha",
    "candidate",
    "measurements_20m",
    "measurement_authority",
    "requirement_evidence",
    "decision",
}
_EXPECTED_FIELDS = {
    "packet_sha256",
    "roadmap_snapshot_sha256",
    "source_git_sha",
    "measurements_20m_sha256",
    "requirement_evidence_sha256",
}

MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_JSON_INTEGER_DIGITS = 64


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite_json_constant:{value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("json_number_not_finite")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ValueError("json_number_underflow")
    return parsed


def _parse_bounded_int(value: str) -> int:
    if len(value.removeprefix("-")) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError("json_integer_too_large")
    return int(value)


def _file_stamp(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _path_key(path: Path, *, label: str) -> str:
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        raise ValueError(f"{label}_path_unresolvable") from None
    return os.path.normcase(str(resolved))


def _require_distinct_paths(named_paths: list[tuple[str, Path]]) -> None:
    seen: dict[str, str] = {}
    for label, path in named_paths:
        key = _path_key(path, label=label)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(f"{label}_path_collides_with_{previous}")
        seen[key] = label


def _read_json(path: Path, *, label: str) -> Any:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(f"{label}_symlink_not_allowed")
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}_not_regular_file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise ValueError(f"{label}_not_regular_file")
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ValueError(f"{label}_changed_between_check_and_open")
                if _file_stamp(before) != _file_stamp(opened):
                    raise ValueError(f"{label}_changed_before_open")
                raw = source.read(MAX_INPUT_BYTES + 1)
                after = os.fstat(source.fileno())
                if _file_stamp(after) != _file_stamp(opened):
                    raise ValueError(f"{label}_changed_during_read")
        finally:
            if descriptor != -1:
                os.close(descriptor)
    except OSError:
        raise ValueError(f"{label}_unreadable") from None

    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError(f"{label}_exceeds_byte_limit")
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
    except RecursionError:
        raise ValueError(f"{label}_json_too_deep") from None


def _render_json(value: Any) -> bytes:
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return (rendered + "\n").encode("utf-8")


def _write_json(path: Path, value: Any, *, label: str) -> None:
    payload = _render_json(value)
    temporary: Path | None = None
    descriptor = -1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(f"{label}_destination_not_regular_file")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as sink:
            descriptor = -1
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError:
        raise ValueError(f"{label}_write_failed") from None
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _build(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        [
            ("roadmap", args.roadmap),
            ("build_input", args.input),
            ("output", args.output),
            ("external_identities", args.external_identities),
        ]
    )

    roadmap = _read_json(args.roadmap, label="roadmap")
    request = _read_json(args.input, label="build_input")
    if not isinstance(request, dict) or set(request) != _BUILD_FIELDS:
        raise ValueError("build_input_fields_mismatch")
    packet = build_200m_feasibility_packet(
        roadmap_snapshot=roadmap,
        source_git_sha=request["source_git_sha"],
        candidate=request["candidate"],
        measurements_20m=request["measurements_20m"],
        measurement_authority=request["measurement_authority"],
        requirement_evidence=request["requirement_evidence"],
        decision=request["decision"],
    )
    retained_identities = retained_identities_for_built_packet(packet)
    _write_json(
        args.external_identities,
        retained_identities,
        label="external_identities",
    )
    _write_json(args.output, packet, label="output")
    print(packet["packet_sha256"])
    return 0


def _verify(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        [
            ("roadmap", args.roadmap),
            ("packet", args.packet),
            ("expected_identities", args.expected_identities),
        ]
    )
    roadmap = _read_json(args.roadmap, label="roadmap")
    packet = _read_json(args.packet, label="packet")
    expected = _read_json(args.expected_identities, label="expected_identities")
    if not isinstance(expected, dict) or set(expected) != _EXPECTED_FIELDS:
        raise ValueError("expected_identities_fields_mismatch")
    errors = validate_200m_feasibility_packet(
        packet,
        roadmap_snapshot=roadmap,
        expected_packet_sha256=expected["packet_sha256"],
        expected_roadmap_snapshot_sha256=expected[
            "roadmap_snapshot_sha256"
        ],
        expected_source_git_sha=expected["source_git_sha"],
        expected_measurements_20m_sha256=expected[
            "measurements_20m_sha256"
        ],
        expected_requirement_evidence_sha256=expected[
            "requirement_evidence_sha256"
        ],
    )
    packet_sha256 = (
        packet.get("packet_sha256") if isinstance(packet, dict) else None
    )
    if not (
        isinstance(packet_sha256, str)
        and len(packet_sha256) == 64
        and all(character in "0123456789abcdef" for character in packet_sha256)
    ):
        packet_sha256 = None
    report = {
        "valid": not errors,
        "errors": errors,
        "packet_sha256": packet_sha256,
        "authority_granted": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--roadmap", type=Path, required=True)
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--external-identities", type=Path, required=True)
    build.set_defaults(run=_build)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--roadmap", type=Path, required=True)
    verify.add_argument("--packet", type=Path, required=True)
    verify.add_argument("--expected-identities", type=Path, required=True)
    verify.set_defaults(run=_verify)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return args.run(args)
    except (OSError, ValueError, TypeError, FeasibilityPacketError) as exc:
        print(f"error:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
