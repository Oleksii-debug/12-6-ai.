#!/usr/bin/env python3
"""Build or verify the non-authorizing R01 ~200M feasibility packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from twelve_six.feasibility_200m import (
    FeasibilityPacketError,
    build_200m_feasibility_packet,
    expected_external_identities,
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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite_json_constant:{value}")


def _read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(rendered + "\n", encoding="utf-8")


def _build(args: argparse.Namespace) -> int:
    roadmap = _read_json(args.roadmap)
    request = _read_json(args.input)
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
    _write_json(args.output, packet)
    if args.external_identities is not None:
        _write_json(
            args.external_identities,
            expected_external_identities(packet),
        )
    print(packet["packet_sha256"])
    return 0


def _verify(args: argparse.Namespace) -> int:
    roadmap = _read_json(args.roadmap)
    packet = _read_json(args.packet)
    expected = _read_json(args.expected_identities)
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
    report = {
        "valid": not errors,
        "errors": errors,
        "packet_sha256": (
            packet.get("packet_sha256") if isinstance(packet, dict) else None
        ),
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
    build.add_argument("--external-identities", type=Path)
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
