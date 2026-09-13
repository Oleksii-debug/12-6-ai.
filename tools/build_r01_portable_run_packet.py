#!/usr/bin/env python3
"""Build an evidence-bound portable run packet without self-authorizing compute."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from twelve_six.portable_run_binding import bind_portable_run_packet, canonical_sha256
from twelve_six.portable_run_packet import assess_portable_run_packet
from twelve_six.preoptimizer_authority import (
    bind_preoptimizer_to_packet,
    canonical_sha256 as preoptimizer_sha256,
)
from twelve_six.readiness_trust_root import authenticated_trusted_launch_bundle

DEFAULT_READINESS = Path("configs/research/r01_learned20m_launch_readiness_v1.json")
DEFAULT_TEMPLATE = Path("configs/research/r01_portable_local_free_run_packet_v1.json")
DEFAULT_OVERLAY = Path("configs/research/r01_portable_session_overlay_v1.json")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    """Create, fsync and atomically publish; never overwrite an existing packet."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trusted-bindings",
        type=Path,
        help=(
            "separate trusted readiness bundle; this is only accepted with an "
            "independently supplied expected SHA-256"
        ),
    )
    parser.add_argument(
        "--expected-trusted-bindings-sha256",
        help=(
            "independently supplied SHA-256 identity for --trusted-bindings; "
            "never derive this expectation from readiness or bundle input"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        readiness = _load_object(args.readiness)
        template = _load_object(args.template)
        overlay = _load_object(args.overlay)

        verified_scientific: set[str] = set()
        verified_refs: set[str] = set()
        portable_execution: dict[str, Any] | None = None
        preoptimizer_authorities: dict[str, Any] | None = None
        if args.trusted_bindings is None:
            if args.expected_trusted_bindings_sha256 is not None:
                raise ValueError(
                    "--expected-trusted-bindings-sha256 requires --trusted-bindings"
                )
        else:
            if args.expected_trusted_bindings_sha256 is None:
                raise ValueError(
                    "--trusted-bindings requires --expected-trusted-bindings-sha256"
                )
            bindings = _load_object(args.trusted_bindings)
            resolved = authenticated_trusted_launch_bundle(
                bindings,
                expected_identity_sha256=args.expected_trusted_bindings_sha256,
            )
            if resolved is None:
                raise ValueError(
                    "trusted bindings are malformed or do not match the independent "
                    "expected identity"
                )
            (
                verified_scientific,
                verified_refs,
                portable_execution,
                preoptimizer_authorities,
            ) = resolved

        result = bind_portable_run_packet(
            readiness,
            template,
            overlay,
            expected_portable_execution=portable_execution,
            verified_scientific_authorities=verified_scientific,
            verified_authorization_refs=verified_refs,
        )
        report = result.as_dict()
        report["output_written"] = False
        packet_to_write: dict[str, Any] | None = None

        if result.binding_ready:
            if (
                preoptimizer_authorities is None
                or args.expected_trusted_bindings_sha256 is None
            ):
                raise ValueError(
                    "ready portable launch requires schema-v3 trusted bindings with "
                    "preoptimizer authorities"
                )
            assert result.packet is not None
            packet_to_write = bind_preoptimizer_to_packet(
                result.packet,
                preoptimizer_authorities,
                trusted_readiness_bundle_sha256=args.expected_trusted_bindings_sha256,
            )
            assessment = assess_portable_run_packet(packet_to_write)
            packet_ready = (
                result.mode == "FRESH_START"
                and assessment.ready_for_initial_local_free_launch
            ) or (
                result.mode == "RESUME"
                and assessment.ready_for_cross_provider_resume
            )
            if not assessment.contract_valid or not packet_ready:
                raise ValueError(
                    "preoptimizer-bound packet failed the existing portable run contract"
                )
            report["packet_sha256"] = canonical_sha256(packet_to_write)
            report["preoptimizer_authorities_sha256"] = preoptimizer_sha256(
                preoptimizer_authorities
            )
            report["trusted_readiness_bundle_sha256"] = (
                args.expected_trusted_bindings_sha256
            )

        if result.binding_ready and args.output is not None:
            assert packet_to_write is not None
            _atomic_create(args.output, packet_to_write)
            report["output_written"] = True
            report["output_path"] = str(args.output)
        elif result.binding_ready:
            report["output_path"] = None
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.binding_ready else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"binding_ready": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
