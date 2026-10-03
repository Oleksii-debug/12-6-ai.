"""Adapt an independently identified D03 family vector for NEXT100-106."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.postdecontam_balance_projection_v1 import (
    ProjectionError,
    load_json,
)
from twelve_six.data.postdecontam_next100_adapter_v1 import (
    adapt_family_vector_to_next100_106,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-vector", type=Path, required=True)
    parser.add_argument("--expected-family-vector-identity-sha256", required=True)
    parser.add_argument("--dedup-authority", type=Path, required=True)
    parser.add_argument("--expected-dedup-worker-id", required=True)
    parser.add_argument("--expected-dedup-head-sha", required=True)
    parser.add_argument("--expected-dedup-evidence-identity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_new_output(
    path: Path, payload: bytes, *, family_vector: Path, dedup_authority: Path,
) -> None:
    """Publish once; never truncate source evidence or an earlier result."""
    if path.exists() or path.is_symlink():
        raise ProjectionError(f"refusing to overwrite existing adapter output: {path}")
    try:
        resolved = path.resolve()
        protected = {family_vector.resolve(), dedup_authority.resolve()}
    except (OSError, RuntimeError) as exc:
        raise ProjectionError("adapter output path cannot be resolved safely") from exc
    if resolved in protected:
        raise ProjectionError("adapter output must not alias an input authority")
    try:
        with path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError as exc:
        raise ProjectionError(f"refusing to overwrite existing adapter output: {path}") from exc
    except OSError as exc:
        raise ProjectionError(f"cannot create adapter output safely: {path}: {exc}") from exc


def main() -> int:
    args = _parser().parse_args()
    try:
        result = adapt_family_vector_to_next100_106(
            load_json(args.family_vector),
            expected_family_vector_identity_sha256=(
                args.expected_family_vector_identity_sha256
            ),
            dedup_authority=load_json(args.dedup_authority),
            expected_dedup_worker_id=args.expected_dedup_worker_id,
            expected_dedup_head_sha=args.expected_dedup_head_sha,
            expected_dedup_evidence_identity_sha256=(
                args.expected_dedup_evidence_identity_sha256
            ),
        )
        payload = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        _write_new_output(
            args.output, payload.encode("utf-8"),
            family_vector=args.family_vector,
            dedup_authority=args.dedup_authority,
        )
    except ProjectionError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
