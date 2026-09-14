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
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except ProjectionError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
