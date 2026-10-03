"""Materialize the exact post-decontamination family-vector handoff for NEXT100-106."""

from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.data.postdecontam_balance_projection_v1 import (
    ProjectionError,
    build_family_vector,
    load_json,
    read_records_jsonl,
    write_family_vector,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--decontamination-binding", type=Path, required=True)
    parser.add_argument("--g05-g06-coverage", type=Path, required=True)
    parser.add_argument("--expected-g05-g06-coverage-identity-sha256", required=True)
    parser.add_argument("--expected-privacy-policy-identity-sha256", required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--expected-family-map-identity-sha256", required=True)
    parser.add_argument("--family-provenance", type=Path, required=True)
    parser.add_argument("--expected-family-provenance-identity-sha256", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        records, records_sha256 = read_records_jsonl(args.records)
        result = build_family_vector(
            records=records,
            records_jsonl_sha256=records_sha256,
            decontamination_binding=load_json(args.decontamination_binding),
            g05_g06_coverage=load_json(args.g05_g06_coverage),
            expected_g05_g06_coverage_identity_sha256=(
                args.expected_g05_g06_coverage_identity_sha256
            ),
            expected_privacy_policy_identity_sha256=(
                args.expected_privacy_policy_identity_sha256
            ),
            family_map=load_json(args.family_map),
            expected_family_map_identity_sha256=args.expected_family_map_identity_sha256,
            family_provenance=load_json(args.family_provenance),
            expected_family_provenance_identity_sha256=(
                args.expected_family_provenance_identity_sha256
            ),
            source_git_sha=args.source_git_sha,
        )
        write_family_vector(args.output, result)
    except ProjectionError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc
    print(result["family_vector_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
