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
    parser.add_argument("--family-map", type=Path, required=True)
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
            family_map=load_json(args.family_map),
            source_git_sha=args.source_git_sha,
        )
        write_family_vector(args.output, result)
    except ProjectionError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc
    print(result["family_vector_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
