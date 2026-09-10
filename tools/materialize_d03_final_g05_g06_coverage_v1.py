#!/usr/bin/env python3
"""Materialize the text-free D03 final G05/G06 coverage authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.data.final_g05_g06_coverage_v1 import build_final_g05_g06_coverage


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-inventory", required=True, type=Path)
    parser.add_argument("--expected-retained-inventory-identity", required=True)
    parser.add_argument("--decontamination-binding", required=True, type=Path)
    parser.add_argument("--expected-decontamination-authority", required=True)
    parser.add_argument("--expected-records-jsonl-sha256", required=True)
    parser.add_argument(
        "--qualification-authority",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--expected-qualification-authority-identity",
        required=True,
        action="append",
    )
    parser.add_argument("--expected-privacy-policy-identity", required=True)
    parser.add_argument("--expected-privacy-implementation-git-blob", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = build_final_g05_g06_coverage(
        retained_inventory=_load(args.retained_inventory),
        expected_retained_inventory_identity_sha256=(
            args.expected_retained_inventory_identity
        ),
        decontamination_binding=_load(args.decontamination_binding),
        expected_decontamination_authority_sha256=(
            args.expected_decontamination_authority
        ),
        expected_records_jsonl_sha256=args.expected_records_jsonl_sha256,
        qualification_authorities=[
            _load(path) for path in args.qualification_authority
        ],
        expected_qualification_authority_identities_sha256=(
            args.expected_qualification_authority_identity
        ),
        expected_privacy_policy_identity_sha256=(
            args.expected_privacy_policy_identity
        ),
        expected_privacy_implementation_git_blob_sha=(
            args.expected_privacy_implementation_git_blob
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(result["g05_g06_coverage_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
