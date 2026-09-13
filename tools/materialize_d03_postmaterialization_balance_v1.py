"""Materialize authenticated post-G05/G06 family-vector and balance binding evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postmaterialization_balance_projection_v1 import (
    build_balance_result_binding,
    build_postmaterialization_family_vector,
    require_balanced_selection_ready,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProjectionError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    vector = commands.add_parser("family-vector")
    vector.add_argument("--inventory", type=Path, required=True)
    vector.add_argument("--materialization-evidence", type=Path, required=True)
    vector.add_argument("--expected-execution-head-sha", required=True)
    vector.add_argument("--expected-materialization-identity-sha256", required=True)
    vector.add_argument("--expected-result-jsonl-sha256", required=True)
    vector.add_argument("--expected-record-count", type=int, required=True)
    vector.add_argument("--expected-total-payload-bytes", type=int, required=True)
    vector.add_argument("--expected-source-object-count", type=int, required=True)
    vector.add_argument(
        "--expected-record-inventory-digest-sha256",
        required=True,
    )
    vector.add_argument(
        "--expected-payload-inventory-digest-sha256",
        required=True,
    )
    vector.add_argument("--source-git-sha", required=True)
    vector.add_argument("--output", type=Path, required=True)

    binding = commands.add_parser("bind-result")
    binding.add_argument("--family-vector", type=Path, required=True)
    binding.add_argument("--expected-family-vector-identity-sha256", required=True)
    binding.add_argument("--next100-input", type=Path, required=True)
    binding.add_argument("--balance-result", type=Path, required=True)
    binding.add_argument("--expected-policy-identity-sha256", required=True)
    binding.add_argument("--expected-result-identity-sha256", required=True)
    binding.add_argument("--output", type=Path, required=True)
    binding.add_argument(
        "--require-selection-ready",
        action="store_true",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "family-vector":
            result = build_postmaterialization_family_vector(
                inventory=_load(args.inventory),
                materialization_evidence=_load(args.materialization_evidence),
                expected_execution_head_sha=args.expected_execution_head_sha,
                expected_materialization_identity_sha256=(
                    args.expected_materialization_identity_sha256
                ),
                expected_result_jsonl_sha256=args.expected_result_jsonl_sha256,
                expected_record_count=args.expected_record_count,
                expected_total_payload_bytes=args.expected_total_payload_bytes,
                expected_source_object_count=args.expected_source_object_count,
                expected_record_inventory_digest_sha256=(
                    args.expected_record_inventory_digest_sha256
                ),
                expected_payload_inventory_digest_sha256=(
                    args.expected_payload_inventory_digest_sha256
                ),
                source_git_sha=args.source_git_sha,
            )
            _write(args.output, result)
            print(result["family_vector_identity_sha256"])
            return 0

        family_vector = _load(args.family_vector)
        binding = build_balance_result_binding(
            family_vector=family_vector,
            expected_family_vector_identity_sha256=(
                args.expected_family_vector_identity_sha256
            ),
            next100_input=_load(args.next100_input),
            balance_result=_load(args.balance_result),
            expected_policy_identity_sha256=args.expected_policy_identity_sha256,
            expected_result_identity_sha256=args.expected_result_identity_sha256,
        )
        if args.require_selection_ready:
            require_balanced_selection_ready(
                binding,
                expected_binding_identity_sha256=binding[
                    "binding_identity_sha256"
                ],
            )
        _write(args.output, binding)
        print(binding["binding_identity_sha256"])
        return 0
    except ProjectionError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
