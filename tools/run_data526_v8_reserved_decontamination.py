#!/usr/bin/env python3
"""Execute exact DATA-526 V8 reserved-evaluation decontamination from local payloads.

All payload-bearing inputs are ephemeral runtime files. Only the incumbent DATA-232
hash-only report and the DATA-526 wrapper evidence are written as durable outputs.
The caller must provide every independently expected authority identity explicitly;
this runner never upgrades a self-consistent input into external authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.data.data526_v8_reserved_decontamination_v1 import (
    execute_data526_v8_reserved_decontamination,
    verify_data526_v8_execution_evidence,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {line_number} must be an object: {path}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run record-bound DATA-526 V8 reserved-evaluation decontamination"
    )
    parser.add_argument("--training-records-jsonl", type=Path, required=True)
    parser.add_argument("--record-inventory-json", type=Path, required=True)
    parser.add_argument("--materialization-evidence-json", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--reserved-binding-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--execution-evidence-json", type=Path, required=True)
    parser.add_argument("--expected-record-inventory-digest-sha256", required=True)
    parser.add_argument("--expected-payload-inventory-digest-sha256", required=True)
    parser.add_argument("--expected-materialization-evidence-identity-sha256", required=True)
    parser.add_argument("--expected-survivor-authority-sha256", required=True)
    parser.add_argument("--expected-reserved-binding-identity-sha256", required=True)
    parser.add_argument("--expected-selection-validation-identity-sha256", required=True)
    parser.add_argument("--expected-final-test-identity-sha256", required=True)
    parser.add_argument(
        "--allow-cross-source-family-without-quarantine",
        action="store_true",
        help="Disable incumbent conservative cross-source-family quarantine (test/debug only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, evidence = execute_data526_v8_reserved_decontamination(
        _load_jsonl(args.training_records_jsonl),
        _load_jsonl(args.evaluation_records_jsonl),
        record_inventory=_load_json(args.record_inventory_json),
        materialization_evidence=_load_json(args.materialization_evidence_json),
        reserved_payload_binding=_load_json(args.reserved_binding_json),
        expected_record_inventory_digest_sha256=(
            args.expected_record_inventory_digest_sha256
        ),
        expected_payload_inventory_digest_sha256=(
            args.expected_payload_inventory_digest_sha256
        ),
        expected_materialization_evidence_identity_sha256=(
            args.expected_materialization_evidence_identity_sha256
        ),
        expected_survivor_authority_sha256=args.expected_survivor_authority_sha256,
        expected_reserved_binding_identity_sha256=(
            args.expected_reserved_binding_identity_sha256
        ),
        expected_selection_validation_identity_sha256=(
            args.expected_selection_validation_identity_sha256
        ),
        expected_final_test_identity_sha256=args.expected_final_test_identity_sha256,
        quarantine_cross_source_families=(
            not args.allow_cross_source_family_without_quarantine
        ),
    )
    verify_data526_v8_execution_evidence(evidence)
    _write_json(args.report_json, report)
    _write_json(args.execution_evidence_json, evidence)
    print(evidence["execution_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
