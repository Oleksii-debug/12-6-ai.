#!/usr/bin/env python3
"""Validate/project the exact real Rada laws Q/P payload without running dedup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.rada_laws_qp_dedup_adapter import (
    UPSTREAM_FINAL_EVIDENCE_COMMIT,
    validate_and_project_rada_laws_qp,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--execution-evidence", type=Path, required=True)
    parser.add_argument(
        "--upstream-head",
        default=UPSTREAM_FINAL_EVIDENCE_COMMIT,
        help="Exact PR #655 final evidence commit; drift fails closed.",
    )
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    projection = validate_and_project_rada_laws_qp(
        args.candidate_jsonl,
        args.quality_report,
        args.execution_evidence,
        upstream_head=args.upstream_head,
        retain_payloads=False,
    )
    receipt_bytes = (
        json.dumps(
            projection.receipt,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(args.output_receipt, receipt_bytes)
    summary = {
        "status": "PASS_RADA_LAWS_QP_DEDUP_INTAKE_PROJECTED_ZERO_CREDIT",
        "receipt_identity_sha256": projection.receipt["receipt_identity_sha256"],
        "source_object_count": projection.receipt["projection"]["source_object_count"],
        "payload_utf8_bytes": projection.receipt["projection"]["payload_utf8_bytes"],
        "canonical_global_dedup_executed": False,
        "training_authorized_bytes": 0,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
