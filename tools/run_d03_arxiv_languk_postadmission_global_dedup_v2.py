#!/usr/bin/env python3
"""Run repaired ArXiv+LangUK post-admission dedup on exact incumbent V9 authority."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import compose_data526_records_from_v8 as data526
import run_d03_expanded_global_dedup_v9 as v9_runner
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data.post_admission_dedup_intake_v2 import (
    PostAdmissionDedupIntakeError,
    run_post_admission_global_dedup,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument(
        "--v8-config",
        type=Path,
        default=ROOT / "configs/data/next100_065f_global_dedup_v8.json",
    )
    parser.add_argument(
        "--data526-config",
        type=Path,
        default=ROOT / "configs/data/data526_v8_record_composition_v1.json",
    )
    parser.add_argument("--v8-report", type=Path, required=True)
    parser.add_argument("--v8-survivors", type=Path, required=True)
    parser.add_argument(
        "--data526-evidence",
        type=Path,
        default=ROOT / "evidence/data526/v8/materialization_evidence.json",
    )
    parser.add_argument(
        "--data526-record-inventory",
        type=Path,
        default=ROOT / "evidence/data526/v8/record_inventory.json",
    )
    parser.add_argument(
        "--rada-language-report",
        type=Path,
        default=ROOT / "evidence/d03-rada-trees/secondary-plaintext-language-gate-v1.json",
    )
    parser.add_argument("--rada-quality-privacy-jsonl", type=Path, required=True)
    parser.add_argument("--rada-quality-privacy-report", type=Path, required=True)
    parser.add_argument("--expected-rada-report-sha256", required=True)
    parser.add_argument(
        "--arxiv-authority",
        type=Path,
        default=(
            ROOT
            / "configs/data/d03_common_pile_arxiv_abstracts_source_admission_v1.json"
        ),
    )
    parser.add_argument("--arxiv-candidate", type=Path, required=True)
    parser.add_argument(
        "--languk-authority",
        type=Path,
        default=(
            ROOT
            / "configs/data/d03_languk_supreme_court_postexecution_rights_v1.json"
        ),
    )
    parser.add_argument("--languk-candidate", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    args = parser.parse_args()

    try:
        v8_config = v8.load_config(args.v8_config)
        data526_config = v9_runner.read_json(args.data526_config)
        data526.verify_config(data526_config, require_terminal_v8=True)
        v8_report = v9_runner.read_json(args.v8_report)
        v8_survivors = v9_runner.read_json(args.v8_survivors)
        data526.validate_v8_inputs(v8_report, v8_survivors, data526_config)

        matcher, v8_inventory, v8_payloads = v9_runner.reconstruct_v8_source_inputs(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            v8_config=v8_config,
        )
        rada_rows, rada_raw = v9_runner.read_jsonl(args.rada_quality_privacy_jsonl)

        report, survivors = run_post_admission_global_dedup(
            matcher_audit=matcher.audit_payloads,
            matcher_verify=matcher.verify_report,
            reconstructed_v8_inventory=v8_inventory,
            reconstructed_v8_payloads=v8_payloads,
            v8_survivor_authority=v8_survivors,
            data526_evidence=v9_runner.read_json(args.data526_evidence),
            data526_record_inventory=v9_runner.read_json(
                args.data526_record_inventory
            ),
            rada_language_report=v9_runner.read_json(args.rada_language_report),
            rada_quality_privacy_report=v9_runner.read_json(
                args.rada_quality_privacy_report
            ),
            expected_rada_report_sha256=args.expected_rada_report_sha256,
            rada_rows=rada_rows,
            rada_raw_jsonl=rada_raw,
            arxiv_authority_raw=args.arxiv_authority.read_bytes(),
            arxiv_candidate_raw=args.arxiv_candidate.read_bytes(),
            languk_authority_raw=args.languk_authority.read_bytes(),
            languk_candidate_raw=args.languk_candidate.read_bytes(),
        )
        v9_runner.write_json(args.output_report, report)
        v9_runner.write_json(args.output_survivors, survivors)
    except (
        PostAdmissionDedupIntakeError,
        v9_runner.ExpandedDedupError,
        data526.Data526V8Error,
        v8.V8Error,
        OSError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("D03_ARXIV_LANGUK_POSTADMISSION_GLOBAL_DEDUP_V2=PASS_ZERO_CREDIT")
    print("REPORT_SHA256=" + report["report_sha256"])
    print("SURVIVOR_AUTHORITY_SHA256=" + survivors["survivor_authority_sha256"])
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
