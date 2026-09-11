#!/usr/bin/env python3
"""Run post-V9 global dedup with exact real PEP+LoC materialization bytes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import compose_data526_records_from_v8 as data526
import run_d03_expanded_global_dedup_v9 as v9_runner
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data.expanded_global_dedup_v10 import (
    ExpandedDedupV10Error,
    run_expanded_dedup_v10,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExpandedDedupV10Error(f"JSON root must be object: {path}")
    return value


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ExpandedDedupV10Error(f"JSONL must be a regular file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ExpandedDedupV10Error(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise ExpandedDedupV10Error(f"blank JSONL line at {line_no}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExpandedDedupV10Error(f"JSONL row {line_no} must be object")
        rows.append(value)
    return rows, raw


def read_exact_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ExpandedDedupV10Error(f"{label} must be a regular file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ExpandedDedupV10Error(f"{label} is empty: {path}")
    return raw


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ExpandedDedupV10Error(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
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
        default=ROOT
        / "evidence/d03-rada-trees/secondary-plaintext-language-gate-v1.json",
    )
    parser.add_argument("--rada-quality-privacy-jsonl", type=Path, required=True)
    parser.add_argument("--rada-quality-privacy-report", type=Path, required=True)
    parser.add_argument("--expected-rada-report-sha256", required=True)
    parser.add_argument("--pep-candidate", type=Path, required=True)
    parser.add_argument("--pep-report", type=Path, required=True)
    parser.add_argument("--loc-candidate", type=Path, required=True)
    parser.add_argument("--loc-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    args = parser.parse_args()

    try:
        v8_config = v8.load_config(args.v8_config)
        data526_config = read_json(args.data526_config)
        data526.verify_config(data526_config, require_terminal_v8=True)
        v8_report = read_json(args.v8_report)
        v8_survivors = read_json(args.v8_survivors)
        data526.validate_v8_inputs(v8_report, v8_survivors, data526_config)
        matcher_module, v8_inventory, v8_payloads = v9_runner.reconstruct_v8_source_inputs(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            v8_config=v8_config,
        )
        rada_rows, rada_raw = read_jsonl(args.rada_quality_privacy_jsonl)
        report, survivors = run_expanded_dedup_v10(
            matcher_audit=matcher_module.audit_payloads,
            matcher_verify=matcher_module.verify_report,
            reconstructed_v8_inventory=v8_inventory,
            reconstructed_v8_payloads=v8_payloads,
            v8_survivor_authority=v8_survivors,
            data526_evidence=read_json(args.data526_evidence),
            data526_record_inventory=read_json(args.data526_record_inventory),
            rada_language_report=read_json(args.rada_language_report),
            rada_quality_privacy_report=read_json(args.rada_quality_privacy_report),
            expected_rada_report_sha256=args.expected_rada_report_sha256,
            rada_rows=rada_rows,
            rada_raw_jsonl=rada_raw,
            pep_candidate_raw=read_exact_bytes(
                args.pep_candidate,
                label="PEP candidate",
            ),
            pep_report_raw=read_exact_bytes(args.pep_report, label="PEP report"),
            loc_candidate_raw=read_exact_bytes(
                args.loc_candidate,
                label="LoC candidate",
            ),
            loc_report_raw=read_exact_bytes(args.loc_report, label="LoC report"),
        )
        write_json(args.output_report, report)
        write_json(args.output_survivors, survivors)
    except (
        ExpandedDedupV10Error,
        data526.Data526V8Error,
        v8.V8Error,
        v9_runner.ExpandedDedupError,
        OSError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("D03_EXPANDED_GLOBAL_DEDUP_V10_PEP_LOC=PASS_ZERO_CREDIT")
    print("REPORT_SHA256=" + report["report_sha256"])
    print("SURVIVOR_AUTHORITY_SHA256=" + survivors["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
