#!/usr/bin/env python3
"""Converge independent Rada_Trees language and quality/privacy authorities.

This seam is deliberately zero-credit and fail-closed.  It binds the exact terminal
#917 language report to the exact candidate JSONL consumed by #916 quality/privacy,
reconstructs the rights-scoped inventory from source-native record metadata, verifies
the quality/privacy output is an exact subset of that candidate inventory, and only
then sets the combined language+quality+privacy completion bit on retained records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "12-6.d03-rada-trees-language-quality-privacy-convergence.v1"
EXPECTED_DATASET = "uacorpus/Rada_Trees"
EXPECTED_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
EXPECTED_FAMILY = "ua.rada.open-data.plenary-transcripts"
EXPECTED_RECORDS = 4384
EXPECTED_SOURCE_BYTES = 877_899_128
EXPECTED_RIGHTS_INVENTORY = "7b93056f38fbc87c11e14df9069066e21380db1f1b8d8016314330f841bfa6fc"
EXPECTED_LANGUAGE_REPORT = "adb89b227d2b0631ca7b35ba59ff744d2efa74ec9c9f6f2f8334738213074291"
EXPECTED_LANGUAGE_DECISION_INVENTORY = "7b0c722ed53283decb5248e1ee2d28ff943f258ea98c3112e36dd843ae1c1e90"


class ConvergenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConvergenceError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConvergenceError(f"cannot load JSON authority: {path}") from exc
    require(isinstance(value, dict), f"authority root must be object: {path}")
    return value


def verify_self_hash(report: Mapping[str, Any], *, label: str) -> str:
    claimed = report.get("report_sha256")
    require(isinstance(claimed, str) and len(claimed) == 64, f"{label} report hash missing")
    core = dict(report)
    core.pop("report_sha256", None)
    require(sha256_bytes(canonical_bytes(core)) == claimed, f"{label} report self-hash invalid")
    return claimed


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    require(path.is_file() and not path.is_symlink(), f"JSONL must be regular file: {path}")
    raw = path.read_bytes()
    require(raw, f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw.splitlines(), 1):
        require(line.strip(), f"blank JSONL line at {line_no}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConvergenceError(f"invalid JSONL at line {line_no}: {path}") from exc
        require(isinstance(row, dict), f"row {line_no} must be object")
        rid = row.get("record_id")
        require(isinstance(rid, str) and rid, f"record_id missing at line {line_no}")
        require(rid not in seen, f"duplicate record_id: {rid}")
        seen.add(rid)
        rows.append(row)
    return rows, raw


def rights_inventory(rows: list[dict[str, Any]]) -> tuple[str, int]:
    compact: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        rid = str(row.get("record_id", ""))
        require(row.get("source_dataset") == EXPECTED_DATASET, f"dataset drift: {rid}")
        require(row.get("source_revision") == EXPECTED_REVISION, f"revision drift: {rid}")
        require(row.get("source_family") == EXPECTED_FAMILY, f"family drift: {rid}")
        path = row.get("source_path")
        digest = row.get("source_payload_sha256")
        size = row.get("source_payload_bytes")
        session_date = row.get("session_date")
        require(isinstance(path, str) and path, f"source_path missing: {rid}")
        require(isinstance(digest, str) and len(digest) == 64, f"source SHA missing: {rid}")
        require(isinstance(size, int) and size >= 0, f"source bytes malformed: {rid}")
        require(isinstance(session_date, str) and session_date, f"session_date missing: {rid}")
        compact.append({"path": path, "size_bytes": size, "sha256": digest, "date": session_date})
        total += size
    compact.sort(key=lambda row: str(row["path"]))
    return sha256_bytes(canonical_bytes(compact)), total


def converge(candidate_path: Path, qp_path: Path, qp_report_path: Path, language_report_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    candidate, candidate_raw = read_jsonl(candidate_path)
    require(len(candidate) == EXPECTED_RECORDS, "candidate record count drift")
    inventory, source_bytes = rights_inventory(candidate)
    require(inventory == EXPECTED_RIGHTS_INVENTORY, "candidate rights inventory drift")
    require(source_bytes == EXPECTED_SOURCE_BYTES, "candidate source-byte total drift")

    language = load_json(language_report_path)
    require(language.get("schema_version") == "12-6.d03-rada-trees-language-gate-report.v1", "language schema drift")
    require(verify_self_hash(language, label="language") == EXPECTED_LANGUAGE_REPORT, "language terminal identity drift")
    source = language.get("source")
    require(isinstance(source, Mapping), "language source binding missing")
    require(source.get("dataset") == EXPECTED_DATASET, "language dataset drift")
    require(source.get("dataset_revision") == EXPECTED_REVISION, "language revision drift")
    parent = language.get("parent_provenance_rights")
    require(isinstance(parent, Mapping), "language rights parent missing")
    require(parent.get("accepted_path_inventory_sha256") == inventory, "language/candidate rights inventory mismatch")
    result = language.get("language_result")
    require(isinstance(result, Mapping), "language result missing")
    require(result.get("language_pass_members") == EXPECTED_RECORDS, "language pass count incomplete")
    require(result.get("language_reject_members") == 0, "language rejects present")
    require(result.get("language_pass_bytes") == EXPECTED_SOURCE_BYTES, "language pass bytes drift")
    require(result.get("language_decision_inventory_sha256") == EXPECTED_LANGUAGE_DECISION_INVENTORY, "language decision inventory drift")

    qp_report = load_json(qp_report_path)
    require(qp_report.get("schema_version") == "12-6.d03-rada-trees-quality-privacy-gate.v2", "quality/privacy schema drift")
    verify_self_hash(qp_report, label="quality/privacy")
    require(qp_report.get("input_candidate_jsonl_sha256") == sha256_bytes(candidate_raw), "quality/privacy input identity mismatch")
    require(qp_report.get("input_records") == EXPECTED_RECORDS, "quality/privacy input count drift")
    boundary = qp_report.get("claim_boundary")
    require(isinstance(boundary, Mapping), "quality/privacy claim boundary missing")
    require(boundary.get("language_authority_bound") is False, "quality/privacy report already claims language binding")
    require(boundary.get("training_authorized_bytes") == 0, "premature training credit")

    qp_rows, qp_raw = read_jsonl(qp_path)
    require(qp_report.get("output_jsonl_sha256") == sha256_bytes(qp_raw), "quality/privacy output identity mismatch")
    require(qp_report.get("output_records") == len(qp_rows), "quality/privacy output count mismatch")
    by_id = {str(row["record_id"]): row for row in candidate}
    converged: list[dict[str, Any]] = []
    for row in qp_rows:
        rid = str(row["record_id"])
        require(rid in by_id, f"quality/privacy output is not candidate subset: {rid}")
        parent_row = by_id[rid]
        for key in ("source_path", "source_payload_sha256", "source_payload_bytes", "session_date", "decoded_text_utf8_sha256"):
            require(row.get(key) == parent_row.get(key), f"source identity mutation in quality/privacy output: {rid}:{key}")
        require(row.get("quality_privacy_complete") is True, f"quality/privacy incomplete: {rid}")
        require(row.get("language_quality_privacy_complete") is False, f"combined gate already set before convergence: {rid}")
        require(row.get("training_eligible") is False and row.get("evaluation_eligible") is False, f"premature eligibility: {rid}")
        clean = dict(row)
        clean["language_quality_privacy_complete"] = True
        clean["language_authority_report_sha256"] = EXPECTED_LANGUAGE_REPORT
        clean["language_decision_inventory_sha256"] = EXPECTED_LANGUAGE_DECISION_INVENTORY
        clean["training_eligible"] = False
        clean["evaluation_eligible"] = False
        converged.append(clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_line(row) for row in converged)
    output_path.write_bytes(payload)
    core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "candidate_jsonl_sha256": sha256_bytes(candidate_raw),
        "rights_inventory_sha256": inventory,
        "language_report_sha256": EXPECTED_LANGUAGE_REPORT,
        "language_decision_inventory_sha256": EXPECTED_LANGUAGE_DECISION_INVENTORY,
        "quality_privacy_report_sha256": qp_report["report_sha256"],
        "quality_privacy_output_jsonl_sha256": sha256_bytes(qp_raw),
        "output_records": len(converged),
        "output_jsonl_sha256": sha256_bytes(payload),
        "claim_boundary": {
            "language_quality_privacy_complete_only_for_output_records": True,
            "global_dedup_complete": False,
            "reserved_evaluation_decontamination_complete": False,
            "family_caps_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
        "safe_result": "RADA_TREES_LANGUAGE_QUALITY_PRIVACY_CONVERGED_ZERO_CREDIT",
    }
    report = {**core, "report_sha256": sha256_bytes(canonical_bytes(core))}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_line(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--quality-privacy-jsonl", type=Path, required=True)
    parser.add_argument("--quality-privacy-report", type=Path, required=True)
    parser.add_argument("--language-report", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = converge(args.candidate_jsonl, args.quality_privacy_jsonl, args.quality_privacy_report, args.language_report, args.output_jsonl, args.report)
    except (ConvergenceError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_TREES_LANGUAGE_QUALITY_PRIVACY=PASS_ZERO_CREDIT")
    print("OUTPUT_RECORDS=" + str(report["output_records"]))
    print("REPORT_SHA256=" + report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
