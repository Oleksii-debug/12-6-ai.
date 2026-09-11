#!/usr/bin/env python3
"""Run the historical Caselaw Product through the merged source-admission authority."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

PRODUCT_PR = 904
PRODUCT_SEMANTIC_COMMIT = "e63f2fd4eddbd348b57a42e1fb41cb658d0c3a82"
PRODUCT_CONFIG_BLOB_SHA1 = "e8b7e29ce2ab4e634c5f8c056eecb6b0c2579467"
PRODUCT_TOOL_BLOB_SHA1 = "a7fcaeee0302c46c43e5a6ae43da16363810f14a"
ADMISSION_PR = 1079
ADMISSION_MERGE_COMMIT = "b482d345867779fb8226726d73f11ec8aedd6e27"
ADMISSION_POLICY = Path("configs/data/d03_common_pile_caselaw_source_admission_v1.json")
ADMISSION_POLICY_BLOB_SHA1 = "7a9caa78095dd1ced12568edaf726ec3657f28ed"
ADMISSION_TOOL = Path("tools/verify_d03_common_pile_caselaw_source_admission.py")
ADMISSION_TOOL_BLOB_SHA1 = "fb9d68339c38c8e44959d830d414b3e4cea0b756"
HISTORICAL_CANDIDATE_SHA256 = (
    "45043f528ee83cb87a94f42ca9164c16463502308bf849a6736498d3b8caa273"
)
HISTORICAL_RETAINED_RECORDS = 5_658
HISTORICAL_RETAINED_NORMALIZED_BYTES = 5_962_147
HISTORICAL_ROWS_SCANNED = 75_518
HISTORICAL_DECOMPRESSED_BYTES = 79_993_495
REPORT_SCHEMA = "12-6.d03-caselaw-source-admission-real-replay.v1"
ADMITTED_DECISION = "CONDITIONAL_SOURCE_ADMISSION"
REQUIRED_DOWNSTREAM_GATES = (
    "CURRENT_GLOBAL_EXACT_NEAR_LINEAGE_DEDUP",
    "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
    "POST_COMPOSITION_QUALITY_PRIVACY",
    "BALANCE_AND_FAMILY_CAPS",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_PACKING_TWO_CLEAN_BUILDS",
    "POSITIVE_UNIQUE_LOSS_LEDGER",
)
CANDIDATE_KEYS = {
    "record_id",
    "normalized_sha256",
    "normalized_utf8_bytes",
    "text",
    "source_family",
    "source_kind",
    "rights_basis",
    "training_eligible",
    "evaluation_eligible",
}


class ReplayError(RuntimeError):
    """Fail-closed execution or authority mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _verify_blob(path: Path, expected: str, label: str) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReplayError(f"cannot read {label}: {path}") from exc
    require(git_blob_sha1(raw) == expected, f"{label} Git blob drift")
    return raw


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_id_digest(record_id: str) -> str:
    return sha256(b"caselaw-record-id-v1\0" + record_id.encode("utf-8"))


def _inventory_identity(record_ids: Sequence[str]) -> str:
    digests = [_record_id_digest(item) for item in record_ids]
    return sha256(canonical(digests))


def scan_source_admission(
    source_gzips: Sequence[Path],
    admission_module: ModuleType,
    policy: Mapping[str, Any],
    *,
    max_line_bytes: int,
    max_scanned_bytes: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    dispositions: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    rows_scanned = 0
    decompressed_bytes = 0
    next_scan_bytes_if_budget_exceeded: int | None = None
    scan_budget_reached = False
    stop = False

    try:
        for source_gzip in source_gzips:
            with gzip.open(source_gzip, "rb") as handle:
                while not stop:
                    line = handle.readline(max_line_bytes + 1)
                    if not line:
                        break
                    require(len(line) <= max_line_bytes, "JSONL line exceeds safety envelope")
                    prospective = decompressed_bytes + len(line)
                    if prospective > max_scanned_bytes:
                        scan_budget_reached = True
                        next_scan_bytes_if_budget_exceeded = prospective
                        stop = True
                        break
                    decompressed_bytes = prospective
                    rows_scanned += 1
                    try:
                        record = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ReplayError("invalid UTF-8 JSONL in source-admission scan") from exc
                    require(isinstance(record, Mapping), "source-admission row must be object")
                    record_id = record.get("id")
                    require(isinstance(record_id, str), "source-admission row id missing")
                    require(record_id not in decisions, f"duplicate raw record id: {record_id}")
                    result = admission_module.assess_record(record, policy)
                    require(isinstance(result, Mapping), "source-admission result must be object")
                    decision = result.get("decision")
                    reason = result.get("reason")
                    require(isinstance(decision, str), "source-admission decision missing")
                    require(isinstance(reason, str), "source-admission reason missing")
                    decisions[record_id] = dict(result)
                    dispositions[reason] = dispositions.get(reason, 0) + 1
                    source = record.get("source")
                    source_key = source if isinstance(source, str) else "<non-string>"
                    source_counts[source_key] = source_counts.get(source_key, 0) + 1
    except (OSError, EOFError) as exc:
        raise ReplayError("cannot decode exact Caselaw source vector") from exc

    summary = {
        "rows_scanned": rows_scanned,
        "decompressed_bytes_scanned": decompressed_bytes,
        "next_scan_bytes_if_budget_exceeded": next_scan_bytes_if_budget_exceeded,
        "scan_budget_reached": scan_budget_reached,
        "decision_reason_counts": dict(sorted(dispositions.items())),
        "raw_source_label_counts": dict(sorted(source_counts.items())),
    }
    return decisions, summary


def filter_candidate(
    candidate_path: Path,
    admitted_path: Path,
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        raw = candidate_path.read_bytes()
    except OSError as exc:
        raise ReplayError("cannot read historical candidate") from exc

    require(sha256(raw) == HISTORICAL_CANDIDATE_SHA256, "historical candidate hash drift")
    input_ids: list[str] = []
    retained_ids: list[str] = []
    retained_lines: list[bytes] = []
    retained_normalized_bytes = 0
    denied_by_reason: dict[str, int] = {}

    lines = raw.splitlines(keepends=True)
    require(len(lines) == HISTORICAL_RETAINED_RECORDS, "historical candidate row count drift")
    for line in lines:
        require(line.endswith(b"\n"), "candidate JSONL newline drift")
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplayError("candidate JSONL parse failure") from exc
        require(isinstance(row, Mapping), "candidate row must be object")
        require(set(row) == CANDIDATE_KEYS, "candidate row key drift")
        record_id = row.get("record_id")
        require(isinstance(record_id, str), "candidate record id missing")
        require(record_id not in input_ids, f"duplicate candidate record id: {record_id}")
        input_ids.append(record_id)
        require(
            record_id in decisions,
            f"candidate row lacks raw source-admission decision: {record_id}",
        )
        decision = decisions[record_id]
        if decision.get("decision") != ADMITTED_DECISION:
            reason = decision.get("reason")
            require(isinstance(reason, str), "denied candidate reason missing")
            denied_by_reason[reason] = denied_by_reason.get(reason, 0) + 1
            continue
        source_kind = row.get("source_kind")
        require(source_kind == "Caselaw Access Project", "admitted candidate source-kind drift")
        normalized_bytes = row.get("normalized_utf8_bytes")
        require(
            type(normalized_bytes) is int and normalized_bytes > 0,
            "candidate normalized byte count invalid",
        )
        retained_normalized_bytes += normalized_bytes
        retained_ids.append(record_id)
        retained_lines.append(line)

    total_normalized_bytes = 0
    for line in lines:
        row = json.loads(line.decode("utf-8"))
        value = row["normalized_utf8_bytes"]
        require(type(value) is int and value > 0, "historical candidate byte count invalid")
        total_normalized_bytes += value
    require(
        total_normalized_bytes == HISTORICAL_RETAINED_NORMALIZED_BYTES,
        "historical candidate normalized-byte total drift",
    )

    admitted_raw = b"".join(retained_lines)
    admitted_path.parent.mkdir(parents=True, exist_ok=True)
    admitted_path.write_bytes(admitted_raw)
    return {
        "input_records": len(input_ids),
        "input_normalized_utf8_bytes": total_normalized_bytes,
        "input_candidate_sha256": sha256(raw),
        "input_record_id_inventory_sha256": _inventory_identity(input_ids),
        "source_admitted_records": len(retained_ids),
        "source_admitted_normalized_utf8_bytes": retained_normalized_bytes,
        "source_admitted_candidate_sha256": sha256(admitted_raw),
        "source_admitted_record_id_inventory_sha256": _inventory_identity(retained_ids),
        "candidate_denied_reason_counts": dict(sorted(denied_by_reason.items())),
    }


def execute(
    *,
    product_tool_path: Path,
    product_config_path: Path,
    admission_tool_path: Path,
    admission_policy_path: Path,
    source_gzips: Sequence[Path],
    candidate_path: Path,
    admitted_candidate_path: Path,
    materializer_report_path: Path,
    report_path: Path,
    download: bool,
) -> dict[str, Any]:
    _verify_blob(product_tool_path, PRODUCT_TOOL_BLOB_SHA1, "historical Product materializer")
    _verify_blob(product_config_path, PRODUCT_CONFIG_BLOB_SHA1, "historical Product config")
    _verify_blob(admission_tool_path, ADMISSION_TOOL_BLOB_SHA1, "source-admission verifier")
    _verify_blob(admission_policy_path, ADMISSION_POLICY_BLOB_SHA1, "source-admission policy")

    product = _load_module(product_tool_path, "caselaw_product_exact")
    admission = _load_module(admission_tool_path, "caselaw_admission_exact")
    product_cfg = product.load_config(product_config_path)
    policy = admission.load_policy(admission_policy_path)
    admission.validate_candidate_identity(policy["candidate_binding"], policy)

    source_gzips = list(source_gzips)
    require(len(source_gzips) == 2, "exact two-object source vector required")
    if download:
        product.download_exact(product_cfg, source_gzips)
    materializer_report = product.materialize(
        source_gzips,
        candidate_path,
        materializer_report_path,
        product_cfg,
    )
    require(
        materializer_report.get("retained_records") == HISTORICAL_RETAINED_RECORDS,
        "historical Product retained-record count drift",
    )
    require(
        materializer_report.get("retained_normalized_utf8_bytes")
        == HISTORICAL_RETAINED_NORMALIZED_BYTES,
        "historical Product retained-byte count drift",
    )
    require(
        materializer_report.get("candidate_jsonl_sha256") == HISTORICAL_CANDIDATE_SHA256,
        "historical Product candidate hash drift",
    )
    require(
        materializer_report.get("source_rows_scanned") == HISTORICAL_ROWS_SCANNED,
        "historical Product row-scan drift",
    )
    require(
        materializer_report.get("decompressed_bytes_scanned") == HISTORICAL_DECOMPRESSED_BYTES,
        "historical Product scan-byte drift",
    )
    decisions, raw_summary = scan_source_admission(
        source_gzips,
        admission,
        policy,
        max_line_bytes=product.MAX_LINE_BYTES,
        max_scanned_bytes=product.MAX_SCANNED_BYTES,
    )
    require(
        raw_summary["rows_scanned"] == HISTORICAL_ROWS_SCANNED,
        "source-admission raw row-scan drift",
    )
    require(
        raw_summary["decompressed_bytes_scanned"] == HISTORICAL_DECOMPRESSED_BYTES,
        "source-admission raw byte-scan drift",
    )
    filtered = filter_candidate(candidate_path, admitted_candidate_path, decisions)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "product_pr": PRODUCT_PR,
        "product_semantic_commit_sha": PRODUCT_SEMANTIC_COMMIT,
        "product_config_git_blob_sha1": PRODUCT_CONFIG_BLOB_SHA1,
        "product_materializer_git_blob_sha1": PRODUCT_TOOL_BLOB_SHA1,
        "source_admission_pr": ADMISSION_PR,
        "source_admission_merge_commit_sha": ADMISSION_MERGE_COMMIT,
        "source_admission_policy_git_blob_sha1": ADMISSION_POLICY_BLOB_SHA1,
        "source_admission_verifier_git_blob_sha1": ADMISSION_TOOL_BLOB_SHA1,
        "source_objects": materializer_report["source_objects"],
        "source_ordering_policy": materializer_report["source_ordering_policy"],
        "historical_product_replay_reproduced": True,
        "historical_product_rows_scanned": HISTORICAL_ROWS_SCANNED,
        "historical_product_decompressed_bytes_scanned": HISTORICAL_DECOMPRESSED_BYTES,
        **raw_summary,
        **filtered,
        "source_admission_executed": True,
        "source_admission_eligible_candidate_only": True,
        "global_dedup_completed": False,
        "reserved_evaluation_decontamination_completed": False,
        "post_composition_privacy_complete": False,
        "canonical_capacity_credited": 0,
        "family_credit_added": 0,
        "training_authorized_bytes": 0,
        "unique_causal_loss_positions_authorized": 0,
        "tokenizer_fit_authorized": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "learned_weights_created": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
        "required_downstream_gates": list(REQUIRED_DOWNSTREAM_GATES),
        "candidate_text_retained_in_durable_evidence": False,
        "safe_result": "CASELAW_SOURCE_ADMITTED_CANDIDATE_ONLY_ZERO_CREDIT",
    }
    report["report_identity_sha256"] = sha256(canonical(report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-tool", type=Path, required=True)
    parser.add_argument("--product-config", type=Path, required=True)
    parser.add_argument("--admission-tool", type=Path, default=ADMISSION_TOOL)
    parser.add_argument("--admission-policy", type=Path, default=ADMISSION_POLICY)
    parser.add_argument("--source-gzip", type=Path, action="append", required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--admitted-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--materializer-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    report = execute(
        product_tool_path=args.product_tool,
        product_config_path=args.product_config,
        admission_tool_path=args.admission_tool,
        admission_policy_path=args.admission_policy,
        source_gzips=args.source_gzip,
        candidate_path=args.candidate_jsonl,
        admitted_candidate_path=args.admitted_candidate_jsonl,
        materializer_report_path=args.materializer_report,
        report_path=args.report,
        download=args.download,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
