"""Fail-closed expanded global dedup composition for the current D03 graph.

This module does not implement duplicate matching. It validates the sealed DATA-526
V8 authority plus one externally anchored Rada_Trees quality/privacy execution,
composes their exact payloads, and delegates matching to the incumbent #824 V3 API.
All outputs remain zero-credit until later decontamination/balance/split/pack gates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

REPORT_SCHEMA = "12-6.d03-expanded-global-dedup-v9-report.v1"
SURVIVOR_SCHEMA = "12-6.d03-expanded-global-dedup-v9-survivors.v1"
RADA_QP_SCHEMA = "12-6.d03-rada-trees-quality-window-authority-bound.v1"
RADA_QP_SAFE_RESULT = "RADA_TREES_QUALITY_WINDOWS_AUTHORITY_BOUND_ZERO_CREDIT"
RADA_LANGUAGE_SCHEMA = "12-6.d03-rada-trees-language-gate-report.v1"
RADA_LANGUAGE_REPORT_SHA256 = (
    "adb89b227d2b0631ca7b35ba59ff744d2efa74ec9c9f6f2f8334738213074291"
)
RADA_RIGHTS_INVENTORY_SHA256 = (
    "7b93056f38fbc87c11e14df9069066e21380db1f1b8d8016314330f841bfa6fc"
)
RADA_DATASET = "uacorpus/Rada_Trees"
RADA_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
RADA_FAMILY = "ua.rada.open-data.plenary-transcripts"
RADA_INPUT_RECORDS = 4_384
RADA_SOURCE_BYTES = 877_899_128

V8_REPORT_SHA256 = "942cc15af60ee36a79345beba77fec347e33ee919d8148fb319b19d9ee072e5a"
V8_NESTED_V3_SHA256 = "e7244cb7f6df6062dc838112b1e3e6fe87e41644a9e8b0071e93c50f1166b67d"
V8_SURVIVOR_SHA256 = "8115b35662fa89900b37f54f7cffd5f2fa8b2f8c60a72bcb18b89c92aaade0cf"
DATA526_EVIDENCE_SHA256 = "aa69822a50d99d0804b6c91c9d576e23d590db7a428b62cf1ae149f56da8ad32"
DATA526_RECORD_INVENTORY_SHA256 = (
    "766cc01d610373183e73a632e18674b1d7afc5aaa7e3888638029e0044b26103"
)
DATA526_PAYLOAD_INVENTORY_SHA256 = (
    "56ef7a457f4c4f649ad97359c2a177131632d838756d769e4b1a2ec3f3a2a278"
)
DATA526_RECORDS = 275
DATA526_SOURCES = 262
DATA526_BYTES = 6_095_321
SELECTION_RULE = "largest_declared_capacity_then_lexicographically_smallest_source_id"


class ExpandedDedupError(RuntimeError):
    """Raised when an upstream authority or composition invariant fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpandedDedupError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any, *, ascii_only: bool = False, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ascii_only,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _verify_self_hash(
    report: Mapping[str, Any],
    *,
    label: str,
    expected: str | None = None,
    ascii_only: bool = False,
    newline: bool = False,
) -> str:
    claimed = report.get("report_sha256")
    _require(_is_sha256(claimed), f"{label} report SHA-256 missing")
    core = dict(report)
    core.pop("report_sha256", None)
    observed = _sha256(_canonical(core, ascii_only=ascii_only, newline=newline))
    _require(claimed == observed, f"{label} report self-hash mismatch")
    if expected is not None:
        _require(claimed == expected, f"{label} report identity mismatch")
    return str(claimed)


def validate_data526_authority(
    evidence: Mapping[str, Any], record_inventory: Mapping[str, Any]
) -> None:
    core = dict(evidence)
    claimed = core.pop("evidence_identity_sha256", None)
    _require(claimed == DATA526_EVIDENCE_SHA256, "DATA-526 evidence identity drift")
    _require(_sha256(_canonical(core)) == claimed, "DATA-526 evidence self-hash mismatch")
    exact = {
        "v8_report_sha256": V8_REPORT_SHA256,
        "v8_nested_v3_report_sha256": V8_NESTED_V3_SHA256,
        "v8_survivor_authority_sha256": V8_SURVIVOR_SHA256,
        "record_inventory_digest_sha256": DATA526_RECORD_INVENTORY_SHA256,
        "payload_inventory_digest_sha256": DATA526_PAYLOAD_INVENTORY_SHA256,
        "record_count": DATA526_RECORDS,
        "source_object_count": DATA526_SOURCES,
        "total_payload_bytes": DATA526_BYTES,
    }
    for key, value in exact.items():
        _require(evidence.get(key) == value, f"DATA-526 evidence drift: {key}")
    _require(evidence.get("decontamination_executed") is False, "premature decontam claim")
    _require(evidence.get("authorized_unique_optimized_targets") == 0, "premature target credit")
    _require(evidence.get("tokenizer_fit_executed") is False, "premature tokenizer claim")
    _require(evidence.get("training_executed") is False, "premature training claim")
    _require(evidence.get("optimizer_updates") == 0, "premature optimizer claim")
    _require(evidence.get("final_test_payload_accessed") is False, "final-test boundary weakened")
    _require(evidence.get("paid_compute_used") is False, "paid-compute boundary weakened")

    rows = record_inventory.get("records")
    _require(
        record_inventory.get("schema_version") == "12-6.data526-record-inventory.v1",
        "DATA-526 inventory schema drift",
    )
    _require(isinstance(rows, list) and len(rows) == DATA526_RECORDS, "DATA-526 row count drift")
    _require(
        record_inventory.get("record_count") == DATA526_RECORDS,
        "DATA-526 inventory count drift",
    )
    _require(
        record_inventory.get("total_payload_bytes") == DATA526_BYTES,
        "DATA-526 inventory bytes drift",
    )
    _require(
        record_inventory.get("record_inventory_digest_sha256") == DATA526_RECORD_INVENTORY_SHA256,
        "DATA-526 record inventory identity drift",
    )
    _require(
        record_inventory.get("payload_inventory_digest_sha256") == DATA526_PAYLOAD_INVENTORY_SHA256,
        "DATA-526 payload inventory identity drift",
    )
    normalized: list[dict[str, Any]] = []
    payload_projection: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    source_ids: set[str] = set()
    total = 0
    for row in rows:
        _require(isinstance(row, Mapping), "DATA-526 inventory row must be object")
        record_id = row.get("record_id")
        source_id = row.get("source_id")
        family = row.get("family")
        modality = row.get("modality")
        payload_sha = row.get("payload_sha256")
        payload_bytes = row.get("payload_bytes")
        _require(
            isinstance(record_id, str) and record_id and record_id not in seen_records,
            "invalid/duplicate DATA-526 record_id",
        )
        _require(
            isinstance(source_id, str) and source_id,
            f"DATA-526 source_id missing: {record_id}",
        )
        _require(isinstance(family, str) and family, f"DATA-526 family missing: {record_id}")
        _require(modality in {"uk", "en", "code"}, f"DATA-526 modality drift: {record_id}")
        _require(_is_sha256(payload_sha), f"DATA-526 payload hash malformed: {record_id}")
        _require(
            type(payload_bytes) is int and payload_bytes > 0,
            f"DATA-526 bytes malformed: {record_id}",
        )
        seen_records.add(record_id)
        source_ids.add(source_id)
        total += payload_bytes
        item = {
            "record_id": record_id,
            "source_id": source_id,
            "family": family,
            "modality": modality,
            "payload_sha256": payload_sha,
            "payload_bytes": payload_bytes,
        }
        normalized.append(item)
        payload_projection.append(
            {
                "record_id": record_id,
                "payload_sha256": payload_sha,
                "payload_bytes": payload_bytes,
            }
        )
    normalized.sort(key=lambda item: item["record_id"])
    payload_projection.sort(key=lambda item: item["record_id"])
    _require(len(source_ids) == DATA526_SOURCES, "DATA-526 source count drift")
    _require(total == DATA526_BYTES, "DATA-526 record byte arithmetic drift")
    _require(
        _sha256(_canonical(normalized)) == DATA526_RECORD_INVENTORY_SHA256,
        "DATA-526 record digest does not reproduce",
    )
    _require(
        _sha256(_canonical(payload_projection)) == DATA526_PAYLOAD_INVENTORY_SHA256,
        "DATA-526 payload digest does not reproduce",
    )


def validate_language_authority(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == RADA_LANGUAGE_SCHEMA, "Rada language schema drift")
    _verify_self_hash(report, label="Rada language", expected=RADA_LANGUAGE_REPORT_SHA256)
    source = _mapping(report.get("source"), "Rada language source")
    _require(source.get("dataset") == RADA_DATASET, "Rada language dataset drift")
    _require(source.get("dataset_revision") == RADA_REVISION, "Rada language revision drift")
    parent = _mapping(report.get("parent_provenance_rights"), "Rada language rights parent")
    _require(
        parent.get("accepted_path_inventory_sha256") == RADA_RIGHTS_INVENTORY_SHA256,
        "Rada language rights inventory drift",
    )
    result = _mapping(report.get("language_result"), "Rada language result")
    _require(
        result.get("language_pass_members") == RADA_INPUT_RECORDS,
        "Rada language pass count drift",
    )
    _require(result.get("language_reject_members") == 0, "Rada language rejects present")
    _require(result.get("language_pass_bytes") == RADA_SOURCE_BYTES, "Rada language byte drift")
    boundary = _mapping(report.get("claim_boundary"), "Rada language boundary")
    _require(
        boundary.get("training_authorized_bytes") == 0,
        "language report grants training credit",
    )
    _require(boundary.get("optimizer_updates") == 0, "language report grants optimizer credit")
    _require(
        boundary.get("model_training_executed") is False,
        "language report claims training",
    )
    _require(
        boundary.get("final_test_payload_accessed") is False,
        "language report leaks final test",
    )
    _require(boundary.get("paid_compute_used") is False, "language report uses paid compute")


def validate_rada_quality_privacy_report(
    report: Mapping[str, Any], *, expected_report_sha256: str
) -> str:
    _require(_is_sha256(expected_report_sha256), "expected Rada report SHA-256 malformed")
    _require(report.get("schema_version") == RADA_QP_SCHEMA, "Rada quality/privacy schema drift")
    report_sha = _verify_self_hash(
        report,
        label="Rada quality/privacy",
        expected=expected_report_sha256,
        newline=True,
    )
    _require(report.get("execution_profile") == "LOCAL_FREE", "Rada execution profile drift")
    _require(report.get("safe_result") == RADA_QP_SAFE_RESULT, "Rada safe-result drift")
    authority = _mapping(report.get("authority_binding"), "Rada authority binding")
    _require(
        authority.get("candidate_record_count") == RADA_INPUT_RECORDS,
        "Rada candidate count drift",
    )
    _require(
        authority.get("candidate_source_payload_bytes") == RADA_SOURCE_BYTES,
        "Rada candidate source bytes drift",
    )
    _require(authority.get("candidate_exact_keyset_enforced") is True, "Rada candidate schema open")
    _require(
        authority.get("unknown_candidate_fields_rejected") is True,
        "Rada unknown fields accepted",
    )
    _require(
        authority.get("executed_privacy_mechanics_pinned") is True,
        "Rada privacy mechanics unpinned",
    )
    _require(
        _is_git_sha(authority.get("privacy_filter_v3_git_blob_sha")),
        "Rada privacy blob missing",
    )

    _require(report.get("input_records") == RADA_INPUT_RECORDS, "Rada input count drift")
    quality = _mapping(report.get("quality"), "Rada quality accounting")
    statuses = _mapping(quality.get("statuses"), "Rada quality statuses")
    allowed_statuses = {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"}
    _require(set(statuses) <= allowed_statuses, "Rada quality status vocabulary drift")
    for key in allowed_statuses:
        count = statuses.get(key, 0)
        _require(type(count) is int and count >= 0, f"Rada quality status malformed: {key}")
    _require(
        sum(int(statuses.get(key, 0)) for key in statuses) == RADA_INPUT_RECORDS,
        "Rada quality parent accounting incomplete",
    )
    retained_units = quality.get("retained_units_before_privacy")
    retained_bytes = quality.get("retained_utf8_bytes")
    rejected_bytes = quality.get("rejected_utf8_bytes")
    _require(
        type(retained_units) is int and retained_units >= 0,
        "Rada retained-unit count malformed",
    )
    _require(type(retained_bytes) is int and retained_bytes >= 0, "Rada retained bytes malformed")
    _require(type(rejected_bytes) is int and rejected_bytes >= 0, "Rada rejected bytes malformed")
    _require(
        quality.get("partial_records_materialized") == statuses.get("RETAIN_PARTIAL", 0),
        "Rada partial parent accounting drift",
    )

    privacy = _mapping(report.get("privacy"), "Rada privacy accounting")
    output_records = report.get("output_records")
    output_text_bytes = report.get("output_text_utf8_bytes")
    held_units = privacy.get("held_units")
    held_bytes = privacy.get("held_utf8_bytes")
    _require(type(output_records) is int and output_records > 0, "Rada output count invalid")
    _require(type(output_text_bytes) is int and output_text_bytes > 0, "Rada output bytes invalid")
    _require(type(held_units) is int and held_units >= 0, "Rada held-unit count malformed")
    _require(type(held_bytes) is int and held_bytes >= 0, "Rada held bytes malformed")
    _require(privacy.get("allowed_units") == output_records, "Rada allowed-unit accounting drift")
    _require(
        privacy.get("allowed_utf8_bytes") == output_text_bytes,
        "Rada allowed-byte accounting drift",
    )
    _require(output_records + held_units == retained_units, "Rada privacy unit conservation failed")
    _require(
        output_text_bytes + held_bytes == retained_bytes,
        "Rada privacy byte conservation failed",
    )
    _require(
        privacy.get("quality_retained_bytes_reconciled") is True,
        "Rada byte reconciliation false",
    )
    _require(
        privacy.get("quality_retained_units_reconciled") is True,
        "Rada unit reconciliation false",
    )
    _require(
        privacy.get("non_allow_payload_mutation_attempted") is False,
        "Rada non-ALLOW mutation attempted",
    )
    _require(_is_sha256(report.get("output_jsonl_sha256")), "Rada output JSONL identity missing")

    boundary = _mapping(report.get("claim_boundary"), "Rada claim boundary")
    _require(
        boundary.get("upstream_handoff_authority_bound") is True,
        "Rada upstream authority unbound",
    )
    _require(boundary.get("candidate_schema_exact") is True, "Rada candidate schema not exact")
    _require(
        boundary.get("canonical_privacy_repair_bound") is True,
        "Rada privacy repair unbound",
    )
    _require(boundary.get("global_dedup_complete") is False, "Rada report preclaims global dedup")
    _require(boundary.get("training_authorized_bytes") == 0, "Rada report grants training bytes")
    _require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "Rada report grants loss positions",
    )
    _require(
        boundary.get("tokenizer_fit_authorized") is False,
        "Rada report grants tokenizer fit",
    )
    _require(boundary.get("optimizer_updates") == 0, "Rada report grants optimizer updates")
    _require(boundary.get("model_training_executed") is False, "Rada report claims training")
    _require(
        boundary.get("final_test_payload_accessed") is False,
        "Rada report leaks final test",
    )
    _require(boundary.get("paid_compute_used") is False, "Rada report uses paid compute")
    return report_sha


def validate_rada_rows(
    rows: Sequence[Mapping[str, Any]], raw_jsonl: bytes, report: Mapping[str, Any]
) -> None:
    _require(
        _sha256(raw_jsonl) == report.get("output_jsonl_sha256"),
        "Rada output JSONL hash drift",
    )
    _require(len(rows) == report.get("output_records"), "Rada output row count drift")
    seen: set[str] = set()
    text_bytes = 0
    partial_rows = 0
    for row in rows:
        record_id = row.get("record_id")
        parent_id = row.get("quality_parent_record_id")
        _require(
            isinstance(record_id, str) and record_id and record_id not in seen,
            "invalid/duplicate Rada unit id",
        )
        _require(isinstance(parent_id, str) and parent_id, f"Rada parent id missing: {record_id}")
        seen.add(record_id)
        _require(row.get("source_dataset") == RADA_DATASET, f"Rada dataset drift: {record_id}")
        _require(row.get("source_revision") == RADA_REVISION, f"Rada revision drift: {record_id}")
        _require(row.get("source_family") == RADA_FAMILY, f"Rada family drift: {record_id}")
        source_path = row.get("source_path")
        source_sha = row.get("source_payload_sha256")
        source_bytes = row.get("source_payload_bytes")
        _require(
            isinstance(source_path, str) and source_path,
            f"Rada source path missing: {record_id}",
        )
        _require(_is_sha256(source_sha), f"Rada source payload hash malformed: {record_id}")
        _require(
            type(source_bytes) is int and source_bytes > 0,
            f"Rada source payload bytes malformed: {record_id}",
        )
        _require(row.get("quality_privacy_complete") is True, f"Rada Q/P incomplete: {record_id}")
        _require(
            row.get("language_quality_privacy_complete") is False,
            f"Rada row preclaims language convergence: {record_id}",
        )
        _require(
            row.get("privacy_gate_action") == "ALLOW",
            f"Rada non-ALLOW row emitted: {record_id}",
        )
        _require(row.get("training_eligible") is False, f"Rada row preclaims training: {record_id}")
        _require(
            row.get("evaluation_eligible") is False,
            f"Rada row preclaims evaluation: {record_id}",
        )
        _require(
            row.get("global_dedup_complete") is False,
            f"Rada row preclaims dedup: {record_id}",
        )
        _require(
            row.get("reserved_evaluation_decontamination_complete") is False,
            f"Rada row preclaims decontam: {record_id}",
        )
        status = row.get("quality_gate_status")
        kind = row.get("quality_unit_kind")
        index = row.get("quality_window_index")
        _require(
            status in {"RETAIN_ALL", "RETAIN_PARTIAL"},
            f"Rada quality status invalid: {record_id}",
        )
        if status == "RETAIN_ALL":
            _require(record_id == parent_id, f"RETAIN_ALL id/parent drift: {record_id}")
            _require(
                kind == "SOURCE_NATIVE_DOCUMENT" and index is None,
                f"RETAIN_ALL unit drift: {record_id}",
            )
        else:
            partial_rows += 1
            _require(
                record_id != parent_id,
                f"RETAIN_PARTIAL raw parent substituted: {record_id}",
            )
            _require(
                kind == "BOUNDED_NATURAL_LANGUAGE_WINDOW",
                f"partial unit kind drift: {record_id}",
            )
            _require(
                type(index) is int and index >= 0,
                f"partial window index drift: {record_id}",
            )
        text = row.get("text")
        _require(isinstance(text, str) and text, f"Rada unit text missing: {record_id}")
        raw = text.encode("utf-8")
        digest = _sha256(raw)
        _require(
            row.get("quality_unit_utf8_sha256") == digest,
            f"Rada unit hash drift: {record_id}",
        )
        _require(
            row.get("decoded_text_utf8_sha256") == digest,
            f"Rada decoded hash drift: {record_id}",
        )
        _require(
            row.get("privacy_scan_input_sha256") == digest,
            f"Rada privacy input hash drift: {record_id}",
        )
        _require(
            row.get("quality_unit_utf8_bytes") == len(raw),
            f"Rada unit bytes drift: {record_id}",
        )
        _require(
            row.get("decoded_text_utf8_bytes") == len(raw),
            f"Rada decoded bytes drift: {record_id}",
        )
        _require(
            row.get("privacy_scan_input_bytes") == len(raw),
            f"Rada privacy bytes drift: {record_id}",
        )
        text_bytes += len(raw)
    _require(text_bytes == report.get("output_text_utf8_bytes"), "Rada output byte total drift")
    quality = _mapping(report.get("quality"), "Rada quality accounting")
    expected_partial_units = quality.get("partial_units_materialized_before_privacy")
    held_units = _mapping(report.get("privacy"), "Rada privacy accounting").get("held_units")
    _require(
        type(expected_partial_units) is int and expected_partial_units >= partial_rows,
        "Rada partial-unit accounting malformed",
    )
    _require(type(held_units) is int, "Rada held-unit accounting malformed")


def build_rada_matcher_inputs(
    rows: Sequence[Mapping[str, Any]], *, authority_report_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    _require(_is_sha256(authority_report_sha256), "Rada matcher authority SHA malformed")
    inventory_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    archive_url = (
        "https://huggingface.co/datasets/uacorpus/Rada_Trees/resolve/"
        f"{RADA_REVISION}/rada_xtag_texts.7z"
    )
    for row in rows:
        record_id = str(row["record_id"])
        payload = str(row["text"]).encode("utf-8")
        payload_sha = _sha256(payload)
        source_id = f"rada-trees-quality-unit:{record_id}"
        stable_unit = f"{RADA_DATASET}@{RADA_REVISION}:{record_id}"
        _require(source_id not in payloads, f"duplicate Rada matcher source id: {source_id}")
        inventory_rows.append(
            {
                "source_id": source_id,
                "source_family": RADA_FAMILY,
                "stable_origin_id": stable_unit,
                "stable_object_id": f"sha256:{payload_sha}",
                "modality": "uk",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"D03-RADA-QP:{authority_report_sha256}",
                "declared_capacity_bytes": len(payload),
                "expected_raw_bytes": len(payload),
                "expected_raw_sha256": payload_sha,
                "acquisition_url": archive_url,
                "origin_key": f"rada-quality-unit:{record_id}",
            }
        )
        payloads[source_id] = payload
    return inventory_rows, payloads


def validate_v8_survivor_authority(authority: Mapping[str, Any]) -> None:
    _require(
        authority.get("schema_version") == "12-6.next100-065f-post-dedup-survivors.v1",
        "V8 survivor schema drift",
    )
    _require(authority.get("v8_report_sha256") == V8_REPORT_SHA256, "V8 report binding drift")
    _require(
        authority.get("nested_v3_report_sha256") == V8_NESTED_V3_SHA256,
        "V8 nested binding drift",
    )
    claimed = authority.get("survivor_authority_sha256")
    _require(claimed == V8_SURVIVOR_SHA256, "V8 survivor identity drift")
    core = dict(authority)
    core.pop("survivor_authority_sha256", None)
    _require(
        _sha256(_canonical(core, ascii_only=True)) == claimed,
        "V8 survivor self-hash mismatch",
    )
    rows = authority.get("survivors")
    _require(
        isinstance(rows, list) and len(rows) == DATA526_SOURCES,
        "V8 survivor source count drift",
    )
    _require(
        authority.get("post_dedup_survivor_source_object_count") == DATA526_SOURCES,
        "V8 survivor count summary drift",
    )
    _require(
        authority.get("post_dedup_declared_capacity_bytes") == DATA526_BYTES,
        "V8 survivor byte summary drift",
    )
    boundary = _mapping(authority.get("truth_boundary"), "V8 truth boundary")
    _require(boundary.get("source_object_authority_only") is True, "V8 authority purpose drift")
    _require(
        boundary.get("authorized_training_exposure") == 0,
        "V8 authority grants training exposure",
    )
    _require(boundary.get("model_training_executed") is False, "V8 authority claims training")
    _require(boundary.get("final_test_payload_read") is False, "V8 authority leaks final test")
    _require(boundary.get("paid_compute_used") is False, "V8 authority uses paid compute")


def filter_v8_survivor_inputs(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    survivor_authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    survivor_rows = survivor_authority.get("survivors")
    _require(isinstance(survivor_rows, list) and survivor_rows, "V8 survivor rows missing")
    source_rows = inventory.get("sources")
    _require(isinstance(source_rows, list) and source_rows, "V8 reconstructed inventory missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        _require(isinstance(row, Mapping), "V8 reconstructed source row must be object")
        source_id = row.get("source_id")
        _require(
            isinstance(source_id, str) and source_id and source_id not in by_id,
            "invalid/duplicate reconstructed V8 source id",
        )
        by_id[source_id] = row
    selected_rows: list[dict[str, Any]] = []
    selected_payloads: dict[str, bytes] = {}
    total = 0
    for survivor in survivor_rows:
        _require(isinstance(survivor, Mapping), "V8 survivor row must be object")
        source_id = survivor.get("source_id")
        _require(
            isinstance(source_id, str) and source_id in by_id,
            "V8 survivor references unavailable source",
        )
        payload = payloads.get(source_id)
        _require(isinstance(payload, bytes), f"V8 survivor payload missing: {source_id}")
        expected_bytes = survivor.get("declared_capacity_bytes")
        _require(
            type(expected_bytes) is int and expected_bytes == len(payload),
            f"V8 survivor payload bytes drift: {source_id}",
        )
        _require(
            survivor.get("verified_raw_sha256") == _sha256(payload),
            f"V8 survivor raw hash drift: {source_id}",
        )
        source = by_id[source_id]
        _require(
            source.get("source_family") == survivor.get("source_family"),
            f"V8 survivor family drift: {source_id}",
        )
        _require(
            source.get("modality") == survivor.get("modality"),
            f"V8 survivor modality drift: {source_id}",
        )
        _require(
            source.get("declared_capacity_bytes") == expected_bytes,
            f"V8 survivor declared bytes drift: {source_id}",
        )
        selected_rows.append(copy.deepcopy(dict(source)))
        selected_payloads[source_id] = payload
        total += len(payload)
    _require(
        total == survivor_authority.get("post_dedup_declared_capacity_bytes"),
        "V8 selected byte total drift",
    )
    return selected_rows, selected_payloads


def _derive_survivors(dedup_report: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = dedup_report.get("sources")
    _require(isinstance(source_rows, list) and source_rows, "matcher report source rows missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        _require(isinstance(row, Mapping), "matcher source row must be object")
        source_id = row.get("source_id")
        capacity = row.get("declared_capacity_bytes")
        _require(
            isinstance(source_id, str) and source_id and source_id not in by_id,
            "matcher source id invalid/duplicate",
        )
        _require(type(capacity) is int and capacity > 0, f"matcher capacity invalid: {source_id}")
        by_id[source_id] = row
    terminal = _mapping(dedup_report.get("terminal_candidates"), "matcher terminal candidates")
    clusters = terminal.get("duplicate_clusters")
    _require(isinstance(clusters, list), "matcher duplicate clusters missing")
    dropped: set[str] = set()
    cluster_members: set[str] = set()
    normalized_clusters: list[dict[str, Any]] = []
    for raw_cluster in clusters:
        _require(
            isinstance(raw_cluster, Sequence) and not isinstance(raw_cluster, (str, bytes)),
            "invalid matcher duplicate cluster",
        )
        cluster = sorted(str(source_id) for source_id in raw_cluster)
        _require(
            len(cluster) >= 2 and len(cluster) == len(set(cluster)),
            "matcher cluster cardinality invalid",
        )
        _require(
            all(source_id in by_id for source_id in cluster),
            "matcher cluster references unknown source",
        )
        _require(not (cluster_members & set(cluster)), "matcher duplicate clusters overlap")
        cluster_members.update(cluster)
        maximum = max(int(by_id[source_id]["declared_capacity_bytes"]) for source_id in cluster)
        selected = min(
            source_id
            for source_id in cluster
            if int(by_id[source_id]["declared_capacity_bytes"]) == maximum
        )
        dropped.update(source_id for source_id in cluster if source_id != selected)
        normalized_clusters.append(
            {
                "member_source_ids": cluster,
                "selected_source_id": selected,
                "selected_declared_capacity_bytes": maximum,
            }
        )
    survivors = sorted(source_id for source_id in by_id if source_id not in dropped)
    capacity = sum(int(by_id[source_id]["declared_capacity_bytes"]) for source_id in survivors)
    _require(
        capacity == terminal.get("conservative_unique_capacity_bytes_after"),
        "derived survivor bytes do not reproduce matcher report",
    )
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "matcher_report_sha256": dedup_report.get("report_sha256"),
        "pre_dedup_source_object_count": len(by_id),
        "post_dedup_survivor_source_object_count": len(survivors),
        "pre_dedup_declared_capacity_bytes": terminal.get("declared_capacity_bytes_before"),
        "post_dedup_declared_capacity_bytes": capacity,
        "duplicate_discount_bytes": terminal.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": len(normalized_clusters),
        "duplicate_clusters": sorted(
            normalized_clusters,
            key=lambda item: tuple(item["member_source_ids"]),
        ),
        "survivor_source_ids": survivors,
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "retained_inventory_freeze_complete": False,
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
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def run_expanded_dedup(
    *,
    matcher_audit: Callable[[Mapping[str, Any], Mapping[str, bytes]], Mapping[str, Any]],
    matcher_verify: Callable[[Mapping[str, Any]], None],
    reconstructed_v8_inventory: Mapping[str, Any],
    reconstructed_v8_payloads: Mapping[str, bytes],
    v8_survivor_authority: Mapping[str, Any],
    data526_evidence: Mapping[str, Any],
    data526_record_inventory: Mapping[str, Any],
    rada_language_report: Mapping[str, Any],
    rada_quality_privacy_report: Mapping[str, Any],
    expected_rada_report_sha256: str,
    rada_rows: Sequence[Mapping[str, Any]],
    rada_raw_jsonl: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_data526_authority(data526_evidence, data526_record_inventory)
    validate_v8_survivor_authority(v8_survivor_authority)
    validate_language_authority(rada_language_report)
    rada_report_sha = validate_rada_quality_privacy_report(
        rada_quality_privacy_report,
        expected_report_sha256=expected_rada_report_sha256,
    )
    validate_rada_rows(rada_rows, rada_raw_jsonl, rada_quality_privacy_report)
    base_rows, base_payloads = filter_v8_survivor_inputs(
        reconstructed_v8_inventory,
        reconstructed_v8_payloads,
        v8_survivor_authority,
    )
    rada_inventory_rows, rada_payloads = build_rada_matcher_inputs(
        rada_rows,
        authority_report_sha256=rada_report_sha,
    )
    base_ids = set(base_payloads)
    _require(not (base_ids & set(rada_payloads)), "DATA-526/Rada source-id collision")
    combined_inventory = copy.deepcopy(dict(reconstructed_v8_inventory))
    combined_inventory["sources"] = [*base_rows, *rada_inventory_rows]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_rule"] = (
        "Expanded V9 composes exact sealed DATA-526 V8 survivors with the externally "
        "anchored complete Rada_Trees quality/privacy survivor units after terminal "
        "all-parent accounting; matching delegates to incumbent #824 V3 semantics."
    )
    combined_payloads = dict(base_payloads)
    combined_payloads.update(rada_payloads)

    dedup = matcher_audit(combined_inventory, combined_payloads)
    _require(isinstance(dedup, Mapping), "incumbent matcher returned non-object report")
    matcher_verify(dedup)
    matcher_sha = dedup.get("report_sha256")
    _require(_is_sha256(matcher_sha), "incumbent matcher report identity missing")
    terminal = _mapping(dedup.get("terminal_candidates"), "incumbent matcher terminal result")
    expected_before = DATA526_BYTES + int(rada_quality_privacy_report["output_text_utf8_bytes"])
    _require(
        terminal.get("declared_capacity_bytes_before") == expected_before,
        "expanded pre-dedup bytes drift",
    )
    _require(dedup.get("source_count") == len(combined_payloads), "expanded source count drift")

    core = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "matcher_lineage": "MERGED_PR_824_V3_REUSED_WITHOUT_SEMANTIC_CHANGES",
        "data526": {
            "evidence_identity_sha256": DATA526_EVIDENCE_SHA256,
            "v8_survivor_authority_sha256": V8_SURVIVOR_SHA256,
            "record_inventory_digest_sha256": DATA526_RECORD_INVENTORY_SHA256,
            "source_object_count": DATA526_SOURCES,
            "payload_bytes": DATA526_BYTES,
        },
        "rada_trees": {
            "language_report_sha256": RADA_LANGUAGE_REPORT_SHA256,
            "quality_privacy_report_sha256": rada_report_sha,
            "quality_privacy_output_jsonl_sha256": rada_quality_privacy_report[
                "output_jsonl_sha256"
            ],
            "survivor_unit_count": len(rada_rows),
            "survivor_payload_bytes": rada_quality_privacy_report["output_text_utf8_bytes"],
            "all_parent_records_accounted_before_global_dedup": True,
            "retained_partial_units_enter_before_global_dedup": True,
            "privacy_non_allow_units_contribute_zero_capacity": True,
            "language_authority_inherited_from_all_4384_parent_pass": True,
        },
        "source_vector": {
            "pre_dedup_source_object_count": dedup["source_count"],
            "pre_dedup_declared_capacity_bytes": terminal["declared_capacity_bytes_before"],
            "post_dedup_conservative_unique_bytes": terminal[
                "conservative_unique_capacity_bytes_after"
            ],
            "duplicate_discount_bytes": terminal["duplicate_discount_bytes"],
            "duplicate_cluster_count": terminal["duplicate_cluster_count"],
        },
        "dedup_v3": copy.deepcopy(dict(dedup)),
        "raw_text_emitted": False,
        "claim_boundary": {
            "expanded_global_dedup_complete": True,
            "retained_inventory_freeze_complete": False,
            "reserved_evaluation_decontamination_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "two_clean_builds_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }
    report = {**core, "report_sha256": _sha256(_canonical(core))}
    survivor = _derive_survivors(dedup)
    return report, survivor
