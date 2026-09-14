"""Record-level DATA-526 V8 compatibility seam for current reserved decontamination.

This module does not implement contamination matching. It binds the exact text-free
DATA-526 V8 record inventory/materialization evidence to ephemeral raw records,
translates those records into the incumbent #874 matcher-row shape, and delegates to
``execute_reserved_decontamination``. Raw payload text is never placed in durable
evidence returned by this adapter.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.current_reserved_decontamination_v1 import (
    execute_reserved_decontamination,
    verify_execution_evidence,
)

RECORD_INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
MATERIALIZATION_EVIDENCE_SCHEMA = "12-6.data526-v8-record-composition-evidence.v1"
LEGACY_HANDOFF_SCHEMA = "12-6.postdedup-decontam-handoff.v1"
EXECUTION_SCHEMA = "12-6.data526-v8-reserved-decontamination-execution.v1"


class Data526V8DecontaminationBindingError(RuntimeError):
    """Raised when the DATA-526 V8 record-level authority does not match payloads."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Data526V8DecontaminationBindingError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _require_nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _inventory_core(record_inventory: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    _require(
        record_inventory.get("schema_version") == RECORD_INVENTORY_SCHEMA,
        "DATA-526 record inventory schema drift",
    )
    raw_records = record_inventory.get("records")
    _require(
        isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)),
        "DATA-526 record inventory records must be a sequence",
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        _require(isinstance(raw, Mapping), f"inventory record[{index}] must be an object")
        row: dict[str, Any] = {}
        for key in ("record_id", "source_id", "family", "modality"):
            value = raw.get(key)
            _require(
                isinstance(value, str) and bool(value),
                f"inventory record[{index}].{key} must be non-empty text",
            )
            row[key] = value
        record_id = str(row["record_id"])
        _require(record_id not in seen, f"duplicate inventory record_id: {record_id}")
        seen.add(record_id)
        row["payload_sha256"] = _require_sha256(
            raw.get("payload_sha256"),
            f"inventory payload_sha256: {record_id}",
        )
        row["payload_bytes"] = _require_nonnegative_int(
            raw.get("payload_bytes"),
            f"inventory payload_bytes: {record_id}",
        )
        records.append(row)

    records.sort(key=lambda row: str(row["record_id"]))
    _require(
        list(raw_records) == records,
        "DATA-526 record inventory is not canonically sorted or contains extra fields",
    )
    record_digest = _sha256(_canonical_bytes(records))
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in records
    ]
    payload_digest = _sha256(_canonical_bytes(payload_projection))
    _require(
        record_inventory.get("record_inventory_digest_sha256") == record_digest,
        "DATA-526 record inventory digest mismatch",
    )
    _require(
        record_inventory.get("payload_inventory_digest_sha256") == payload_digest,
        "DATA-526 payload inventory digest mismatch",
    )
    _require(
        record_inventory.get("record_count") == len(records),
        "DATA-526 record count mismatch",
    )
    _require(
        record_inventory.get("total_payload_bytes")
        == sum(int(row["payload_bytes"]) for row in records),
        "DATA-526 payload byte total mismatch",
    )
    return records, record_digest, payload_digest


def _verify_materialization_evidence(
    evidence: Mapping[str, Any],
    *,
    record_inventory: Mapping[str, Any],
    record_digest: str,
    payload_digest: str,
    expected_materialization_evidence_identity_sha256: str,
    expected_survivor_authority_sha256: str,
) -> str:
    _require(
        evidence.get("schema_version") == MATERIALIZATION_EVIDENCE_SCHEMA,
        "DATA-526 materialization evidence schema drift",
    )
    claimed = _require_sha256(
        evidence.get("evidence_identity_sha256"),
        "evidence_identity_sha256",
    )
    _require(
        claimed
        == _require_sha256(
            expected_materialization_evidence_identity_sha256,
            "expected_materialization_evidence_identity_sha256",
        ),
        "DATA-526 materialization evidence is not independently expected",
    )
    body = deepcopy(dict(evidence))
    body.pop("evidence_identity_sha256", None)
    _require(
        _sha256(_canonical_bytes(body)) == claimed,
        "DATA-526 materialization evidence self-hash mismatch",
    )
    survivor = _require_sha256(
        evidence.get("v8_survivor_authority_sha256"),
        "v8_survivor_authority_sha256",
    )
    _require(
        survivor
        == _require_sha256(
            expected_survivor_authority_sha256,
            "expected_survivor_authority_sha256",
        ),
        "DATA-526 V8 survivor authority is not independently expected",
    )
    _require(
        evidence.get("record_inventory_digest_sha256") == record_digest,
        "materialization/record inventory digest mismatch",
    )
    _require(
        evidence.get("payload_inventory_digest_sha256") == payload_digest,
        "materialization/payload inventory digest mismatch",
    )
    for key in ("record_count", "total_payload_bytes"):
        _require(
            evidence.get(key) == record_inventory.get(key),
            f"materialization/inventory {key} mismatch",
        )
    _require(
        evidence.get("authorized_unique_optimized_targets") == 0,
        "materialization evidence grants optimized-target exposure",
    )
    _require(evidence.get("optimizer_updates") == 0, "optimizer updates already occurred")
    for key in (
        "decontamination_executed",
        "tokenizer_fit_executed",
        "training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
        "raw_payloads_emitted_to_public_evidence",
    ):
        _require(evidence.get(key) is False, f"materialization boundary weakened: {key}")
    return survivor


def bind_data526_v8_training_records(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    record_inventory: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    expected_materialization_evidence_identity_sha256: str,
    expected_survivor_authority_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Bind ephemeral DATA-526 records to the exact terminal text-free authority.

    The returned legacy-shaped handoff exists only to enter the incumbent #874
    executor. Its externally meaningful authority is the exact DATA-526 record and
    payload inventory digests checked here, not the historical field name
    ``retained_source_count``.
    """
    inventory_rows, record_digest, payload_digest = _inventory_core(record_inventory)
    _require(
        record_digest
        == _require_sha256(
            expected_record_inventory_digest_sha256,
            "expected_record_inventory_digest_sha256",
        ),
        "DATA-526 record inventory is not independently expected",
    )
    _require(
        payload_digest
        == _require_sha256(
            expected_payload_inventory_digest_sha256,
            "expected_payload_inventory_digest_sha256",
        ),
        "DATA-526 payload inventory is not independently expected",
    )
    survivor = _verify_materialization_evidence(
        materialization_evidence,
        record_inventory=record_inventory,
        record_digest=record_digest,
        payload_digest=payload_digest,
        expected_materialization_evidence_identity_sha256=(
            expected_materialization_evidence_identity_sha256
        ),
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
    )

    _require(
        isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)),
        "DATA-526 raw records must be a sequence",
    )
    by_id = {str(row["record_id"]): row for row in inventory_rows}
    matcher_rows: list[dict[str, str]] = []
    canonical_raw_records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        _require(isinstance(raw, Mapping), f"raw record[{index}] must be an object")
        _require(
            set(raw)
            == {"record_id", "source_id", "family", "modality", "normalized_payload"},
            f"raw record[{index}] schema drift",
        )
        record_id = raw.get("record_id")
        _require(
            isinstance(record_id, str) and bool(record_id),
            f"raw record[{index}].record_id must be non-empty text",
        )
        _require(record_id not in seen, f"duplicate raw record_id: {record_id}")
        seen.add(record_id)
        expected = by_id.get(record_id)
        _require(expected is not None, f"raw record not present in inventory: {record_id}")
        for raw_key, inv_key in (
            ("source_id", "source_id"),
            ("family", "family"),
            ("modality", "modality"),
        ):
            value = raw.get(raw_key)
            _require(
                isinstance(value, str) and bool(value),
                f"raw record {raw_key} missing: {record_id}",
            )
            _require(value == expected[inv_key], f"raw record {raw_key} drift: {record_id}")
        payload = raw.get("normalized_payload")
        _require(
            isinstance(payload, str) and bool(payload),
            f"raw record normalized_payload missing: {record_id}",
        )
        payload_bytes = payload.encode("utf-8")
        _require(
            len(payload_bytes) == expected["payload_bytes"],
            f"raw record payload byte drift: {record_id}",
        )
        _require(
            _sha256(payload_bytes) == expected["payload_sha256"],
            f"raw record payload SHA-256 drift: {record_id}",
        )
        canonical_raw_records.append(
            {
                "record_id": record_id,
                "source_id": str(expected["source_id"]),
                "family": str(expected["family"]),
                "modality": str(expected["modality"]),
                "normalized_payload": payload,
            }
        )
        matcher_rows.append(
            {
                "record_id": record_id,
                "source_id": str(expected["source_id"]),
                "source_family": str(expected["family"]),
                "modality": str(expected["modality"]),
                "text": payload,
            }
        )

    _require(seen == set(by_id), "raw DATA-526 record coverage differs from inventory")
    canonical_raw_records.sort(key=lambda row: row["record_id"])
    matcher_rows.sort(key=lambda row: row["record_id"])
    record_jsonl_sha256 = _sha256(
        b"".join(_canonical_bytes(row) + b"\n" for row in canonical_raw_records)
    )
    _require(
        record_jsonl_sha256 == materialization_evidence.get("record_payload_jsonl_sha256"),
        "DATA-526 record payload JSONL identity mismatch",
    )
    _require(
        len({row["source_id"] for row in canonical_raw_records})
        == materialization_evidence.get("source_object_count"),
        "DATA-526 source-object count mismatch",
    )
    projection = []
    for row in matcher_rows:
        text_bytes = row["text"].encode("utf-8")
        projection.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "source_family": row["source_family"],
                "modality": row["modality"].lower(),
                "text_sha256": _sha256(text_bytes),
                "text_utf8_bytes": len(text_bytes),
            }
        )

    handoff: dict[str, Any] = {
        "schema_version": LEGACY_HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": record_digest,
        "input_survivor_authority_sha256": survivor,
        "retained_source_count": len(matcher_rows),
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": _sha256(_canonical_bytes(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff["handoff_identity_sha256"] = _sha256(_canonical_bytes(handoff))
    return matcher_rows, handoff


def execute_data526_v8_reserved_decontamination(
    raw_training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    record_inventory: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
    reserved_payload_binding: Mapping[str, Any],
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    expected_materialization_evidence_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_reserved_binding_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
    quarantine_cross_source_families: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute incumbent #874 decontamination from exact DATA-526 V8 records."""
    matcher_rows, handoff = bind_data526_v8_training_records(
        raw_training_records,
        record_inventory=record_inventory,
        materialization_evidence=materialization_evidence,
        expected_record_inventory_digest_sha256=expected_record_inventory_digest_sha256,
        expected_payload_inventory_digest_sha256=expected_payload_inventory_digest_sha256,
        expected_materialization_evidence_identity_sha256=(
            expected_materialization_evidence_identity_sha256
        ),
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
    )
    report, inner = execute_reserved_decontamination(
        matcher_rows,
        evaluation_records,
        training_handoff_evidence=handoff,
        reserved_payload_binding=reserved_payload_binding,
        expected_inventory_identity_sha256=expected_record_inventory_digest_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        expected_training_handoff_identity_sha256=str(handoff["handoff_identity_sha256"]),
        expected_reserved_binding_identity_sha256=expected_reserved_binding_identity_sha256,
        expected_selection_validation_identity_sha256=expected_selection_validation_identity_sha256,
        expected_final_test_identity_sha256=expected_final_test_identity_sha256,
        quarantine_cross_source_families=quarantine_cross_source_families,
    )
    verify_execution_evidence(inner, report)
    core: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "status": report["status"],
        "data526_record_inventory_digest_sha256": expected_record_inventory_digest_sha256,
        "data526_payload_inventory_digest_sha256": expected_payload_inventory_digest_sha256,
        "data526_materialization_evidence_identity_sha256": (
            expected_materialization_evidence_identity_sha256
        ),
        "data526_record_payload_jsonl_sha256": materialization_evidence[
            "record_payload_jsonl_sha256"
        ],
        "data526_record_count": materialization_evidence["record_count"],
        "data526_source_object_count": materialization_evidence["source_object_count"],
        "data526_total_payload_bytes": materialization_evidence["total_payload_bytes"],
        "v8_survivor_authority_sha256": expected_survivor_authority_sha256,
        "inner_execution_identity_sha256": inner["execution_identity_sha256"],
        "decontamination_report_sha256": report["report_sha256"],
        "counts": deepcopy(report["counts"]),
        "durable_evidence_hash_only": True,
        "final_test_payload_accessed_for_decontamination": True,
        "final_test_outcomes_read": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    core["execution_identity_sha256"] = _sha256(_canonical_bytes(core))
    return report, core


def verify_data526_v8_execution_evidence(evidence: Mapping[str, Any]) -> None:
    """Verify the wrapper's durable hash-only evidence self-identity and boundaries."""
    _require(evidence.get("schema_version") == EXECUTION_SCHEMA, "execution schema drift")
    claimed = _require_sha256(
        evidence.get("execution_identity_sha256"),
        "execution_identity_sha256",
    )
    body = deepcopy(dict(evidence))
    body.pop("execution_identity_sha256", None)
    _require(_sha256(_canonical_bytes(body)) == claimed, "execution evidence hash drift")
    for key in (
        "data526_record_inventory_digest_sha256",
        "data526_payload_inventory_digest_sha256",
        "data526_materialization_evidence_identity_sha256",
        "data526_record_payload_jsonl_sha256",
        "v8_survivor_authority_sha256",
        "inner_execution_identity_sha256",
        "decontamination_report_sha256",
    ):
        _require_sha256(evidence.get(key), key)
    _require(evidence.get("durable_evidence_hash_only") is True, "durable text boundary weakened")
    _require(
        evidence.get("final_test_payload_accessed_for_decontamination") is True,
        "final-test decontamination payload access truth was erased",
    )
    _require(evidence.get("final_test_outcomes_read") is False, "final-test outcomes were read")
    _require(evidence.get("authorized_training_exposure") == 0, "training exposure fabricated")
    for key in ("tokenizer_fit_authorized", "training_executed", "paid_compute_used"):
        _require(evidence.get(key) is False, f"execution boundary weakened: {key}")
