"""Authenticated post-G05/G06 materialization -> NEXT100-106 balance bridge.

This module does not implement a second balance policy. It authenticates the durable
text-free post-materialization inventory from D03, derives the exact current family
capacity vector, adapts that vector to the already-merged NEXT100-106 gate, and binds
the gate result back to the exact physical materialization authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.trusted_family_authority_v1 import (
    TRUSTED_FAMILY_SEMANTICS,
    trusted_family_authority_root_sha256,
)

FAMILY_VECTOR_SCHEMA = "12-6.d03-postmaterialization-family-vector.v1"
CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA = "12-6.d03-current-clean-family-vector.v1"
MATERIALIZATION_SCHEMA = "12-6.d03-post-g05-g06-materialization.v1"
CURRENT_CLEAN_RECEIPT_SCHEMA = "12-6.current-clean-decontam-quality-privacy-survivor.v2"
CURRENT_CLEAN_REPEAT_SCHEMA = "12-6.d03-current-clean-composition-physical-repeat.v1"
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
NEXT100_INPUT_SCHEMA = "12-6.next100-106-post-dedup-family-vector.v1"
BALANCE_RESULT_SCHEMA = "12-6.next100-106-balance-gate-result.v1"
BALANCE_BINDING_SCHEMA = "12-6.d03-postmaterialization-balance-binding.v1"
CURRENT_CLEAN_BALANCE_BINDING_SCHEMA = "12-6.d03-current-clean-balance-binding.v1"
TARGET_STATUS = "TARGET_20M_SOURCE_MIX_FEASIBLE"
STRATUM_MAP = {"uk": "ua", "en": "en", "code": "code"}
_HEX = frozenset("0123456789abcdef")
_CURRENT_CLEAN_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "clean_training_records_sha256",
    "clean_training_handoff_sha256",
    "data232_report_sha256",
    "decontamination_execution_identity_sha256",
    "eval647_execution_receipt_identity_sha256",
    "quality_execution_identity_sha256",
    "privacy_execution_identity_sha256",
    "post_decontamination_input_rows_sha256",
    "post_quality_input_rows_sha256",
    "survivor_jsonl_sha256",
    "survivor_record_inventory_digest_sha256",
    "survivor_payload_inventory_digest_sha256",
    "input_training_records",
    "post_decontamination_records",
    "post_quality_records",
    "survivor_records",
    "survivor_payload_bytes",
    "survivor_source_objects",
    "rejection_counts",
    "privacy_detector_counts",
    "dependency_git_blobs",
    "durable_evidence_hash_only",
    "terminal_post_g05_g06_authority",
    "independent_qualification_required",
    "current_corpus_launch_authority_promoted",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
    "receipt_identity_sha256",
}
_CURRENT_CLEAN_REPEAT_KEYS = {
    "schema_version",
    "status",
    "execution_head_sha",
    "execution_profile",
    "output_files_sha256",
    "survivor_jsonl_sha256",
    "survivor_record_inventory_digest_sha256",
    "survivor_payload_inventory_digest_sha256",
    "survivor_records",
    "survivor_payload_bytes",
    "two_fresh_executions_byte_identical",
    "terminal_post_g05_g06_authority",
    "independent_qualification_required",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
    "proof_identity_sha256",
}
_CURRENT_CLEAN_REPEAT_OUTPUT_FILES = {
    "composition_receipt.json",
    "data232_report.json",
    "decontamination_execution.json",
    "eval647_execution_receipt.json",
    "privacy_execution.json",
    "quality_execution.json",
    "survivor_inventory.json",
    "survivor_records.jsonl",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_constant(value: str) -> Any:
    raise ProjectionError(f"nonstandard JSON constant is forbidden: {value}")


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProjectionError(f"nonfinite JSON number is forbidden: {value}")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ProjectionError("nonzero JSON number underflowed to zero")
    return parsed


def load_strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    """Decode one trust-boundary JSON object without lossy key/value aliases."""

    if not isinstance(raw, bytes):
        raise ProjectionError(f"{label} raw input must be bytes")
    try:
        text_value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProjectionError(f"{label} is not strict UTF-8") from exc
    try:
        value = json.loads(
            text_value,
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_strict_json_constant,
            parse_float=_strict_json_float,
        )
    except ProjectionError:
        raise
    except (json.JSONDecodeError, OverflowError, RecursionError, ValueError) as exc:
        raise ProjectionError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ProjectionError(f"{label} must contain a top-level JSON object")
    return value


def _self_hash(document: Mapping[str, Any], identity_field: str) -> str:
    clean = dict(document)
    clean.pop(identity_field, None)
    return _sha256(clean)


def _current_clean_receipt_self_hash(document: Mapping[str, Any]) -> str:
    """Match the current-clean producer's canonical JSON plus terminal LF."""
    clean = dict(document)
    clean.pop("receipt_identity_sha256", None)
    return _sha256_bytes(_canonical_bytes(clean) + b"\n")


def _require_hex(value: Any, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(ch not in _HEX for ch in value)
    ):
        raise ProjectionError(f"{field} must be {length} lowercase hex characters")
    return value


def _require_sha256(value: Any, field: str) -> str:
    return _require_hex(value, field, 64)


def _require_git_sha(value: Any, field: str) -> str:
    return _require_hex(value, field, 40)


def _require_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionError(f"{field} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectionError(f"{field} must be a non-negative integer")
    return value


def _expect(value: Any, expected: Any, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ProjectionError(f"{field} does not match independently expected value")


def _verify_zero_credit_truth(boundary: Mapping[str, Any]) -> None:
    expected = {
        "authorized_optimized_target_exposure": 0,
        "authorized_unique_loss_positions": 0,
        "current_corpus_eligible": False,
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "final_test_outcomes_read": False,
        "foreign_pretrained_weights": False,
        "learned_weights_created": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "paid_compute_used": False,
        "tokenizer_fit_authorized": False,
        "training_authorized_bytes": 0,
        "training_executed": False,
    }
    if set(boundary) != set(expected):
        raise ProjectionError("materialization zero-credit truth boundary drift")
    for field, expected_value in expected.items():
        value = boundary[field]
        if type(value) is not type(expected_value) or value != expected_value:
            raise ProjectionError("materialization zero-credit truth boundary drift")


def _verify_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_record_count: int,
    expected_total_payload_bytes: int,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    expected_source_object_count: int,
) -> list[dict[str, Any]]:
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ProjectionError("unsupported post-materialization inventory schema")
    rows = inventory.get("records")
    if not isinstance(rows, list) or not rows:
        raise ProjectionError("post-materialization inventory must contain records")

    required = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
    }
    normalized: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ProjectionError(
                f"inventory.records[{index}] fields are not closed-world"
            )
        row = {
            "record_id": _require_nonempty(raw.get("record_id"), "record_id"),
            "source_id": _require_nonempty(raw.get("source_id"), "source_id"),
            "family": _require_nonempty(raw.get("family"), "family"),
            "modality": _require_nonempty(raw.get("modality"), "modality"),
            "payload_sha256": _require_sha256(
                raw.get("payload_sha256"), "payload_sha256"
            ),
            "payload_bytes": _require_nonnegative_int(
                raw.get("payload_bytes"), "payload_bytes"
            ),
        }
        if row["payload_bytes"] <= 0:
            raise ProjectionError("retained record payload_bytes must be positive")
        if row["record_id"] in seen_record_ids:
            raise ProjectionError(f"duplicate retained record_id: {row['record_id']}")
        seen_record_ids.add(row["record_id"])
        normalized.append(row)

    normalized.sort(key=lambda row: row["record_id"])
    if normalized != rows:
        raise ProjectionError("post-materialization inventory record order drift")

    record_count = len(normalized)
    total_payload_bytes = sum(row["payload_bytes"] for row in normalized)
    source_object_count = len({row["source_id"] for row in normalized})
    record_digest = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    payload_digest = hashlib.sha256(_canonical_bytes(payload_projection)).hexdigest()

    declared_checks = {
        "record_count": record_count,
        "total_payload_bytes": total_payload_bytes,
        "record_inventory_digest_sha256": record_digest,
        "payload_inventory_digest_sha256": payload_digest,
    }
    for field, actual in declared_checks.items():
        declared = inventory.get(field)
        if type(declared) is not type(actual) or declared != actual:
            raise ProjectionError(f"post-materialization inventory {field} drift")

    independent_checks = {
        "record_count": (record_count, expected_record_count),
        "total_payload_bytes": (total_payload_bytes, expected_total_payload_bytes),
        "record_inventory_digest_sha256": (
            record_digest,
            _require_sha256(
                expected_record_inventory_digest_sha256,
                "expected_record_inventory_digest_sha256",
            ),
        ),
        "payload_inventory_digest_sha256": (
            payload_digest,
            _require_sha256(
                expected_payload_inventory_digest_sha256,
                "expected_payload_inventory_digest_sha256",
            ),
        ),
        "source_object_count": (source_object_count, expected_source_object_count),
    }
    for field, (actual, expected) in independent_checks.items():
        _expect(actual, expected, field)
    return normalized


def _validate_record_family(row: Mapping[str, Any]) -> str:
    family = str(row["family"])
    authority = TRUSTED_FAMILY_SEMANTICS.get(family)
    if authority is None:
        raise ProjectionError(
            f"retained family absent from trusted DATA526/V8 authority: {family}"
        )
    stratum = str(authority["stratum"])
    modality = str(row["modality"])
    if stratum == "code":
        if modality != "code":
            raise ProjectionError(f"code family has non-code retained modality: {family}")
    elif stratum == "en":
        if modality not in {"en", "text"}:
            raise ProjectionError(f"English family modality drift: {family}")
    elif stratum == "uk":
        if modality not in {"uk", "ua", "text"}:
            raise ProjectionError(f"Ukrainian family modality drift: {family}")
    else:
        raise ProjectionError(f"trusted family has unsupported stratum: {family}")
    return stratum


def _family_capacity_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_bytes: dict[tuple[str, str], int] = defaultdict(int)
    family_records: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        stratum = _validate_record_family(row)
        key = (stratum, str(row["family"]))
        family_bytes[key] += int(row["payload_bytes"])
        family_records[key] += 1

    families = [
        {
            "stratum": stratum,
            "family": family,
            "record_count": family_records[(stratum, family)],
            "capacity_bytes": family_bytes[(stratum, family)],
        }
        for stratum, family in sorted(family_bytes)
    ]
    family_names = {row["family"] for row in families}
    trusted_root = trusted_family_authority_root_sha256(family_names)
    stratum_capacity_bytes = {
        stratum: sum(
            row["capacity_bytes"] for row in families if row["stratum"] == stratum
        )
        for stratum in ("code", "en", "uk")
    }
    stratum_family_counts = {
        stratum: sum(1 for row in families if row["stratum"] == stratum)
        for stratum in ("code", "en", "uk")
    }
    record_membership = [
        {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "family": row["family"],
            "modality": row["modality"],
        }
        for row in rows
    ]
    return {
        "families": families,
        "stratum_capacity_bytes": stratum_capacity_bytes,
        "stratum_family_counts": stratum_family_counts,
        "record_membership_sha256": _sha256(record_membership),
        "trusted_family_authority_root_sha256": trusted_root,
    }


def build_postmaterialization_family_vector(
    *,
    inventory: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
    expected_execution_head_sha: str,
    expected_materialization_identity_sha256: str,
    expected_result_jsonl_sha256: str,
    expected_record_count: int,
    expected_total_payload_bytes: int,
    expected_source_object_count: int,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    source_git_sha: str,
) -> dict[str, Any]:
    """Authenticate #1630 machine evidence and derive exact current family capacities."""

    expected_execution_head_sha = _require_git_sha(
        expected_execution_head_sha, "expected_execution_head_sha"
    )
    source_git_sha = _require_git_sha(source_git_sha, "source_git_sha")
    expected_materialization_identity_sha256 = _require_sha256(
        expected_materialization_identity_sha256,
        "expected_materialization_identity_sha256",
    )
    expected_result_jsonl_sha256 = _require_sha256(
        expected_result_jsonl_sha256, "expected_result_jsonl_sha256"
    )
    expected_record_count = _require_nonnegative_int(
        expected_record_count, "expected_record_count"
    )
    expected_total_payload_bytes = _require_nonnegative_int(
        expected_total_payload_bytes, "expected_total_payload_bytes"
    )
    expected_source_object_count = _require_nonnegative_int(
        expected_source_object_count, "expected_source_object_count"
    )

    if materialization_evidence.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise ProjectionError("unsupported post-G05/G06 materialization evidence schema")
    if materialization_evidence.get("status") != "MATERIALIZED_ZERO_CREDIT":
        raise ProjectionError("post-G05/G06 materialization is not terminal")
    if materialization_evidence.get("execution_profile") != "LOCAL_FREE":
        raise ProjectionError("post-G05/G06 materialization is not LOCAL_FREE")
    _expect(
        materialization_evidence.get("execution_head_sha"),
        expected_execution_head_sha,
        "materialization execution_head_sha",
    )
    _expect(
        materialization_evidence.get("materialization_identity_sha256"),
        expected_materialization_identity_sha256,
        "materialization identity",
    )
    if materialization_evidence.get("repeat_materialization_byte_identical") is not True:
        raise ProjectionError("post-G05/G06 materialization repeat proof is nonterminal")
    if materialization_evidence.get("remaining_materialization_blockers") != []:
        raise ProjectionError("post-G05/G06 materialization still reports blockers")

    result = materialization_evidence.get("result")
    if not isinstance(result, Mapping):
        raise ProjectionError("materialization result must be an object")
    expected_result = {
        "record_payload_jsonl_sha256": expected_result_jsonl_sha256,
        "record_count": expected_record_count,
        "total_payload_bytes": expected_total_payload_bytes,
        "source_object_count": expected_source_object_count,
        "record_inventory_digest_sha256": _require_sha256(
            expected_record_inventory_digest_sha256,
            "expected_record_inventory_digest_sha256",
        ),
        "payload_inventory_digest_sha256": _require_sha256(
            expected_payload_inventory_digest_sha256,
            "expected_payload_inventory_digest_sha256",
        ),
    }
    for field, expected in expected_result.items():
        _expect(result.get(field), expected, f"materialization result.{field}")

    truth = materialization_evidence.get("truth_boundary")
    if not isinstance(truth, Mapping):
        raise ProjectionError("materialization truth_boundary must be an object")
    _verify_zero_credit_truth(truth)

    rows = _verify_inventory(
        inventory,
        expected_record_count=expected_record_count,
        expected_total_payload_bytes=expected_total_payload_bytes,
        expected_record_inventory_digest_sha256=(
            expected_record_inventory_digest_sha256
        ),
        expected_payload_inventory_digest_sha256=(
            expected_payload_inventory_digest_sha256
        ),
        expected_source_object_count=expected_source_object_count,
    )
    if (
        type(inventory.get("record_count")) is not type(result.get("record_count"))
        or inventory.get("record_count") != result.get("record_count")
    ):
        raise ProjectionError("inventory/result record-count cross-bind mismatch")
    if (
        type(inventory.get("total_payload_bytes"))
        is not type(result.get("total_payload_bytes"))
        or inventory.get("total_payload_bytes") != result.get("total_payload_bytes")
    ):
        raise ProjectionError("inventory/result payload-byte cross-bind mismatch")
    if (
        inventory.get("record_inventory_digest_sha256")
        != result.get("record_inventory_digest_sha256")
    ):
        raise ProjectionError("inventory/result record-root cross-bind mismatch")
    if (
        inventory.get("payload_inventory_digest_sha256")
        != result.get("payload_inventory_digest_sha256")
    ):
        raise ProjectionError("inventory/result payload-root cross-bind mismatch")

    family_projection = _family_capacity_projection(rows)
    families = family_projection["families"]
    trusted_root = family_projection["trusted_family_authority_root_sha256"]
    stratum_capacity_bytes = family_projection["stratum_capacity_bytes"]
    stratum_family_counts = family_projection["stratum_family_counts"]
    record_membership_sha256 = family_projection["record_membership_sha256"]

    vector: dict[str, Any] = {
        "schema": FAMILY_VECTOR_SCHEMA,
        "status": "PASS",
        "source_git_sha": source_git_sha,
        "materialization_identity_sha256": expected_materialization_identity_sha256,
        "materialization_execution_head_sha": expected_execution_head_sha,
        "record_payload_jsonl_sha256": expected_result_jsonl_sha256,
        "record_inventory_digest_sha256": expected_record_inventory_digest_sha256,
        "payload_inventory_digest_sha256": expected_payload_inventory_digest_sha256,
        "record_count": expected_record_count,
        "total_payload_bytes": expected_total_payload_bytes,
        "source_object_count": expected_source_object_count,
        "record_membership_sha256": record_membership_sha256,
        "trusted_family_authority_root_sha256": trusted_root,
        "families": families,
        "stratum_capacity_bytes": stratum_capacity_bytes,
        "stratum_family_counts": stratum_family_counts,
        "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP",
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "training_authorized_by_this_report": False,
    }
    vector["family_vector_identity_sha256"] = _self_hash(
        vector, "family_vector_identity_sha256"
    )
    return vector


def _verify_current_clean_zero_credit(document: Mapping[str, Any], *, label: str) -> None:
    for field in (
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        if type(document.get(field)) is not int or document.get(field) != 0:
            raise ProjectionError(f"{label} widened zero-credit field: {field}")
    for field in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
    ):
        if document.get(field) is not False:
            raise ProjectionError(f"{label} widened zero-credit field: {field}")
    if document.get("terminal_post_g05_g06_authority") is not False:
        raise ProjectionError(f"{label} fabricated terminal post-G05/G06 authority")
    if document.get("independent_qualification_required") is not True:
        raise ProjectionError(f"{label} erased independent qualification requirement")


def _verify_current_clean_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_receipt_identity_sha256: str,
    expected_survivor_jsonl_sha256: str,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    expected_record_count: int,
    expected_total_payload_bytes: int,
    expected_source_object_count: int,
) -> None:
    if set(receipt) != _CURRENT_CLEAN_RECEIPT_KEYS:
        raise ProjectionError("current-clean receipt fields are not closed-world")
    if receipt.get("schema_version") != CURRENT_CLEAN_RECEIPT_SCHEMA:
        raise ProjectionError("unsupported current-clean receipt schema")
    if (
        receipt.get("status")
        != "CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION"
    ):
        raise ProjectionError("current-clean receipt status drift")
    claimed = _require_sha256(
        receipt.get("receipt_identity_sha256"), "current-clean receipt identity"
    )
    if claimed != _require_sha256(
        expected_receipt_identity_sha256, "expected current-clean receipt identity"
    ):
        raise ProjectionError("current-clean receipt is not independently expected")
    if claimed != _current_clean_receipt_self_hash(receipt):
        raise ProjectionError("current-clean receipt self-hash mismatch")
    if receipt.get("durable_evidence_hash_only") is not True:
        raise ProjectionError("current-clean durable evidence boundary weakened")
    if receipt.get("current_corpus_launch_authority_promoted") is not False:
        raise ProjectionError("current-clean receipt fabricated corpus launch authority")
    _verify_current_clean_zero_credit(receipt, label="current-clean receipt")

    for field in (
        "clean_training_records_sha256",
        "clean_training_handoff_sha256",
        "data232_report_sha256",
        "decontamination_execution_identity_sha256",
        "eval647_execution_receipt_identity_sha256",
        "quality_execution_identity_sha256",
        "privacy_execution_identity_sha256",
        "post_decontamination_input_rows_sha256",
        "post_quality_input_rows_sha256",
    ):
        _require_sha256(receipt.get(field), field)
    dependency_blobs = receipt.get("dependency_git_blobs")
    if not isinstance(dependency_blobs, Mapping) or not dependency_blobs:
        raise ProjectionError("current-clean receipt dependency Git blobs are missing")
    for path, blob_sha in dependency_blobs.items():
        _require_nonempty(path, "dependency path")
        _require_git_sha(blob_sha, f"dependency blob: {path}")

    exact_roots = {
        "survivor_jsonl_sha256": expected_survivor_jsonl_sha256,
        "survivor_record_inventory_digest_sha256": (
            expected_record_inventory_digest_sha256
        ),
        "survivor_payload_inventory_digest_sha256": (
            expected_payload_inventory_digest_sha256
        ),
    }
    for field, expected in exact_roots.items():
        _expect(
            receipt.get(field),
            _require_sha256(expected, f"expected {field}"),
            f"current-clean receipt.{field}",
        )
    exact_counts = {
        "survivor_records": expected_record_count,
        "survivor_payload_bytes": expected_total_payload_bytes,
        "survivor_source_objects": expected_source_object_count,
    }
    for field, expected in exact_counts.items():
        expected = _require_nonnegative_int(expected, f"expected {field}")
        if expected <= 0:
            raise ProjectionError(f"current-clean receipt {field} must be positive")
        _expect(receipt.get(field), expected, f"current-clean receipt.{field}")
    for field in (
        "input_training_records",
        "post_decontamination_records",
        "post_quality_records",
    ):
        if _require_nonnegative_int(receipt.get(field), field) <= 0:
            raise ProjectionError(f"current-clean receipt {field} must be positive")
    if not (
        receipt["input_training_records"]
        >= receipt["post_decontamination_records"]
        >= receipt["post_quality_records"]
        >= receipt["survivor_records"]
    ):
        raise ProjectionError("current-clean survivor count monotonicity drift")
    if receipt["survivor_source_objects"] > receipt["survivor_records"]:
        raise ProjectionError("current-clean source-object count exceeds record count")

    for field in ("rejection_counts", "privacy_detector_counts"):
        values = receipt.get(field)
        if not isinstance(values, Mapping):
            raise ProjectionError(f"current-clean receipt {field} must be an object")
        for key, value in values.items():
            _require_nonempty(key, f"{field} key")
            _require_nonnegative_int(value, f"{field}.{key}")


def _verify_current_clean_repeat(
    proof: Mapping[str, Any],
    *,
    expected_proof_identity_sha256: str,
    expected_execution_head_sha: str,
    expected_output_files_sha256: Mapping[str, str],
    expected_survivor_jsonl_sha256: str,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    expected_record_count: int,
    expected_total_payload_bytes: int,
) -> None:
    if set(proof) != _CURRENT_CLEAN_REPEAT_KEYS:
        raise ProjectionError("current-clean repeat proof fields are not closed-world")
    if proof.get("schema_version") != CURRENT_CLEAN_REPEAT_SCHEMA:
        raise ProjectionError("unsupported current-clean repeat proof schema")
    if (
        proof.get("status")
        != "PHYSICAL_EXECUTION_COMPLETE_PENDING_INDEPENDENT_QUALIFICATION"
    ):
        raise ProjectionError("current-clean repeat proof status drift")
    if proof.get("execution_profile") != "LOCAL_FREE":
        raise ProjectionError("current-clean repeat proof is not LOCAL_FREE")
    _expect(
        proof.get("execution_head_sha"),
        _require_git_sha(
            expected_execution_head_sha, "expected current-clean execution_head_sha"
        ),
        "current-clean repeat execution_head_sha",
    )
    claimed = _require_sha256(
        proof.get("proof_identity_sha256"), "current-clean repeat proof identity"
    )
    if claimed != _require_sha256(
        expected_proof_identity_sha256, "expected current-clean repeat proof identity"
    ):
        raise ProjectionError("current-clean repeat proof is not independently expected")
    if claimed != _self_hash(proof, "proof_identity_sha256"):
        raise ProjectionError("current-clean repeat proof self-hash mismatch")
    _verify_current_clean_zero_credit(proof, label="current-clean repeat proof")
    if proof.get("two_fresh_executions_byte_identical") is not True:
        raise ProjectionError("current-clean repeat proof is not byte-identical")

    outputs = proof.get("output_files_sha256")
    if not isinstance(outputs, Mapping) or set(outputs) != _CURRENT_CLEAN_REPEAT_OUTPUT_FILES:
        raise ProjectionError("current-clean repeat output file set drift")
    for name, value in outputs.items():
        _require_sha256(value, f"repeat output_files_sha256.{name}")
    for name, expected in expected_output_files_sha256.items():
        _expect(
            outputs.get(name),
            _require_sha256(expected, f"expected raw SHA-256 for {name}"),
            f"current-clean repeat raw root: {name}",
        )

    exact_roots = {
        "survivor_jsonl_sha256": expected_survivor_jsonl_sha256,
        "survivor_record_inventory_digest_sha256": (
            expected_record_inventory_digest_sha256
        ),
        "survivor_payload_inventory_digest_sha256": (
            expected_payload_inventory_digest_sha256
        ),
    }
    for field, expected in exact_roots.items():
        _expect(
            proof.get(field),
            _require_sha256(expected, f"expected repeat {field}"),
            f"current-clean repeat.{field}",
        )
    _expect(
        proof.get("survivor_records"),
        _require_nonnegative_int(expected_record_count, "expected survivor_records"),
        "current-clean repeat.survivor_records",
    )
    _expect(
        proof.get("survivor_payload_bytes"),
        _require_nonnegative_int(
            expected_total_payload_bytes, "expected survivor_payload_bytes"
        ),
        "current-clean repeat.survivor_payload_bytes",
    )


def _rebuild_current_clean_survivor_inventory(
    survivor_records_raw: bytes,
) -> dict[str, Any]:
    """Rebuild the DATA526 inventory from authenticated current-clean JSONL bytes."""

    if not isinstance(survivor_records_raw, bytes) or not survivor_records_raw:
        raise ProjectionError("current-clean survivor JSONL bytes are missing")
    if not survivor_records_raw.endswith(b"\n"):
        raise ProjectionError("current-clean survivor JSONL must end with LF")

    expected_keys = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "normalized_payload",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, physical_line in enumerate(
        survivor_records_raw.splitlines(keepends=True)
    ):
        if not physical_line.endswith(b"\n"):
            raise ProjectionError(
                f"current-clean survivor record[{index}] lacks terminal LF"
            )
        line = physical_line[:-1]
        record = load_strict_json_object(
            line,
            label=f"current-clean survivor record[{index}]",
        )
        if set(record) != expected_keys:
            raise ProjectionError(
                f"current-clean survivor record[{index}] schema drift"
            )
        if _canonical_bytes(record) != line:
            raise ProjectionError(
                f"current-clean survivor record[{index}] is not canonical JSON"
            )
        for key in expected_keys:
            if not isinstance(record.get(key), str) or not record[key]:
                raise ProjectionError(
                    f"current-clean survivor record[{index}].{key} malformed"
                )
        record_id = record["record_id"]
        if record_id in seen:
            raise ProjectionError(
                f"duplicate current-clean survivor record_id: {record_id}"
            )
        seen.add(record_id)
        payload_raw = record["normalized_payload"].encode("utf-8")
        rows.append(
            {
                "record_id": record_id,
                "source_id": record["source_id"],
                "family": record["family"],
                "modality": record["modality"],
                "payload_sha256": _sha256_bytes(payload_raw),
                "payload_bytes": len(payload_raw),
            }
        )
    if not rows:
        raise ProjectionError("current-clean survivor JSONL contains no records")

    rows.sort(key=lambda row: row["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    return {
        "schema_version": INVENTORY_SCHEMA,
        "record_count": len(rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "record_inventory_digest_sha256": _sha256(rows),
        "payload_inventory_digest_sha256": _sha256(payload_projection),
        "records": rows,
    }


def build_current_clean_family_vector(
    *,
    composition_receipt_raw: bytes,
    survivor_inventory_raw: bytes,
    repeat_proof_raw: bytes,
    survivor_records_raw: bytes,
    expected_composition_receipt_json_sha256: str,
    expected_survivor_inventory_json_sha256: str,
    expected_repeat_proof_json_sha256: str,
    expected_survivor_records_jsonl_sha256: str,
    expected_receipt_identity_sha256: str,
    expected_repeat_proof_identity_sha256: str,
    expected_execution_head_sha: str,
    expected_record_count: int,
    expected_total_payload_bytes: int,
    expected_source_object_count: int,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    source_git_sha: str,
) -> dict[str, Any]:
    """Authenticate current #2023 physical bytes and project them into #838."""

    raw_roots = {
        "composition_receipt.json": (
            composition_receipt_raw,
            expected_composition_receipt_json_sha256,
        ),
        "survivor_inventory.json": (
            survivor_inventory_raw,
            expected_survivor_inventory_json_sha256,
        ),
        "repeat_proof.json": (repeat_proof_raw, expected_repeat_proof_json_sha256),
        "survivor_records.jsonl": (
            survivor_records_raw,
            expected_survivor_records_jsonl_sha256,
        ),
    }
    for name, (raw, expected) in raw_roots.items():
        if not isinstance(raw, bytes):
            raise ProjectionError(f"{name} raw input must be bytes")
        _expect(
            _sha256_bytes(raw),
            _require_sha256(expected, f"expected raw SHA-256 for {name}"),
            f"raw SHA-256 for {name}",
        )

    receipt = load_strict_json_object(
        composition_receipt_raw, label="current-clean composition receipt"
    )
    inventory = load_strict_json_object(
        survivor_inventory_raw, label="current-clean survivor inventory"
    )
    repeat = load_strict_json_object(
        repeat_proof_raw, label="current-clean repeat proof"
    )

    expected_record_count = _require_nonnegative_int(
        expected_record_count, "expected_record_count"
    )
    expected_total_payload_bytes = _require_nonnegative_int(
        expected_total_payload_bytes, "expected_total_payload_bytes"
    )
    expected_source_object_count = _require_nonnegative_int(
        expected_source_object_count, "expected_source_object_count"
    )
    if min(
        expected_record_count,
        expected_total_payload_bytes,
        expected_source_object_count,
    ) <= 0:
        raise ProjectionError("current-clean survivor counts/bytes must be positive")
    source_git_sha = _require_git_sha(source_git_sha, "source_git_sha")
    expected_execution_head_sha = _require_git_sha(
        expected_execution_head_sha, "expected_execution_head_sha"
    )
    expected_record_inventory_digest_sha256 = _require_sha256(
        expected_record_inventory_digest_sha256,
        "expected_record_inventory_digest_sha256",
    )
    expected_payload_inventory_digest_sha256 = _require_sha256(
        expected_payload_inventory_digest_sha256,
        "expected_payload_inventory_digest_sha256",
    )
    expected_survivor_records_jsonl_sha256 = _require_sha256(
        expected_survivor_records_jsonl_sha256,
        "expected_survivor_records_jsonl_sha256",
    )

    _verify_current_clean_receipt(
        receipt,
        expected_receipt_identity_sha256=expected_receipt_identity_sha256,
        expected_survivor_jsonl_sha256=expected_survivor_records_jsonl_sha256,
        expected_record_inventory_digest_sha256=(
            expected_record_inventory_digest_sha256
        ),
        expected_payload_inventory_digest_sha256=(
            expected_payload_inventory_digest_sha256
        ),
        expected_record_count=expected_record_count,
        expected_total_payload_bytes=expected_total_payload_bytes,
        expected_source_object_count=expected_source_object_count,
    )
    _verify_current_clean_repeat(
        repeat,
        expected_proof_identity_sha256=expected_repeat_proof_identity_sha256,
        expected_execution_head_sha=expected_execution_head_sha,
        expected_output_files_sha256={
            "composition_receipt.json": expected_composition_receipt_json_sha256,
            "survivor_inventory.json": expected_survivor_inventory_json_sha256,
            "survivor_records.jsonl": expected_survivor_records_jsonl_sha256,
        },
        expected_survivor_jsonl_sha256=expected_survivor_records_jsonl_sha256,
        expected_record_inventory_digest_sha256=(
            expected_record_inventory_digest_sha256
        ),
        expected_payload_inventory_digest_sha256=(
            expected_payload_inventory_digest_sha256
        ),
        expected_record_count=expected_record_count,
        expected_total_payload_bytes=expected_total_payload_bytes,
    )
    rebuilt_inventory = _rebuild_current_clean_survivor_inventory(
        survivor_records_raw
    )
    if inventory != rebuilt_inventory:
        raise ProjectionError(
            "current-clean survivor inventory differs from rebuilt survivor JSONL"
        )
    rows = _verify_inventory(
        rebuilt_inventory,
        expected_record_count=expected_record_count,
        expected_total_payload_bytes=expected_total_payload_bytes,
        expected_record_inventory_digest_sha256=(
            expected_record_inventory_digest_sha256
        ),
        expected_payload_inventory_digest_sha256=(
            expected_payload_inventory_digest_sha256
        ),
        expected_source_object_count=expected_source_object_count,
    )
    family_projection = _family_capacity_projection(rows)
    vector: dict[str, Any] = {
        "schema": CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA,
        "status": "PASS",
        "source_git_sha": source_git_sha,
        "current_clean_execution_head_sha": expected_execution_head_sha,
        "current_clean_receipt_identity_sha256": _require_sha256(
            expected_receipt_identity_sha256, "expected current-clean receipt identity"
        ),
        "current_clean_repeat_proof_identity_sha256": _require_sha256(
            expected_repeat_proof_identity_sha256,
            "expected current-clean repeat proof identity",
        ),
        "composition_receipt_json_sha256": _require_sha256(
            expected_composition_receipt_json_sha256,
            "expected composition receipt raw SHA-256",
        ),
        "survivor_inventory_json_sha256": _require_sha256(
            expected_survivor_inventory_json_sha256,
            "expected survivor inventory raw SHA-256",
        ),
        "repeat_proof_json_sha256": _require_sha256(
            expected_repeat_proof_json_sha256,
            "expected repeat proof raw SHA-256",
        ),
        "survivor_records_jsonl_sha256": expected_survivor_records_jsonl_sha256,
        "record_inventory_digest_sha256": expected_record_inventory_digest_sha256,
        "payload_inventory_digest_sha256": expected_payload_inventory_digest_sha256,
        "record_count": expected_record_count,
        "total_payload_bytes": expected_total_payload_bytes,
        "source_object_count": expected_source_object_count,
        "record_membership_sha256": family_projection["record_membership_sha256"],
        "trusted_family_authority_root_sha256": family_projection[
            "trusted_family_authority_root_sha256"
        ],
        "families": family_projection["families"],
        "stratum_capacity_bytes": family_projection["stratum_capacity_bytes"],
        "stratum_family_counts": family_projection["stratum_family_counts"],
        "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP",
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "training_authorized_by_this_report": False,
    }
    vector["family_vector_identity_sha256"] = _self_hash(
        vector, "family_vector_identity_sha256"
    )
    return vector


def verify_postmaterialization_family_vector(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str | None = None,
) -> str:
    schema = document.get("schema")
    if schema not in {FAMILY_VECTOR_SCHEMA, CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA}:
        raise ProjectionError("unsupported post-materialization family vector schema")
    claimed = _require_sha256(
        document.get("family_vector_identity_sha256"),
        "family_vector_identity_sha256",
    )
    if claimed != _self_hash(document, "family_vector_identity_sha256"):
        raise ProjectionError("post-materialization family vector self-hash mismatch")
    if expected_identity_sha256 is not None and claimed != _require_sha256(
        expected_identity_sha256, "expected_family_vector_identity_sha256"
    ):
        raise ProjectionError(
            "post-materialization family vector does not match external expectation"
        )
    if document.get("status") != "PASS":
        raise ProjectionError("post-materialization family vector is not PASS")
    if document.get("next_gate") != "NEXT100-106_BALANCE_FAMILY_CAP":
        raise ProjectionError("post-materialization family vector next-gate drift")

    if schema == FAMILY_VECTOR_SCHEMA:
        for field in (
            "materialization_identity_sha256",
            "record_payload_jsonl_sha256",
            "record_inventory_digest_sha256",
            "payload_inventory_digest_sha256",
            "record_membership_sha256",
            "trusted_family_authority_root_sha256",
        ):
            _require_sha256(document.get(field), field)
        _require_git_sha(
            document.get("materialization_execution_head_sha"),
            "materialization_execution_head_sha",
        )
    else:
        for field in (
            "current_clean_receipt_identity_sha256",
            "current_clean_repeat_proof_identity_sha256",
            "composition_receipt_json_sha256",
            "survivor_inventory_json_sha256",
            "repeat_proof_json_sha256",
            "survivor_records_jsonl_sha256",
            "record_inventory_digest_sha256",
            "payload_inventory_digest_sha256",
            "record_membership_sha256",
            "trusted_family_authority_root_sha256",
        ):
            _require_sha256(document.get(field), field)
        _require_git_sha(
            document.get("current_clean_execution_head_sha"),
            "current_clean_execution_head_sha",
        )
    _require_git_sha(document.get("source_git_sha"), "source_git_sha")

    for field in (
        "training_eligible",
        "evaluation_eligible",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "training_authorized_by_this_report",
    ):
        if document.get(field) is not False:
            raise ProjectionError(f"{field} must remain false")
    if (
        type(document.get("authorized_optimized_target_exposure")) is not int
        or document.get("authorized_optimized_target_exposure") != 0
    ):
        raise ProjectionError("family vector must keep optimized-target exposure at zero")

    families = document.get("families")
    if not isinstance(families, list) or not families:
        raise ProjectionError("post-materialization family vector requires families")
    seen: set[str] = set()
    total = 0
    by_stratum = defaultdict(int)
    counts = defaultdict(int)
    for row in families:
        if not isinstance(row, Mapping) or set(row) != {
            "stratum",
            "family",
            "record_count",
            "capacity_bytes",
        }:
            raise ProjectionError(
                "post-materialization family row fields are not closed-world"
            )
        family = _require_nonempty(row.get("family"), "family")
        if family in seen:
            raise ProjectionError(f"duplicate family row: {family}")
        seen.add(family)
        authority = TRUSTED_FAMILY_SEMANTICS.get(family)
        if authority is None or row.get("stratum") != authority["stratum"]:
            raise ProjectionError(
                f"family stratum differs from trusted authority: {family}"
            )
        record_count = _require_nonnegative_int(row.get("record_count"), "record_count")
        capacity = _require_nonnegative_int(row.get("capacity_bytes"), "capacity_bytes")
        if record_count <= 0 or capacity <= 0:
            raise ProjectionError("surviving family rows must be positive")
        total += capacity
        by_stratum[str(row["stratum"])] += capacity
        counts[str(row["stratum"])] += 1

    if total != document.get("total_payload_bytes"):
        raise ProjectionError("family vector payload-byte arithmetic mismatch")
    if sum(row["record_count"] for row in families) != document.get("record_count"):
        raise ProjectionError("family vector record-count arithmetic mismatch")
    if document.get("source_object_count") > document.get("record_count"):
        raise ProjectionError("source-object count exceeds record count")
    expected_strata = {
        stratum: by_stratum[stratum] for stratum in ("code", "en", "uk")
    }
    if document.get("stratum_capacity_bytes") != expected_strata:
        raise ProjectionError("family vector stratum capacity arithmetic mismatch")
    expected_counts = {
        stratum: counts[stratum] for stratum in ("code", "en", "uk")
    }
    if document.get("stratum_family_counts") != expected_counts:
        raise ProjectionError("family vector stratum family-count arithmetic mismatch")
    expected_trusted = trusted_family_authority_root_sha256(seen)
    if document.get("trusted_family_authority_root_sha256") != expected_trusted:
        raise ProjectionError("trusted family authority root mismatch")
    return claimed


def _normalize_dedup_authority(
    document: Mapping[str, Any],
    *,
    expected_worker_id: str,
    expected_head_sha: str,
    expected_evidence_identity_sha256: str,
) -> dict[str, str]:
    required = {
        "worker_id",
        "head_sha",
        "evidence_identity_sha256",
        "terminal_verdict",
    }
    if set(document) != required:
        raise ProjectionError("dedup authority fields are not closed-world")
    normalized = {
        "worker_id": _require_nonempty(document.get("worker_id"), "dedup worker_id"),
        "head_sha": _require_git_sha(document.get("head_sha"), "dedup head_sha"),
        "evidence_identity_sha256": _require_sha256(
            document.get("evidence_identity_sha256"),
            "dedup evidence_identity_sha256",
        ),
        "terminal_verdict": document.get("terminal_verdict"),
    }
    if normalized["terminal_verdict"] != "PASS":
        raise ProjectionError("dedup authority is not terminal PASS")
    _expect(normalized["worker_id"], expected_worker_id, "dedup worker_id")
    _expect(
        normalized["head_sha"],
        _require_git_sha(expected_head_sha, "expected_dedup_head_sha"),
        "dedup head_sha",
    )
    _expect(
        normalized["evidence_identity_sha256"],
        _require_sha256(
            expected_evidence_identity_sha256,
            "expected_dedup_evidence_identity_sha256",
        ),
        "dedup evidence identity",
    )
    return normalized


def _family_vector_physical_authority(
    family_vector: Mapping[str, Any],
    identity: str,
) -> dict[str, Any]:
    if family_vector.get("schema") == FAMILY_VECTOR_SCHEMA:
        return {
            "family_vector_identity_sha256": identity,
            "materialization_identity_sha256": family_vector[
                "materialization_identity_sha256"
            ],
            "record_payload_jsonl_sha256": family_vector[
                "record_payload_jsonl_sha256"
            ],
            "record_inventory_digest_sha256": family_vector[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": family_vector[
                "payload_inventory_digest_sha256"
            ],
            "record_count": family_vector["record_count"],
            "total_payload_bytes": family_vector["total_payload_bytes"],
            "source_object_count": family_vector["source_object_count"],
        }
    if family_vector.get("schema") == CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA:
        return {
            "family_vector_identity_sha256": identity,
            "current_clean_receipt_identity_sha256": family_vector[
                "current_clean_receipt_identity_sha256"
            ],
            "current_clean_repeat_proof_identity_sha256": family_vector[
                "current_clean_repeat_proof_identity_sha256"
            ],
            "composition_receipt_json_sha256": family_vector[
                "composition_receipt_json_sha256"
            ],
            "survivor_inventory_json_sha256": family_vector[
                "survivor_inventory_json_sha256"
            ],
            "repeat_proof_json_sha256": family_vector["repeat_proof_json_sha256"],
            "survivor_records_jsonl_sha256": family_vector[
                "survivor_records_jsonl_sha256"
            ],
            "record_inventory_digest_sha256": family_vector[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": family_vector[
                "payload_inventory_digest_sha256"
            ],
            "record_count": family_vector["record_count"],
            "total_payload_bytes": family_vector["total_payload_bytes"],
            "source_object_count": family_vector["source_object_count"],
        }
    raise ProjectionError("unsupported family-vector physical authority schema")


def adapt_postmaterialization_family_vector_to_next100_106(
    family_vector: Mapping[str, Any],
    *,
    expected_family_vector_identity_sha256: str,
    dedup_authority: Mapping[str, Any],
    expected_dedup_worker_id: str,
    expected_dedup_head_sha: str,
    expected_dedup_evidence_identity_sha256: str,
) -> dict[str, Any]:
    """Serialize current physical capacities for the canonical NEXT100-106 gate."""

    identity = verify_postmaterialization_family_vector(
        family_vector,
        expected_identity_sha256=expected_family_vector_identity_sha256,
    )
    normalized_authority = _normalize_dedup_authority(
        dedup_authority,
        expected_worker_id=expected_dedup_worker_id,
        expected_head_sha=expected_dedup_head_sha,
        expected_evidence_identity_sha256=expected_dedup_evidence_identity_sha256,
    )

    rows: list[dict[str, Any]] = []
    by_stratum = defaultdict(int)
    family_count = defaultdict(int)
    for row in family_vector["families"]:
        stratum = STRATUM_MAP[str(row["stratum"])]
        capacity = int(row["capacity_bytes"])
        rows.append(
            {
                "family_id": row["family"],
                "stratum": stratum,
                "unique_bytes": capacity,
            }
        )
        by_stratum[stratum] += capacity
        family_count[stratum] += 1
    rows.sort(key=lambda row: row["family_id"])
    strata = ("ua", "en", "code")
    result = {
        "schema_version": NEXT100_INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": normalized_authority,
        "families": rows,
        "totals": {
            "total_unique_bytes": sum(by_stratum.values()),
            "by_stratum": {stratum: by_stratum[stratum] for stratum in strata},
            "family_count": {stratum: family_count[stratum] for stratum in strata},
        },
        "physical_authority": _family_vector_physical_authority(
            family_vector, identity
        ),
    }
    return result


def verify_balance_result(
    result: Mapping[str, Any],
    *,
    expected_result_identity_sha256: str | None = None,
) -> str:
    if result.get("schema_version") != BALANCE_RESULT_SCHEMA:
        raise ProjectionError("unsupported NEXT100-106 balance result schema")
    claimed = _require_sha256(
        result.get("result_identity_sha256"), "result_identity_sha256"
    )
    if claimed != _self_hash(result, "result_identity_sha256"):
        raise ProjectionError("NEXT100-106 balance result self-hash mismatch")
    if expected_result_identity_sha256 is not None and claimed != _require_sha256(
        expected_result_identity_sha256, "expected_balance_result_identity_sha256"
    ):
        raise ProjectionError("NEXT100-106 result does not match external expectation")
    boundary = result.get("claim_boundary")
    expected_boundary = {
        "authorized_training_exposure_loss_positions": 0,
        "corpus_identity": None,
        "shard_identity": None,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "paid_compute_authorized": False,
        "source_bytes_are_loss_positions": False,
    }
    if not isinstance(boundary, Mapping) or set(boundary) != set(expected_boundary):
        raise ProjectionError("NEXT100-106 result claim boundary drift")
    for field, expected in expected_boundary.items():
        if type(boundary[field]) is not type(expected) or boundary[field] != expected:
            raise ProjectionError("NEXT100-106 result claim boundary drift")
    return claimed


_TARGET_20M_STRATUM_BYTES = {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000}
_TARGET_20M_FAMILY_CAP_BYTES = {"ua": 5_000_000, "en": 4_200_000, "code": 2_400_000}
_TARGET_20M_POLICY_IDENTITY_SHA256 = (
    "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
)


def _require_exact_target_totals(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ProjectionError(f"{label} target balance totals schema drift")
    if type(value["total_unique_bytes"]) is not int or value["total_unique_bytes"] != (
        expected["total_unique_bytes"]
    ):
        raise ProjectionError(f"{label} target balance total source-byte drift")
    for field in ("by_stratum", "family_count"):
        observed = value[field]
        required = expected[field]
        if not isinstance(observed, Mapping) or set(observed) != set(required):
            raise ProjectionError(f"{label} target balance {field} schema drift")
        for stratum, amount in required.items():
            if type(observed[stratum]) is not int or observed[stratum] != amount:
                raise ProjectionError(f"{label} target balance {field} type/value drift")


def _verify_target_balance_evidence(
    result: Mapping[str, Any],
    next100_input: Mapping[str, Any],
    family_vector: Mapping[str, Any],
) -> None:
    """Independently reject a self-resealed but unphysical 20M selection claim.

    This is a read-only check of the fixed, independently identified NEXT100-106
    policy, not an alternative source allocator or a source-capacity promotion.
    """
    if result.get("status") != TARGET_STATUS:
        return
    for field in ("target_total_source_bytes", "maximum_feasible_total_source_bytes"):
        if type(result.get(field)) is not int or result[field] != 20_000_000:
            raise ProjectionError(f"target balance {field} is not the fixed 20M budget")
    for field in ("target_stratum_bytes", "maximum_feasible_stratum_bytes"):
        observed = result.get(field)
        if not isinstance(observed, Mapping) or set(observed) != set(_TARGET_20M_STRATUM_BYTES):
            raise ProjectionError(f"target balance {field} schema drift")
        for stratum, expected in _TARGET_20M_STRATUM_BYTES.items():
            if type(observed[stratum]) is not int or observed[stratum] != expected:
                raise ProjectionError(f"target balance {field} violates 45/35/20")

    expected_families = sorted(
        (
            {
                "family_id": row["family"],
                "stratum": STRATUM_MAP[row["stratum"]],
                "unique_bytes": row["capacity_bytes"],
            }
            for row in family_vector["families"]
        ),
        key=lambda row: row["family_id"],
    )
    supplied_families = next100_input.get("families")
    if not isinstance(supplied_families, list) or len(supplied_families) != len(
        expected_families
    ):
        raise ProjectionError("target balance input families differ from physical authority")
    for supplied, expected in zip(supplied_families, expected_families):
        if not isinstance(supplied, Mapping) or set(supplied) != set(expected):
            raise ProjectionError("target balance input family fields differ from authority")
        for field, value in expected.items():
            if type(supplied[field]) is not type(value) or supplied[field] != value:
                raise ProjectionError("target balance input family differs from authority")
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in _TARGET_20M_STRATUM_BYTES}
    seen: set[str] = set()
    for family in expected_families:
        name = family["family_id"]
        stratum = family["stratum"]
        capacity = family["unique_bytes"]
        if (
            type(name) is not str or not name or name in seen
            or stratum not in grouped
            or type(capacity) is not int or capacity < 0
        ):
            raise ProjectionError("target balance family identity/capacity invalid")
        seen.add(name)
        grouped[stratum].append(family)
    counts = {key: len(rows) for key, rows in grouped.items()}
    physical_capacity = {
        key: sum(row["unique_bytes"] for row in rows)
        for key, rows in grouped.items()
    }
    if any(count < 2 for count in counts.values()):
        raise ProjectionError("target balance lacks two independent families per stratum")
    expected_totals = {
        "total_unique_bytes": sum(physical_capacity.values()),
        "by_stratum": physical_capacity,
        "family_count": counts,
    }
    _require_exact_target_totals(
        next100_input.get("totals"), expected_totals, "input"
    )
    _require_exact_target_totals(
        result.get("input_totals"), expected_totals, "result"
    )
    observed_minimum = result.get("family_minimum")
    if (
        not isinstance(observed_minimum, Mapping)
        or set(observed_minimum) != {"required_per_stratum", "observed", "pass"}
        or type(observed_minimum["required_per_stratum"]) is not int
        or observed_minimum["required_per_stratum"] != 2
        or not isinstance(observed_minimum["observed"], Mapping)
        or set(observed_minimum["observed"]) != set(counts)
        or any(
            type(observed_minimum["observed"][key]) is not int
            or observed_minimum["observed"][key] != count
            for key, count in counts.items()
        )
        or observed_minimum["pass"] is not True
    ):
        raise ProjectionError("target balance independent-family minimum drift")
    for field, expected in (
        ("raw_capacity_by_stratum", physical_capacity),
        (
            "raw_gap_to_target_by_stratum",
            {
                key: max(0, _TARGET_20M_STRATUM_BYTES[key] - physical_capacity[key])
                for key in _TARGET_20M_STRATUM_BYTES
            },
        ),
    ):
        observed = result.get(field)
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ProjectionError(f"target balance {field} schema drift")
        for stratum, amount in expected.items():
            if type(observed[stratum]) is not int or observed[stratum] != amount:
                raise ProjectionError(f"target balance {field} physical-capacity drift")

    expected_allocations: list[dict[str, Any]] = []
    for stratum, required in _TARGET_20M_STRATUM_BYTES.items():
        cap = _TARGET_20M_FAMILY_CAP_BYTES[stratum]
        remaining = required
        ordered = sorted(
            grouped[stratum],
            key=lambda row: (-min(row["unique_bytes"], cap), row["family_id"]),
        )
        for family in ordered:
            take = min(family["unique_bytes"], cap, remaining)
            if take:
                expected_allocations.append(
                    {
                        "family_id": family["family_id"],
                        "stratum": stratum,
                        "allocated_bytes": take,
                        "available_unique_bytes": family["unique_bytes"],
                        "effective_family_cap_bytes": cap,
                    }
                )
                remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise ProjectionError("target balance physical capacity cannot meet policy")
    expected_allocations.sort(key=lambda row: row["family_id"])
    actual_allocations = result.get("deterministic_maximum_allocation")
    if not isinstance(actual_allocations, list) or len(actual_allocations) != len(
        expected_allocations
    ):
        raise ProjectionError("target balance allocation size drift")
    for actual, expected in zip(actual_allocations, expected_allocations):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ProjectionError("target balance allocation fields drift")
        for field, value in expected.items():
            if type(actual[field]) is not type(value) or actual[field] != value:
                raise ProjectionError("target balance allocation is not canonical or policy compliant")


def build_balance_result_binding(
    *,
    family_vector: Mapping[str, Any],
    expected_family_vector_identity_sha256: str,
    next100_input: Mapping[str, Any],
    balance_result: Mapping[str, Any],
    expected_policy_identity_sha256: str,
    expected_result_identity_sha256: str,
) -> dict[str, Any]:
    """Bind a canonical #838 result to the exact current materialization universe."""

    family_identity = verify_postmaterialization_family_vector(
        family_vector,
        expected_identity_sha256=expected_family_vector_identity_sha256,
    )
    result_identity = verify_balance_result(
        balance_result,
        expected_result_identity_sha256=expected_result_identity_sha256,
    )
    policy_identity = _require_sha256(
        expected_policy_identity_sha256, "expected_policy_identity_sha256"
    )
    if balance_result.get("policy_identity_sha256") != policy_identity:
        raise ProjectionError("NEXT100-106 policy identity mismatch")
    if (
        balance_result.get("status") == TARGET_STATUS
        and policy_identity != _TARGET_20M_POLICY_IDENTITY_SHA256
    ):
        raise ProjectionError("target balance policy identity is not the reviewed policy")
    if balance_result.get("dedup_authority") != next100_input.get("dedup_authority"):
        raise ProjectionError("NEXT100-106 result dedup authority differs from input")
    if balance_result.get("input_totals") != next100_input.get("totals"):
        raise ProjectionError("NEXT100-106 result totals differ from physical input")
    _verify_target_balance_evidence(balance_result, next100_input, family_vector)

    physical = next100_input.get("physical_authority")
    if not isinstance(physical, Mapping):
        raise ProjectionError("NEXT100-106 input lacks physical authority extension")
    expected_physical = _family_vector_physical_authority(
        family_vector, family_identity
    )
    if dict(physical) != expected_physical:
        raise ProjectionError("NEXT100-106 input physical authority mismatch")

    binding_common: dict[str, Any] = {
        "status": "BOUND_ZERO_CREDIT",
        "family_vector_identity_sha256": family_identity,
        "next100_input_identity_sha256": _sha256(dict(next100_input)),
        "balance_policy_identity_sha256": policy_identity,
        "balance_result_identity_sha256": result_identity,
        "balance_status": balance_result.get("status"),
        "maximum_feasible_total_source_bytes": balance_result.get(
            "maximum_feasible_total_source_bytes"
        ),
        "maximum_feasible_stratum_bytes": balance_result.get(
            "maximum_feasible_stratum_bytes"
        ),
        "deterministic_maximum_allocation": balance_result.get(
            "deterministic_maximum_allocation"
        ),
        "target_total_source_bytes": balance_result.get("target_total_source_bytes"),
        "target_stratum_bytes": balance_result.get("target_stratum_bytes"),
        "raw_capacity_by_stratum": balance_result.get("raw_capacity_by_stratum"),
        "raw_gap_to_target_by_stratum": balance_result.get(
            "raw_gap_to_target_by_stratum"
        ),
        "balanced_selection_authorized": (
            balance_result.get("status") == TARGET_STATUS
        ),
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
    }
    if family_vector.get("schema") == FAMILY_VECTOR_SCHEMA:
        binding: dict[str, Any] = {
            "schema": BALANCE_BINDING_SCHEMA,
            "materialization_identity_sha256": family_vector[
                "materialization_identity_sha256"
            ],
            "record_payload_jsonl_sha256": family_vector[
                "record_payload_jsonl_sha256"
            ],
            "record_inventory_digest_sha256": family_vector[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": family_vector[
                "payload_inventory_digest_sha256"
            ],
            **binding_common,
        }
    else:
        binding = {
            "schema": CURRENT_CLEAN_BALANCE_BINDING_SCHEMA,
            "current_clean_physical_authority": expected_physical,
            **binding_common,
        }
    binding["binding_identity_sha256"] = _self_hash(
        binding, "binding_identity_sha256"
    )
    return binding


def require_balanced_selection_ready(
    binding: Mapping[str, Any],
    *,
    expected_binding_identity_sha256: str,
) -> None:
    """Fail closed unless #838 reached its exact 20M target on this physical universe."""

    if binding.get("schema") not in {
        BALANCE_BINDING_SCHEMA,
        CURRENT_CLEAN_BALANCE_BINDING_SCHEMA,
    }:
        raise ProjectionError("unsupported balance binding schema")
    claimed = _require_sha256(
        binding.get("binding_identity_sha256"), "binding_identity_sha256"
    )
    if claimed != _self_hash(binding, "binding_identity_sha256"):
        raise ProjectionError("balance binding self-hash mismatch")
    if claimed != _require_sha256(
        expected_binding_identity_sha256, "expected_binding_identity_sha256"
    ):
        raise ProjectionError("balance binding does not match external expectation")
    if (
        type(binding.get("authorized_optimized_target_exposure")) is not int
        or binding["authorized_optimized_target_exposure"] != 0
    ):
        raise ProjectionError("balance binding fabricated optimized-target exposure")
    if binding.get("balance_policy_identity_sha256") != (
        _TARGET_20M_POLICY_IDENTITY_SHA256
    ):
        raise ProjectionError("balance binding policy is not independently pinned")
    if binding.get("tokenizer_fit_authorized") is not False:
        raise ProjectionError("balance binding fabricated tokenizer authority")
    if binding.get("model_training_authorized") is not False:
        raise ProjectionError("balance binding fabricated training authority")
    if binding.get("balance_status") != TARGET_STATUS:
        raise ProjectionError(
            "balanced selection blocked until TARGET_20M_SOURCE_MIX_FEASIBLE"
        )
    if type(binding.get("target_total_source_bytes")) is not int or binding.get(
        "target_total_source_bytes"
    ) != 20_000_000:
        raise ProjectionError("balance target total drift")
    if type(binding.get("maximum_feasible_total_source_bytes")) is not int or binding.get(
        "maximum_feasible_total_source_bytes"
    ) != 20_000_000:
        raise ProjectionError("balance maximum does not reach the 20M target")
    for field in ("target_stratum_bytes", "maximum_feasible_stratum_bytes"):
        observed = binding.get(field)
        if not isinstance(observed, Mapping) or set(observed) != set(_TARGET_20M_STRATUM_BYTES):
            raise ProjectionError(f"balance {field} schema drift")
        for stratum, required in _TARGET_20M_STRATUM_BYTES.items():
            if type(observed[stratum]) is not int or observed[stratum] != required:
                raise ProjectionError(f"balance {field} violates 45/35/20")
    if binding.get("balanced_selection_authorized") is not True:
        raise ProjectionError("balanced selection authorization flag is false")
