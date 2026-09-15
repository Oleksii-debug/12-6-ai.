"""Authenticated post-G05/G06 materialization -> NEXT100-106 balance bridge.

This module does not implement a second balance policy. It authenticates the durable
text-free post-materialization inventory from D03, derives the exact current family
capacity vector, adapts that vector to the already-merged NEXT100-106 gate, and binds
the gate result back to the exact physical materialization authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.trusted_family_authority_v1 import (
    TRUSTED_FAMILY_SEMANTICS,
    trusted_family_authority_root_sha256,
)

FAMILY_VECTOR_SCHEMA = "12-6.d03-postmaterialization-family-vector.v1"
MATERIALIZATION_SCHEMA = "12-6.d03-post-g05-g06-materialization.v1"
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
NEXT100_INPUT_SCHEMA = "12-6.next100-106-post-dedup-family-vector.v1"
BALANCE_RESULT_SCHEMA = "12-6.next100-106-balance-gate-result.v1"
BALANCE_BINDING_SCHEMA = "12-6.d03-postmaterialization-balance-binding.v1"
TARGET_STATUS = "TARGET_20M_SOURCE_MIX_FEASIBLE"
STRATUM_MAP = {"uk": "ua", "en": "en", "code": "code"}
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hash(document: Mapping[str, Any], identity_field: str) -> str:
    clean = dict(document)
    clean.pop(identity_field, None)
    return _sha256(clean)


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
        "record_membership_sha256": _sha256(record_membership),
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


def verify_postmaterialization_family_vector(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str | None = None,
) -> str:
    if document.get("schema") != FAMILY_VECTOR_SCHEMA:
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
    if document.get("authorized_optimized_target_exposure") != 0:
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
        "physical_authority": {
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
        },
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
    if boundary != expected_boundary:
        raise ProjectionError("NEXT100-106 result claim boundary drift")
    return claimed


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
    if balance_result.get("dedup_authority") != next100_input.get("dedup_authority"):
        raise ProjectionError("NEXT100-106 result dedup authority differs from input")
    if balance_result.get("input_totals") != next100_input.get("totals"):
        raise ProjectionError("NEXT100-106 result totals differ from physical input")

    physical = next100_input.get("physical_authority")
    if not isinstance(physical, Mapping):
        raise ProjectionError("NEXT100-106 input lacks physical authority extension")
    expected_physical = {
        "family_vector_identity_sha256": family_identity,
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
    if dict(physical) != expected_physical:
        raise ProjectionError("NEXT100-106 input physical authority mismatch")

    binding: dict[str, Any] = {
        "schema": BALANCE_BINDING_SCHEMA,
        "status": "BOUND_ZERO_CREDIT",
        "family_vector_identity_sha256": family_identity,
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

    if binding.get("schema") != BALANCE_BINDING_SCHEMA:
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
    if binding.get("authorized_optimized_target_exposure") != 0:
        raise ProjectionError("balance binding fabricated optimized-target exposure")
    if binding.get("tokenizer_fit_authorized") is not False:
        raise ProjectionError("balance binding fabricated tokenizer authority")
    if binding.get("model_training_authorized") is not False:
        raise ProjectionError("balance binding fabricated training authority")
    if binding.get("balance_status") != TARGET_STATUS:
        raise ProjectionError(
            "balanced selection blocked until TARGET_20M_SOURCE_MIX_FEASIBLE"
        )
    if binding.get("target_total_source_bytes") != 20_000_000:
        raise ProjectionError("balance target total drift")
    if binding.get("maximum_feasible_total_source_bytes") != 20_000_000:
        raise ProjectionError("balance maximum does not reach the 20M target")
    if binding.get("balanced_selection_authorized") is not True:
        raise ProjectionError("balanced selection authorization flag is false")
