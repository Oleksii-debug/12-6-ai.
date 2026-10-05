"""Compose authenticated current-clean and post-G05/G06 code-delta inventories.

This module is deliberately narrower than materialization and balance execution. It
accepts two independently pinned, text-free physical inventories, authenticates the
#2752 delta execution envelope, rejects overlap/collision, and derives one combined
family-capacity authority. It grants zero training/capacity credit and does not
pretend the union is a new materialization receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.trusted_family_authority_current import (
    TRUSTED_FAMILY_SEMANTICS,
    trusted_family_authority_root_sha256,
)

INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
DELTA_SCHEMA = "12-6.d03-code-delta-decontam-g05-g06-execution.v1"
COMPOSITION_SCHEMA = "12-6.d03-current-plus-code-delta-inventory-composition.v1"
EXPECTED_DELTA_PARENT_HEAD = "cc8dbf35f592f9c9a856e2b711278f4ee0205a5c"
EXPECTED_DELTA_PARENT_MATCHER_REPORT_SHA256 = (
    "b354114771d22fc1be59a74007baea2956a9e5b8311596c05eb3f6c92f6ea418"
)
EXPECTED_DELTA_PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "1312186bb19a1e65603afd95fa29bb8f44488b28b86df7136b0a99b3ef8b0254"
)
EXPECTED_DELTA_PARENT_PROOF_IDENTITY_SHA256 = (
    "5f5fe9551692da2b8c037c20b1261bdbb3d707dbc7bfb13776429ab4e792b1bb"
)
EXPECTED_DELTA_PRE_GATE_OBJECTS = 18
EXPECTED_DELTA_PRE_GATE_BYTES = 418_487
CURRENT_POST_QP_CODE_BYTES = 3_664_247
CODE_TARGET_BYTES = 4_000_000
_HEX = frozenset("0123456789abcdef")

_DELTA_TOP_LEVEL_KEYS = {
    "schema_version",
    "execution_profile",
    "execution_head_sha",
    "parent",
    "delta_authority",
    "gate_execution",
    "survivor_inventory",
    "capacity_projection",
    "authority_blobs",
    "content_boundary",
    "truth_boundary",
    "evidence_identity_sha256",
}
_DELTA_PARENT_KEYS = {
    "execution_head_sha",
    "workflow_run_id",
    "artifact_id",
    "artifact_zip_sha256",
    "combined_matcher_report_sha256",
    "survivor_authority_sha256",
    "two_clean_proof_identity_sha256",
    "marginal_global_unique_reserve_bytes",
    "projected_late_gate_headroom_before_execution",
}
_DELTA_AUTHORITY_KEYS = {
    "source_object_count",
    "pre_gate_payload_bytes",
    "inventory_identity_sha256",
    "subset_authority_sha256",
    "training_records_sha256",
    "training_handoff_raw_sha256",
    "training_handoff_identity_sha256",
    "source_ids_sha256",
    "source_authority",
}
_GATE_KEYS = {
    "data232_report_sha256",
    "decontamination_execution_identity_sha256",
    "eval647_execution_receipt_identity_sha256",
    "quality_execution_identity_sha256",
    "privacy_execution_identity_sha256",
    "composition_receipt_identity_sha256",
    "input_records",
    "post_decontamination_records",
    "post_quality_records",
    "survivor_records",
    "survivor_source_objects",
    "survivor_payload_bytes",
    "survivor_jsonl_sha256",
    "survivor_record_inventory_digest_sha256",
    "survivor_payload_inventory_digest_sha256",
    "rejection_counts",
    "privacy_detector_counts",
    "later_gate_loss_bytes",
}
_EXPECTED_CONTENT_BOUNDARY = {
    "raw_training_text_persisted": False,
    "raw_evaluation_text_persisted": False,
    "raw_survivor_text_persisted": False,
    "durable_output_text_free": True,
}
_EXPECTED_TRUTH_BOUNDARY = {
    "global_cross_source_dedup_complete_for_delta": True,
    "reserved_evaluation_decontamination_complete_for_delta": True,
    "canonical_quality_privacy_complete_for_delta": True,
    "balance_diversity_retest_complete": False,
    "family_caps_complete": False,
    "cluster_safe_split_complete": False,
    "deterministic_pack_two_clean_complete": False,
    "positive_exact_unique_loss_ledger": False,
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
}

_COMPOSITION_KEYS = {
    "schema",
    "status",
    "source_git_sha",
    "base",
    "delta",
    "combined_inventory",
    "cross_inventory_record_id_collision_free",
    "cross_inventory_source_id_collision_free",
    "cross_inventory_exact_payload_collision_free",
    "record_membership_sha256",
    "trusted_family_authority_root_sha256",
    "families",
    "stratum_capacity_bytes",
    "stratum_family_counts",
    "next_gate",
    "canonical_capacity_credited",
    "training_authorized_bytes",
    "authorized_unique_loss_positions",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "model_training_authorized",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "composition_identity_sha256",
}
_BASE_KEYS = {
    "physical_authority_identity_sha256",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "record_count",
    "total_payload_bytes",
    "source_object_count",
}
_DELTA_LINEAGE_KEYS = {
    "execution_head_sha",
    "evidence_identity_sha256",
    "parent_execution_head_sha",
    "parent_survivor_authority_sha256",
    "parent_two_clean_proof_identity_sha256",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "record_count",
    "total_payload_bytes",
    "source_object_count",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _self_hash(document: Mapping[str, Any], field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    return _sha256(body)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProjectionError(message)


def _hex(value: Any, field: str, length: int) -> str:
    _require(
        isinstance(value, str)
        and len(value) == length
        and set(value) <= _HEX,
        f"{field} must be {length} lowercase hex characters",
    )
    return str(value)


def _sha(value: Any, field: str) -> str:
    return _hex(value, field, 64)


def _git(value: Any, field: str) -> str:
    return _hex(value, field, 40)


def _positive(value: Any, field: str) -> int:
    _require(type(value) is int and value > 0, f"{field} must be a positive integer")
    return int(value)


def _nonnegative(value: Any, field: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{field} must be a non-negative integer",
    )
    return int(value)


def _inventory_rows(
    inventory: Mapping[str, Any],
    *,
    label: str,
    expected_record_count: int,
    expected_total_payload_bytes: int,
    expected_source_object_count: int,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
) -> list[dict[str, Any]]:
    _require(inventory.get("schema_version") == INVENTORY_SCHEMA, f"{label} schema drift")
    rows = inventory.get("records")
    _require(isinstance(rows, list) and bool(rows), f"{label} records missing")
    required = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
    }
    normalized: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for index, raw in enumerate(rows):
        _require(
            isinstance(raw, Mapping) and set(raw) == required,
            f"{label}.records[{index}] schema drift",
        )
        row = dict(raw)
        for field in ("record_id", "source_id", "family", "modality"):
            _require(
                isinstance(row[field], str) and bool(row[field]),
                f"{label}.records[{index}].{field} invalid",
            )
        _sha(row["payload_sha256"], f"{label}.records[{index}].payload_sha256")
        _positive(row["payload_bytes"], f"{label}.records[{index}].payload_bytes")
        _require(
            row["record_id"] not in seen_records,
            f"{label} duplicate record_id: {row['record_id']}",
        )
        seen_records.add(row["record_id"])
        normalized.append(row)
    normalized.sort(key=lambda row: row["record_id"])
    _require(normalized == rows, f"{label} record order drift")

    record_count = len(normalized)
    total_bytes = sum(int(row["payload_bytes"]) for row in normalized)
    source_count = len({str(row["source_id"]) for row in normalized})
    record_root = _sha256(normalized)
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    payload_root = _sha256(payload_projection)
    declared = {
        "record_count": record_count,
        "total_payload_bytes": total_bytes,
        "record_inventory_digest_sha256": record_root,
        "payload_inventory_digest_sha256": payload_root,
    }
    for field, actual in declared.items():
        _require(
            type(inventory.get(field)) is type(actual) and inventory.get(field) == actual,
            f"{label} declared {field} drift",
        )
    expected = {
        "record_count": expected_record_count,
        "total_payload_bytes": expected_total_payload_bytes,
        "source_object_count": expected_source_object_count,
        "record_inventory_digest_sha256": _sha(
            expected_record_inventory_digest_sha256,
            f"expected {label} record root",
        ),
        "payload_inventory_digest_sha256": _sha(
            expected_payload_inventory_digest_sha256,
            f"expected {label} payload root",
        ),
    }
    actual = {
        "record_count": record_count,
        "total_payload_bytes": total_bytes,
        "source_object_count": source_count,
        "record_inventory_digest_sha256": record_root,
        "payload_inventory_digest_sha256": payload_root,
    }
    for field, expected_value in expected.items():
        _require(
            actual[field] == expected_value,
            f"{label} {field} does not match independent authority",
        )
    return normalized


def _verify_delta_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_execution_head_sha: str,
) -> list[dict[str, Any]]:
    _require(set(evidence) == _DELTA_TOP_LEVEL_KEYS, "delta evidence fields drift")
    _require(evidence.get("schema_version") == DELTA_SCHEMA, "delta evidence schema drift")
    _require(evidence.get("execution_profile") == "LOCAL_FREE", "delta is not LOCAL_FREE")
    _require(
        evidence.get("execution_head_sha")
        == _git(expected_execution_head_sha, "expected delta execution head"),
        "delta execution head drift",
    )
    claimed = _sha(evidence.get("evidence_identity_sha256"), "delta evidence identity")
    _require(
        claimed == _sha(expected_identity_sha256, "expected delta evidence identity"),
        "delta evidence identity is not independently expected",
    )
    _require(
        claimed == _self_hash(evidence, "evidence_identity_sha256"),
        "delta evidence self-hash mismatch",
    )

    parent = evidence.get("parent")
    _require(isinstance(parent, Mapping) and set(parent) == _DELTA_PARENT_KEYS, "delta parent fields drift")
    _require(parent.get("execution_head_sha") == EXPECTED_DELTA_PARENT_HEAD, "delta parent head drift")
    _require(
        parent.get("combined_matcher_report_sha256")
        == EXPECTED_DELTA_PARENT_MATCHER_REPORT_SHA256,
        "delta parent matcher root drift",
    )
    _require(
        parent.get("survivor_authority_sha256")
        == EXPECTED_DELTA_PARENT_SURVIVOR_AUTHORITY_SHA256,
        "delta parent survivor authority drift",
    )
    _require(
        parent.get("two_clean_proof_identity_sha256")
        == EXPECTED_DELTA_PARENT_PROOF_IDENTITY_SHA256,
        "delta parent two-clean proof drift",
    )
    _positive(parent.get("workflow_run_id"), "delta parent workflow_run_id")
    _positive(parent.get("artifact_id"), "delta parent artifact_id")
    _sha(parent.get("artifact_zip_sha256"), "delta parent artifact_zip_sha256")
    _positive(
        parent.get("marginal_global_unique_reserve_bytes"),
        "delta parent marginal unique reserve bytes",
    )
    _nonnegative(
        parent.get("projected_late_gate_headroom_before_execution"),
        "delta parent projected headroom",
    )

    authority = evidence.get("delta_authority")
    _require(
        isinstance(authority, Mapping) and set(authority) == _DELTA_AUTHORITY_KEYS,
        "delta authority fields drift",
    )
    _require(
        authority.get("source_object_count") == EXPECTED_DELTA_PRE_GATE_OBJECTS,
        "delta pre-gate object count drift",
    )
    _require(
        authority.get("pre_gate_payload_bytes") == EXPECTED_DELTA_PRE_GATE_BYTES,
        "delta pre-gate payload byte count drift",
    )
    for field in (
        "inventory_identity_sha256",
        "subset_authority_sha256",
        "training_records_sha256",
        "training_handoff_raw_sha256",
        "training_handoff_identity_sha256",
        "source_ids_sha256",
    ):
        _sha(authority.get(field), f"delta authority.{field}")
    source_authority = authority.get("source_authority")
    _require(
        isinstance(source_authority, Mapping) and bool(source_authority),
        "delta source authority missing",
    )
    for field, value in source_authority.items():
        _require(isinstance(field, str) and bool(field), "delta source authority key invalid")
        _sha(value, f"delta source authority.{field}")

    gate = evidence.get("gate_execution")
    _require(isinstance(gate, Mapping) and set(gate) == _GATE_KEYS, "delta gate fields drift")
    for field in (
        "data232_report_sha256",
        "decontamination_execution_identity_sha256",
        "eval647_execution_receipt_identity_sha256",
        "quality_execution_identity_sha256",
        "privacy_execution_identity_sha256",
        "composition_receipt_identity_sha256",
        "survivor_jsonl_sha256",
        "survivor_record_inventory_digest_sha256",
        "survivor_payload_inventory_digest_sha256",
    ):
        _sha(gate.get(field), f"delta gate.{field}")
    input_records = _positive(gate.get("input_records"), "delta gate input_records")
    post_decontam = _positive(
        gate.get("post_decontamination_records"),
        "delta gate post_decontamination_records",
    )
    post_quality = _positive(gate.get("post_quality_records"), "delta gate post_quality_records")
    survivors = _positive(gate.get("survivor_records"), "delta gate survivor_records")
    _require(
        input_records >= post_decontam >= post_quality >= survivors,
        "delta gate record count monotonicity drift",
    )
    source_objects = _positive(
        gate.get("survivor_source_objects"), "delta gate survivor_source_objects"
    )
    _require(source_objects <= survivors, "delta survivor source count exceeds records")
    survivor_bytes = _positive(gate.get("survivor_payload_bytes"), "delta survivor bytes")
    loss = _nonnegative(gate.get("later_gate_loss_bytes"), "delta later gate loss bytes")
    _require(
        survivor_bytes + loss == EXPECTED_DELTA_PRE_GATE_BYTES,
        "delta later-gate byte accounting drift",
    )
    for field in ("rejection_counts", "privacy_detector_counts"):
        values = gate.get(field)
        _require(isinstance(values, Mapping), f"delta gate.{field} missing")
        for key, value in values.items():
            _require(isinstance(key, str) and bool(key), f"delta gate.{field} key invalid")
            _nonnegative(value, f"delta gate.{field}.{key}")

    inventory = evidence.get("survivor_inventory")
    _require(isinstance(inventory, Mapping), "delta survivor inventory missing")
    rows = _inventory_rows(
        inventory,
        label="delta survivor inventory",
        expected_record_count=survivors,
        expected_total_payload_bytes=survivor_bytes,
        expected_source_object_count=source_objects,
        expected_record_inventory_digest_sha256=gate[
            "survivor_record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=gate[
            "survivor_payload_inventory_digest_sha256"
        ],
    )
    _require(
        all(row["modality"] == "code" for row in rows),
        "delta survivor inventory contains non-code modality",
    )

    capacity = evidence.get("capacity_projection")
    _require(isinstance(capacity, Mapping), "delta capacity projection missing")
    _require(
        capacity.get("current_post_qp_code_bytes_reference") == CURRENT_POST_QP_CODE_BYTES,
        "delta current post-Q/P code reference drift",
    )
    _require(capacity.get("code_target_bytes") == CODE_TARGET_BYTES, "delta code target drift")
    _require(
        capacity.get("pre_gate_gap_bytes") == CODE_TARGET_BYTES - CURRENT_POST_QP_CODE_BYTES,
        "delta pre-gate gap drift",
    )
    _require(
        capacity.get("qualified_delta_survivor_bytes") == survivor_bytes,
        "delta qualified survivor byte drift",
    )
    projected = CURRENT_POST_QP_CODE_BYTES + survivor_bytes
    _require(
        capacity.get("projected_post_qp_code_bytes") == projected,
        "delta projected code bytes drift",
    )
    headroom = projected - CODE_TARGET_BYTES
    _require(
        capacity.get("projected_headroom_after_decontam_g05_g06") == headroom,
        "delta projected headroom drift",
    )
    _require(
        capacity.get("still_possible_to_reach_code_target_after_balance_split_pack")
        is (headroom >= 0),
        "delta code-target possibility flag drift",
    )

    _require(evidence.get("content_boundary") == _EXPECTED_CONTENT_BOUNDARY, "delta content boundary drift")
    _require(evidence.get("truth_boundary") == _EXPECTED_TRUTH_BOUNDARY, "delta truth boundary drift")
    blobs = evidence.get("authority_blobs")
    _require(isinstance(blobs, Mapping) and bool(blobs), "delta authority Git blobs missing")
    for path, blob in blobs.items():
        _require(isinstance(path, str) and bool(path), "delta authority blob path invalid")
        _git(blob, f"delta authority blob.{path}")
    return rows


def _family_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_bytes: dict[tuple[str, str], int] = defaultdict(int)
    family_records: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        family = str(row["family"])
        authority = TRUSTED_FAMILY_SEMANTICS.get(family)
        _require(authority is not None, f"family absent from current authority: {family}")
        stratum = str(authority["stratum"])
        modality = str(row["modality"])
        if stratum == "code":
            _require(modality == "code", f"code family modality drift: {family}")
        elif stratum == "en":
            _require(modality in {"en", "text"}, f"English family modality drift: {family}")
        elif stratum == "uk":
            _require(modality in {"uk", "ua", "text"}, f"Ukrainian family modality drift: {family}")
        else:
            raise ProjectionError(f"unsupported family stratum: {family}")
        key = (stratum, family)
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
    names = {row["family"] for row in families}
    membership = [
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
        "stratum_capacity_bytes": {
            stratum: sum(
                row["capacity_bytes"] for row in families if row["stratum"] == stratum
            )
            for stratum in ("code", "en", "uk")
        },
        "stratum_family_counts": {
            stratum: sum(1 for row in families if row["stratum"] == stratum)
            for stratum in ("code", "en", "uk")
        },
        "record_membership_sha256": _sha256(membership),
        "trusted_family_authority_root_sha256": trusted_family_authority_root_sha256(names),
    }


def compose_current_clean_and_delta_inventory(
    *,
    base_inventory: Mapping[str, Any],
    delta_evidence: Mapping[str, Any],
    expected_base_physical_authority_identity_sha256: str,
    expected_base_record_count: int,
    expected_base_total_payload_bytes: int,
    expected_base_source_object_count: int,
    expected_base_record_inventory_digest_sha256: str,
    expected_base_payload_inventory_digest_sha256: str,
    expected_delta_evidence_identity_sha256: str,
    expected_delta_execution_head_sha: str,
    source_git_sha: str,
) -> dict[str, Any]:
    """Build a zero-credit combined physical inventory and family projection."""

    base_identity = _sha(
        expected_base_physical_authority_identity_sha256,
        "expected base physical authority identity",
    )
    source_git_sha = _git(source_git_sha, "source_git_sha")
    base_rows = _inventory_rows(
        base_inventory,
        label="base current-clean inventory",
        expected_record_count=_positive(expected_base_record_count, "expected base record count"),
        expected_total_payload_bytes=_positive(
            expected_base_total_payload_bytes, "expected base total bytes"
        ),
        expected_source_object_count=_positive(
            expected_base_source_object_count, "expected base source object count"
        ),
        expected_record_inventory_digest_sha256=expected_base_record_inventory_digest_sha256,
        expected_payload_inventory_digest_sha256=expected_base_payload_inventory_digest_sha256,
    )
    delta_rows = _verify_delta_evidence(
        delta_evidence,
        expected_identity_sha256=expected_delta_evidence_identity_sha256,
        expected_execution_head_sha=expected_delta_execution_head_sha,
    )

    base_record_ids = {row["record_id"] for row in base_rows}
    delta_record_ids = {row["record_id"] for row in delta_rows}
    overlap_records = base_record_ids & delta_record_ids
    _require(not overlap_records, "base/delta record_id collision")

    base_source_ids = {row["source_id"] for row in base_rows}
    delta_source_ids = {row["source_id"] for row in delta_rows}
    overlap_sources = base_source_ids & delta_source_ids
    _require(not overlap_sources, "base/delta source_id collision")

    base_payloads = {row["payload_sha256"] for row in base_rows}
    delta_payloads = {row["payload_sha256"] for row in delta_rows}
    _require(
        len(delta_payloads) == len(delta_rows),
        "delta post-gate exact payload collision requires re-deduplication",
    )
    _require(
        not (base_payloads & delta_payloads),
        "base/delta post-gate exact payload collision requires re-deduplication",
    )

    combined_rows = sorted([*base_rows, *delta_rows], key=lambda row: row["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in combined_rows
    ]
    combined_inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "record_count": len(combined_rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in combined_rows),
        "record_inventory_digest_sha256": _sha256(combined_rows),
        "payload_inventory_digest_sha256": _sha256(payload_projection),
        "records": combined_rows,
    }
    family = _family_projection(combined_rows)
    delta_gate = delta_evidence["gate_execution"]

    composition: dict[str, Any] = {
        "schema": COMPOSITION_SCHEMA,
        "status": "COMPOSED_ZERO_CREDIT_PENDING_BALANCE",
        "source_git_sha": source_git_sha,
        "base": {
            "physical_authority_identity_sha256": base_identity,
            "record_inventory_digest_sha256": _sha(
                expected_base_record_inventory_digest_sha256,
                "base record inventory digest",
            ),
            "payload_inventory_digest_sha256": _sha(
                expected_base_payload_inventory_digest_sha256,
                "base payload inventory digest",
            ),
            "record_count": expected_base_record_count,
            "total_payload_bytes": expected_base_total_payload_bytes,
            "source_object_count": expected_base_source_object_count,
        },
        "delta": {
            "execution_head_sha": _git(
                expected_delta_execution_head_sha, "delta execution head"
            ),
            "evidence_identity_sha256": _sha(
                expected_delta_evidence_identity_sha256, "delta evidence identity"
            ),
            "parent_execution_head_sha": EXPECTED_DELTA_PARENT_HEAD,
            "parent_survivor_authority_sha256": (
                EXPECTED_DELTA_PARENT_SURVIVOR_AUTHORITY_SHA256
            ),
            "parent_two_clean_proof_identity_sha256": (
                EXPECTED_DELTA_PARENT_PROOF_IDENTITY_SHA256
            ),
            "record_inventory_digest_sha256": delta_gate[
                "survivor_record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": delta_gate[
                "survivor_payload_inventory_digest_sha256"
            ],
            "record_count": delta_gate["survivor_records"],
            "total_payload_bytes": delta_gate["survivor_payload_bytes"],
            "source_object_count": delta_gate["survivor_source_objects"],
        },
        "combined_inventory": combined_inventory,
        "cross_inventory_record_id_collision_free": True,
        "cross_inventory_source_id_collision_free": True,
        "cross_inventory_exact_payload_collision_free": True,
        "record_membership_sha256": family["record_membership_sha256"],
        "trusted_family_authority_root_sha256": family[
            "trusted_family_authority_root_sha256"
        ],
        "families": family["families"],
        "stratum_capacity_bytes": family["stratum_capacity_bytes"],
        "stratum_family_counts": family["stratum_family_counts"],
        "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP",
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    composition["composition_identity_sha256"] = _self_hash(
        composition, "composition_identity_sha256"
    )
    return composition


def verify_current_clean_and_delta_composition(
    document: Mapping[str, Any],
    *,
    expected_composition_identity_sha256: str | None = None,
) -> str:
    _require(set(document) == _COMPOSITION_KEYS, "composition fields are not closed-world")
    _require(document.get("schema") == COMPOSITION_SCHEMA, "composition schema drift")
    _require(
        document.get("status") == "COMPOSED_ZERO_CREDIT_PENDING_BALANCE",
        "composition status drift",
    )
    claimed = _sha(document.get("composition_identity_sha256"), "composition identity")
    _require(
        claimed == _self_hash(document, "composition_identity_sha256"),
        "composition self-hash mismatch",
    )
    if expected_composition_identity_sha256 is not None:
        _require(
            claimed
            == _sha(
                expected_composition_identity_sha256,
                "expected composition identity",
            ),
            "composition identity is not independently expected",
        )
    _git(document.get("source_git_sha"), "composition source_git_sha")

    base = document.get("base")
    delta = document.get("delta")
    _require(isinstance(base, Mapping) and set(base) == _BASE_KEYS, "composition base fields drift")
    _require(
        isinstance(delta, Mapping) and set(delta) == _DELTA_LINEAGE_KEYS,
        "composition delta lineage fields drift",
    )
    _sha(base.get("physical_authority_identity_sha256"), "base physical authority identity")
    _sha(base.get("record_inventory_digest_sha256"), "base record inventory root")
    _sha(base.get("payload_inventory_digest_sha256"), "base payload inventory root")
    base_records = _positive(base.get("record_count"), "base record count")
    base_bytes = _positive(base.get("total_payload_bytes"), "base total bytes")
    base_sources = _positive(base.get("source_object_count"), "base source object count")
    _require(base_sources <= base_records, "base source object count exceeds record count")

    _git(delta.get("execution_head_sha"), "delta execution head")
    _sha(delta.get("evidence_identity_sha256"), "delta evidence identity")
    _require(
        delta.get("parent_execution_head_sha") == EXPECTED_DELTA_PARENT_HEAD,
        "composition delta parent head drift",
    )
    _require(
        delta.get("parent_survivor_authority_sha256")
        == EXPECTED_DELTA_PARENT_SURVIVOR_AUTHORITY_SHA256,
        "composition delta parent survivor authority drift",
    )
    _require(
        delta.get("parent_two_clean_proof_identity_sha256")
        == EXPECTED_DELTA_PARENT_PROOF_IDENTITY_SHA256,
        "composition delta parent proof drift",
    )
    _sha(delta.get("record_inventory_digest_sha256"), "delta record inventory root")
    _sha(delta.get("payload_inventory_digest_sha256"), "delta payload inventory root")
    delta_records = _positive(delta.get("record_count"), "delta record count")
    delta_bytes = _positive(delta.get("total_payload_bytes"), "delta total bytes")
    delta_sources = _positive(delta.get("source_object_count"), "delta source object count")
    _require(delta_sources <= delta_records, "delta source object count exceeds record count")

    for field in (
        "cross_inventory_record_id_collision_free",
        "cross_inventory_source_id_collision_free",
        "cross_inventory_exact_payload_collision_free",
    ):
        _require(document.get(field) is True, f"composition collision proof weakened: {field}")
    _sha(document.get("record_membership_sha256"), "composition record membership")
    _sha(
        document.get("trusted_family_authority_root_sha256"),
        "composition trusted family authority root",
    )
    _require(
        document.get("next_gate") == "NEXT100-106_BALANCE_FAMILY_CAP",
        "composition next gate drift",
    )
    exact_zero = {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    for field, expected in exact_zero.items():
        _require(
            type(document.get(field)) is type(expected) and document.get(field) == expected,
            f"composition authority widened: {field}",
        )

    inventory = document.get("combined_inventory")
    _require(isinstance(inventory, Mapping), "combined inventory missing")
    combined_records = base_records + delta_records
    combined_bytes = base_bytes + delta_bytes
    combined_sources = base_sources + delta_sources
    rows = _inventory_rows(
        inventory,
        label="combined inventory",
        expected_record_count=combined_records,
        expected_total_payload_bytes=combined_bytes,
        expected_source_object_count=combined_sources,
        expected_record_inventory_digest_sha256=inventory.get(
            "record_inventory_digest_sha256"
        ),
        expected_payload_inventory_digest_sha256=inventory.get(
            "payload_inventory_digest_sha256"
        ),
    )
    _require(
        not (
            {row["record_id"] for row in rows[:base_records]}
            & {row["record_id"] for row in rows[base_records:]}
        ),
        "combined inventory record collision proof drift",
    )
    family = _family_projection(rows)
    _require(document.get("families") == family["families"], "composition family rows drift")
    _require(
        document.get("stratum_capacity_bytes") == family["stratum_capacity_bytes"],
        "composition stratum capacity drift",
    )
    _require(
        document.get("stratum_family_counts") == family["stratum_family_counts"],
        "composition stratum family-count drift",
    )
    _require(
        document.get("record_membership_sha256") == family["record_membership_sha256"],
        "composition record membership drift",
    )
    _require(
        document.get("trusted_family_authority_root_sha256")
        == family["trusted_family_authority_root_sha256"],
        "composition trusted family root drift",
    )
    return claimed

