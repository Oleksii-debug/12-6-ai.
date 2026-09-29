"""Materialize the exact current-clean record selection required by canonical split.

This module is a record-granularity realization seam only. It does not change the
canonical NEXT100-106 balance policy or the canonical split algorithm. A terminal
family-byte allocation is accepted only when whole authenticated survivor records can
realize every allocated family byte exactly. Otherwise the seam fails closed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postmaterialization_balance_projection_v1 import (
    CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA,
    _canonical_bytes,
    _rebuild_current_clean_survivor_inventory,
    _require_nonnegative_int,
    _require_sha256,
    _self_hash,
    _sha256_bytes,
    build_balance_result_binding,
    load_strict_json_object,
    require_balanced_selection_ready,
    verify_postmaterialization_family_vector,
)
from twelve_six.data.trusted_family_authority_v1 import TRUSTED_FAMILY_SEMANTICS

SELECTION_SCHEMA = "12-6.d03-balanced-selection-authority.v1"
SELECTION_REALIZATION_POLICY = "record-id-ascending-exact-family-byte-subset-v1"
_ALLOWED_ALLOCATION_FIELDS = {
    "family_id",
    "stratum",
    "allocated_bytes",
    "available_unique_bytes",
    "effective_family_cap_bytes",
}
_STRATUM_TO_BALANCE = {"uk": "ua", "en": "en", "code": "code"}
_SELECTION_FIELDS = {
    "schema",
    "terminal",
    "status",
    "balanced_selection_identity_sha256",
    "retained_inventory_identity_sha256",
    "decontamination_authority_sha256",
    "dedup_authority_sha256",
    "balance_policy_identity_sha256",
    "balance_result_identity_sha256",
    "records",
    "totals",
    "claim_boundary",
}
_SELECTION_ROW_FIELDS = {
    "record_id",
    "source_id",
    "family",
    "stratum",
    "modality",
    "payload_sha256",
    "payload_bytes",
    "near_duplicate_cluster_id",
    "purpose",
    "training_eligible",
    "evaluation_eligible",
    "evaluation_reserved",
}
_CLAIM_BOUNDARY = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "paid_compute_authorized": False,
    "final_test_outcomes_read": False,
    "authorized_optimized_target_exposure": 0,
}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionError(f"{field} must be non-empty text")
    return value


def _parse_survivor_records(raw: bytes) -> list[dict[str, Any]]:
    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise ProjectionError("current-clean survivor JSONL must be non-empty and LF-terminated")

    expected = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "normalized_payload",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, physical in enumerate(raw.splitlines(keepends=True)):
        if not physical.endswith(b"\n"):
            raise ProjectionError(f"current-clean survivor record[{index}] lacks terminal LF")
        line = physical[:-1]
        row = load_strict_json_object(
            line,
            label=f"current-clean survivor record[{index}]",
        )
        if set(row) != expected:
            raise ProjectionError(f"current-clean survivor record[{index}] schema drift")
        if _canonical_bytes(row) != line:
            raise ProjectionError(f"current-clean survivor record[{index}] is not canonical JSON")
        for field in expected:
            _require_text(row.get(field), f"current-clean survivor record[{index}].{field}")
        record_id = row["record_id"]
        if record_id in seen:
            raise ProjectionError(f"duplicate current-clean survivor record_id: {record_id}")
        seen.add(record_id)

        family = row["family"]
        authority = TRUSTED_FAMILY_SEMANTICS.get(family)
        if authority is None:
            raise ProjectionError(f"survivor family absent from trusted authority: {family}")
        stratum = authority.get("stratum")
        if stratum not in _STRATUM_TO_BALANCE:
            raise ProjectionError(f"unsupported survivor family stratum: {family}")
        modality = row["modality"]
        if stratum == "code" and modality != "code":
            raise ProjectionError(f"code family has non-code survivor modality: {family}")
        if stratum == "en" and modality not in {"en", "text"}:
            raise ProjectionError(f"English family survivor modality drift: {family}")
        if stratum == "uk" and modality not in {"uk", "ua", "text"}:
            raise ProjectionError(f"Ukrainian family survivor modality drift: {family}")

        payload = row["normalized_payload"].encode("utf-8")
        rows.append(
            {
                **row,
                "stratum": stratum,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_bytes": len(payload),
            }
        )
    rows.sort(key=lambda item: item["record_id"])
    return rows


def _exact_record_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_bytes: int,
    family: str,
) -> list[dict[str, Any]]:
    """Return one deterministic exact whole-record realization of a byte allocation."""

    target = _require_nonnegative_int(target_bytes, f"allocation[{family}].allocated_bytes")
    if target <= 0:
        raise ProjectionError(f"allocation[{family}] must be positive")

    ordered = [dict(row) for row in sorted(rows, key=lambda item: str(item["record_id"]))]
    total = sum(_require_nonnegative_int(row.get("payload_bytes"), "payload_bytes") for row in ordered)
    if target > total:
        raise ProjectionError(f"allocation[{family}] exceeds authenticated family capacity")
    if target == total:
        return ordered

    # Sparse exact subset DP. Insertion order is deterministic because record order is
    # deterministic. We stop as soon as the exact allocation first becomes reachable.
    # No partial record, padding, replay or byte truncation is permitted.
    parent: dict[int, tuple[int, int] | None] = {0: None}
    reached_at = -1
    for index, row in enumerate(ordered):
        weight = _require_nonnegative_int(row.get("payload_bytes"), "payload_bytes")
        if weight <= 0:
            raise ProjectionError("selected survivor payload_bytes must be positive")
        existing = tuple(parent)
        for current in existing:
            candidate = current + weight
            if candidate > target or candidate in parent:
                continue
            parent[candidate] = (current, index)
            if candidate == target:
                reached_at = index
                break
        if reached_at >= 0:
            break

    if target not in parent:
        raise ProjectionError(
            f"allocation[{family}] has no exact whole-record realization under "
            f"{SELECTION_REALIZATION_POLICY}"
        )

    selected_indexes: list[int] = []
    cursor = target
    while cursor:
        link = parent.get(cursor)
        if link is None:
            raise ProjectionError(f"allocation[{family}] exact-subset reconstruction failed")
        previous, index = link
        selected_indexes.append(index)
        cursor = previous
    selected_indexes.reverse()
    return [ordered[index] for index in selected_indexes]


def _validated_allocations(
    balance_result: Mapping[str, Any],
    family_vector: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = balance_result.get("deterministic_maximum_allocation")
    if not isinstance(raw, list) or not raw:
        raise ProjectionError("terminal balance result has no deterministic allocation")

    capacities = {
        row["family"]: {
            "stratum": _STRATUM_TO_BALANCE[str(row["stratum"])],
            "capacity": _require_nonnegative_int(row["capacity_bytes"], "capacity_bytes"),
        }
        for row in family_vector["families"]
    }
    allocations: dict[str, dict[str, Any]] = {}
    by_stratum: defaultdict[str, int] = defaultdict(int)
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != _ALLOWED_ALLOCATION_FIELDS:
            raise ProjectionError(f"balance allocation[{index}] fields are not closed-world")
        family = _require_text(item.get("family_id"), f"allocation[{index}].family_id")
        if family in allocations:
            raise ProjectionError(f"duplicate balance allocation family: {family}")
        if family not in capacities:
            raise ProjectionError(f"balance allocation references unknown family: {family}")
        stratum = _require_text(item.get("stratum"), f"allocation[{index}].stratum")
        if stratum != capacities[family]["stratum"]:
            raise ProjectionError(f"balance allocation family/stratum drift: {family}")
        allocated = _require_nonnegative_int(
            item.get("allocated_bytes"),
            f"allocation[{index}].allocated_bytes",
        )
        available = _require_nonnegative_int(
            item.get("available_unique_bytes"),
            f"allocation[{index}].available_unique_bytes",
        )
        cap = _require_nonnegative_int(
            item.get("effective_family_cap_bytes"),
            f"allocation[{index}].effective_family_cap_bytes",
        )
        if allocated <= 0 or available != capacities[family]["capacity"]:
            raise ProjectionError(f"balance allocation family capacity drift: {family}")
        if allocated > available or allocated > cap:
            raise ProjectionError(f"balance allocation exceeds family authority: {family}")
        allocations[family] = dict(item)
        by_stratum[stratum] += allocated

    maximum = _require_nonnegative_int(
        balance_result.get("maximum_feasible_total_source_bytes"),
        "maximum_feasible_total_source_bytes",
    )
    if sum(int(item["allocated_bytes"]) for item in allocations.values()) != maximum:
        raise ProjectionError("balance allocation total does not match maximum feasible total")
    expected_strata = balance_result.get("maximum_feasible_stratum_bytes")
    if not isinstance(expected_strata, Mapping):
        raise ProjectionError("balance result maximum stratum bytes are missing")
    for stratum in ("ua", "en", "code"):
        expected = _require_nonnegative_int(
            expected_strata.get(stratum),
            f"maximum_feasible_stratum_bytes.{stratum}",
        )
        if by_stratum[stratum] != expected:
            raise ProjectionError(f"balance allocation stratum total drift: {stratum}")
    return allocations


def build_current_clean_balanced_selection(
    *,
    family_vector: Mapping[str, Any],
    next100_input: Mapping[str, Any],
    balance_result: Mapping[str, Any],
    balance_binding: Mapping[str, Any],
    composition_receipt_raw: bytes,
    survivor_records_raw: bytes,
    expected_family_vector_identity_sha256: str,
    expected_balance_binding_identity_sha256: str,
    expected_policy_identity_sha256: str,
    expected_result_identity_sha256: str,
) -> dict[str, Any]:
    """Materialize the canonical split selection from exact current-clean records."""

    if family_vector.get("schema") != CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA:
        raise ProjectionError("balanced selection requires current-clean family vector")
    family_identity = verify_postmaterialization_family_vector(
        family_vector,
        expected_identity_sha256=expected_family_vector_identity_sha256,
    )
    rebuilt_binding = build_balance_result_binding(
        family_vector=family_vector,
        expected_family_vector_identity_sha256=family_identity,
        next100_input=next100_input,
        balance_result=balance_result,
        expected_policy_identity_sha256=expected_policy_identity_sha256,
        expected_result_identity_sha256=expected_result_identity_sha256,
    )
    if dict(balance_binding) != rebuilt_binding:
        raise ProjectionError("balance binding differs from deterministic authenticated rebuild")
    require_balanced_selection_ready(
        balance_binding,
        expected_binding_identity_sha256=expected_balance_binding_identity_sha256,
    )

    if _sha256_bytes(composition_receipt_raw) != family_vector.get(
        "composition_receipt_json_sha256"
    ):
        raise ProjectionError("composition receipt raw bytes differ from family-vector authority")
    receipt = load_strict_json_object(
        composition_receipt_raw,
        label="current-clean composition receipt",
    )
    if receipt.get("receipt_identity_sha256") != family_vector.get(
        "current_clean_receipt_identity_sha256"
    ):
        raise ProjectionError("composition receipt identity differs from family-vector authority")
    if _self_hash(receipt, "receipt_identity_sha256") != receipt.get(
        "receipt_identity_sha256"
    ):
        raise ProjectionError("composition receipt self-hash mismatch")

    if _sha256_bytes(survivor_records_raw) != family_vector.get(
        "survivor_records_jsonl_sha256"
    ):
        raise ProjectionError("survivor JSONL bytes differ from family-vector authority")
    survivor_rows = _parse_survivor_records(survivor_records_raw)
    rebuilt_inventory = _rebuild_current_clean_survivor_inventory(survivor_records_raw)
    inventory_checks = {
        "record_count": family_vector.get("record_count"),
        "total_payload_bytes": family_vector.get("total_payload_bytes"),
        "record_inventory_digest_sha256": family_vector.get(
            "record_inventory_digest_sha256"
        ),
        "payload_inventory_digest_sha256": family_vector.get(
            "payload_inventory_digest_sha256"
        ),
    }
    for field, expected in inventory_checks.items():
        if type(rebuilt_inventory.get(field)) is not type(expected) or rebuilt_inventory.get(
            field
        ) != expected:
            raise ProjectionError(f"survivor inventory/family-vector mismatch: {field}")

    allocations = _validated_allocations(balance_result, family_vector)
    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in survivor_rows:
        by_family[row["family"]].append(row)

    selected: list[dict[str, Any]] = []
    for family in sorted(allocations):
        candidates = by_family.get(family, [])
        if not candidates:
            raise ProjectionError(f"allocated family has no authenticated survivor rows: {family}")
        chosen = _exact_record_subset(
            candidates,
            target_bytes=int(allocations[family]["allocated_bytes"]),
            family=family,
        )
        selected.extend(chosen)

    selected.sort(key=lambda row: row["record_id"])
    if len({row["record_id"] for row in selected}) != len(selected):
        raise ProjectionError("balanced selection reuses a survivor record")

    family_bytes: defaultdict[str, int] = defaultdict(int)
    stratum_bytes: defaultdict[str, int] = defaultdict(int)
    output_rows: list[dict[str, Any]] = []
    for row in selected:
        output = {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "family": row["family"],
            "stratum": row["stratum"],
            "modality": row["modality"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
            # Terminal global-dedup already selected one survivor source per duplicate
            # component. Grouping every record from the same survivor source together
            # is conservative and cannot split one source across validation/train.
            "near_duplicate_cluster_id": row["source_id"],
            "purpose": "pretraining_eligible",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        output_rows.append(output)
        family_bytes[row["family"]] += int(row["payload_bytes"])
        stratum_bytes[row["stratum"]] += int(row["payload_bytes"])

    for family, allocation in allocations.items():
        if family_bytes[family] != allocation["allocated_bytes"]:
            raise ProjectionError(f"balanced selection family allocation mismatch: {family}")

    source_bytes = sum(row["payload_bytes"] for row in output_rows)
    if source_bytes != balance_result.get("maximum_feasible_total_source_bytes"):
        raise ProjectionError("balanced selection bytes do not equal terminal balance total")

    dedup = next100_input.get("dedup_authority")
    if not isinstance(dedup, Mapping):
        raise ProjectionError("NEXT100 input lacks dedup authority")
    dedup_identity = _require_sha256(
        dedup.get("evidence_identity_sha256"),
        "dedup evidence_identity_sha256",
    )
    decontamination_identity = _require_sha256(
        receipt.get("decontamination_execution_identity_sha256"),
        "decontamination_execution_identity_sha256",
    )
    policy_identity = _require_sha256(
        expected_policy_identity_sha256,
        "expected_policy_identity_sha256",
    )
    result_identity = _require_sha256(
        expected_result_identity_sha256,
        "expected_result_identity_sha256",
    )

    document: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "terminal": True,
        "status": "PASS",
        "balanced_selection_identity_sha256": "0" * 64,
        "retained_inventory_identity_sha256": family_vector[
            "record_inventory_digest_sha256"
        ],
        "decontamination_authority_sha256": decontamination_identity,
        "dedup_authority_sha256": dedup_identity,
        "balance_policy_identity_sha256": policy_identity,
        "balance_result_identity_sha256": result_identity,
        "records": output_rows,
        "totals": {
            "record_count": len(output_rows),
            "source_bytes": source_bytes,
            "family_source_bytes": dict(sorted(family_bytes.items())),
            "stratum_source_bytes": dict(sorted(stratum_bytes.items())),
        },
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }
    document["balanced_selection_identity_sha256"] = _self_hash(
        document,
        "balanced_selection_identity_sha256",
    )
    return document


def project_selected_current_clean_raw_records(
    selection: Mapping[str, Any],
    survivor_records_raw: bytes,
) -> list[dict[str, Any]]:
    """Project selected current-clean payloads into canonical split raw-record shape."""

    if set(selection) != _SELECTION_FIELDS or selection.get("schema") != SELECTION_SCHEMA:
        raise ProjectionError("balanced selection fields/schema are not closed-world")
    claimed = _require_sha256(
        selection.get("balanced_selection_identity_sha256"),
        "balanced_selection_identity_sha256",
    )
    if claimed != _self_hash(selection, "balanced_selection_identity_sha256"):
        raise ProjectionError("balanced selection self-hash mismatch")
    rows = selection.get("records")
    if not isinstance(rows, list) or not rows:
        raise ProjectionError("balanced selection has no records")
    selected: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _SELECTION_ROW_FIELDS:
            raise ProjectionError(f"balanced selection record[{index}] fields are not closed-world")
        record_id = _require_text(row.get("record_id"), f"records[{index}].record_id")
        if record_id in selected:
            raise ProjectionError(f"duplicate balanced selection record_id: {record_id}")
        selected[record_id] = row

    physical = {row["record_id"]: row for row in _parse_survivor_records(survivor_records_raw)}
    if not set(selected) <= set(physical):
        raise ProjectionError("balanced selection references record absent from survivor JSONL")

    projected: list[dict[str, Any]] = []
    for record_id in sorted(selected):
        authority = selected[record_id]
        raw = physical[record_id]
        exact = {
            "source_id": raw["source_id"],
            "family": raw["family"],
            "stratum": raw["stratum"],
            "modality": raw["modality"],
            "payload_sha256": raw["payload_sha256"],
            "payload_bytes": raw["payload_bytes"],
            "near_duplicate_cluster_id": raw["source_id"],
            "purpose": "pretraining_eligible",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        for field, expected in exact.items():
            if type(authority.get(field)) is not type(expected) or authority.get(field) != expected:
                raise ProjectionError(f"balanced selection/raw survivor drift: {record_id}:{field}")
        projected.append(
            {
                "record_id": record_id,
                "source_id": raw["source_id"],
                "family": raw["family"],
                "stratum": raw["stratum"],
                "modality": raw["modality"],
                "near_duplicate_cluster_id": raw["source_id"],
                "normalized_payload": raw["normalized_payload"],
                "purpose": "pretraining_eligible",
                "training_eligible": False,
                "evaluation_eligible": False,
                "evaluation_reserved": False,
            }
        )
    return projected
