#!/usr/bin/env python3
"""Compose exact current balance with exact post-QP Lesia survivors.

Execution-only bridge. It authenticates the terminal current balance and the
terminal Lesia DATA-232/G05/G06 authority, applies the incumbent exact
cross-inventory no-replay rule used by the current balance lineage, and reruns
the unchanged NEXT100-106 balance/family-cap gate.

This runner grants no corpus, tokenizer-fit, training, final-test, paid-compute,
or scale authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.next100_106_balance_gate as gate

BASE_HEAD = "64a9cb6b8decca7b045d2cc6cad07e80b0655cec"
BASE_COMPOSITION_ID = (
    "6a0860bc85cb7f2bf81580250c69ced7c15c63c40a622070fa9f2655b97e7f74"
)
BASE_RECEIPT_ID = (
    "76dc985be9b0debbea9f5691051122fd36ebba3325f600c9a737e99a023d5355"
)
BASE_DEDUP_ID = (
    "dfadc4165b6406f1b1a6981e71b7e29a7d16c28726e4b7f48963b7c978de2d6a"
)
LESIA_HEAD = "1d594e048365fd2121433210b83c0dc9069c50bf"
LESIA_EVIDENCE_ID = (
    "dddddf9b99e4050d6b8483a8b18822a1453be5511772668315c800b947c17b8d"
)
LESIA_FAMILY = "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"

EXPECTED_BASE_RECORDS = 2_751
EXPECTED_BASE_SOURCES = 2_670
EXPECTED_BASE_BYTES = 20_661_232
EXPECTED_BASE_CAPACITY = {
    "ua": 1_096_637,
    "en": 15_481_867,
    "code": 4_082_728,
}
EXPECTED_BASE_FAMILY_COUNT = {"ua": 4, "en": 7, "code": 18}
EXPECTED_LESIA_RECORDS = 105
EXPECTED_LESIA_SOURCES = 105
EXPECTED_LESIA_BYTES = 159_722
EXPECTED_LESIA_PRE_GATE_BYTES = 169_399
EXPECTED_LESIA_PARENT_MARGINAL_BYTES = 167_920
EXPECTED_LESIA_GATE_LOSS_BYTES = 9_677
UA_TARGET_BYTES = 9_000_000

HEX = frozenset("0123456789abcdef")
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
COMPOSITION_SCHEMA = "12-6.d03-current-balance-lesia-postqp-composition.v1"
DEDUP_SCHEMA = "12-6.d03-current-balance-lesia-postqp-dedup-proof.v1"
RECEIPT_SCHEMA = "12-6.d03-current-balance-lesia-postqp-balance-execution.v1"
MAX_INPUT_BYTES = 16 * 1024 * 1024


class ExecutionError(ValueError):
    """Raised when an input or derived authority is not exact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionError(message)


def reject_constant(value: str) -> None:
    raise ExecutionError(f"non-finite JSON constant: {value}")


def parse_float(value: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), "non-finite JSON number")
    return parsed


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ExecutionError(f"cannot read {path}") from exc
    require(len(raw) <= MAX_INPUT_BYTES, f"{path} exceeds input bound")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExecutionError(f"invalid strict JSON: {path}") from exc
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ExecutionError("value is not canonical strict UTF-8 JSON") from exc


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def self_hash(document: Mapping[str, Any], field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    return sha256(body)


def lowercase_sha(value: object, field: str, length: int = 64) -> str:
    require(
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and set(value) <= HEX,
        f"{field} must be {length} lowercase hex characters",
    )
    return str(value)


def positive(value: object, field: str) -> int:
    require(type(value) is int and value > 0, f"{field} must be positive integer")
    return int(value)


def zero_truth(document: Mapping[str, Any], label: str) -> None:
    expected = {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    for field, expected_value in expected.items():
        if field in document:
            observed = document[field]
            require(
                type(observed) is type(expected_value)
                and observed == expected_value,
                f"{label} widened truth boundary: {field}",
            )


def verify_inventory(
    inventory: Mapping[str, Any],
    *,
    label: str,
    expected_records: int,
    expected_bytes: int,
) -> list[dict[str, Any]]:
    require(
        inventory.get("schema_version") == INVENTORY_SCHEMA,
        f"{label} schema drift",
    )
    rows = inventory.get("records")
    require(isinstance(rows, list), f"{label} rows missing")
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
        require(
            isinstance(raw, Mapping) and set(raw) == required,
            f"{label}[{index}] schema drift",
        )
        row = dict(raw)
        for field in ("record_id", "source_id", "family", "modality"):
            require(
                isinstance(row[field], str) and bool(row[field]),
                f"{label}[{index}].{field} invalid",
            )
        lowercase_sha(
            row["payload_sha256"],
            f"{label}[{index}].payload_sha256",
        )
        positive(row["payload_bytes"], f"{label}[{index}].payload_bytes")
        require(
            row["record_id"] not in seen_records,
            f"{label} duplicate record_id",
        )
        seen_records.add(row["record_id"])
        normalized.append(row)

    normalized.sort(key=lambda row: row["record_id"])
    require(normalized == rows, f"{label} row order drift")
    require(len(rows) == expected_records, f"{label} record count drift")
    require(
        sum(int(row["payload_bytes"]) for row in rows) == expected_bytes,
        f"{label} byte count drift",
    )
    require(
        inventory.get("record_count") == expected_records,
        f"{label} declared record count drift",
    )
    require(
        inventory.get("total_payload_bytes") == expected_bytes,
        f"{label} declared byte count drift",
    )
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    require(
        inventory.get("record_inventory_digest_sha256") == sha256(rows),
        f"{label} record root drift",
    )
    require(
        inventory.get("payload_inventory_digest_sha256")
        == sha256(payload_projection),
        f"{label} payload root drift",
    )
    return normalized


def verify_base(
    composition: Mapping[str, Any],
    vector: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(
        composition.get("schema")
        == "12-6.d03-current-balance-nbu-postqp-composition.v1",
        "base composition schema drift",
    )
    require(
        composition.get("source_git_sha") == BASE_HEAD,
        "base composition head drift",
    )
    require(
        composition.get("composition_identity_sha256") == BASE_COMPOSITION_ID
        and self_hash(composition, "composition_identity_sha256")
        == BASE_COMPOSITION_ID,
        "base composition identity drift",
    )
    require(
        composition.get("combined_dedup_proof_identity_sha256")
        == BASE_DEDUP_ID,
        "base dedup identity drift",
    )
    zero_truth(composition, "base composition")
    rows = verify_inventory(
        composition["combined_inventory"],
        label="base inventory",
        expected_records=EXPECTED_BASE_RECORDS,
        expected_bytes=EXPECTED_BASE_BYTES,
    )
    require(
        len({row["source_id"] for row in rows}) == EXPECTED_BASE_SOURCES,
        "base source count drift",
    )
    require(
        len({row["payload_sha256"] for row in rows})
        == EXPECTED_BASE_RECORDS,
        "base exact-payload uniqueness drift",
    )

    require(
        receipt.get("schema")
        == "12-6.d03-current-balance-nbu-postqp-balance-execution.v1",
        "base receipt schema drift",
    )
    require(
        receipt.get("execution_head_sha") == BASE_HEAD,
        "base receipt head drift",
    )
    require(
        receipt.get("receipt_identity_sha256") == BASE_RECEIPT_ID
        and self_hash(receipt, "receipt_identity_sha256") == BASE_RECEIPT_ID,
        "base receipt identity drift",
    )
    require(
        receipt.get("composition_identity_sha256") == BASE_COMPOSITION_ID,
        "base receipt composition drift",
    )
    require(
        receipt.get("combined_dedup_proof_identity_sha256")
        == BASE_DEDUP_ID,
        "base receipt dedup drift",
    )
    require(
        receipt.get("balance_policy_identity_sha256") == POLICY_ID,
        "base policy identity drift",
    )
    require(
        receipt.get("combined_record_count") == EXPECTED_BASE_RECORDS
        and receipt.get("combined_source_object_count")
        == EXPECTED_BASE_SOURCES
        and receipt.get("combined_total_payload_bytes")
        == EXPECTED_BASE_BYTES,
        "base receipt totals drift",
    )
    require(
        receipt.get("raw_capacity_by_stratum") == EXPECTED_BASE_CAPACITY,
        "base raw capacity drift",
    )
    zero_truth(receipt, "base receipt")

    require(
        vector.get("schema_version") == gate.INPUT_SCHEMA,
        "base vector schema drift",
    )
    require(vector.get("terminal") is True, "base vector not terminal")
    families = vector.get("families")
    require(
        isinstance(families, list) and families,
        "base families missing",
    )
    family_rows: list[dict[str, Any]] = []
    seen_family: set[str] = set()
    for index, raw in enumerate(families):
        require(
            isinstance(raw, Mapping)
            and set(raw) == {"family_id", "stratum", "unique_bytes"},
            f"base family[{index}] schema drift",
        )
        family = dict(raw)
        require(
            isinstance(family["family_id"], str)
            and bool(family["family_id"]),
            f"base family[{index}] id invalid",
        )
        require(
            family["family_id"] not in seen_family,
            "base family duplicate",
        )
        seen_family.add(family["family_id"])
        require(
            family["stratum"] in {"ua", "en", "code"},
            "base family stratum drift",
        )
        positive(
            family["unique_bytes"],
            f"base family[{index}].unique_bytes",
        )
        family_rows.append(family)

    by_family: dict[str, int] = defaultdict(int)
    modalities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_family[row["family"]] += int(row["payload_bytes"])
        modalities[row["family"]].add(row["modality"])
    require(
        set(by_family) == seen_family,
        "base vector/inventory family set drift",
    )
    for family in family_rows:
        family_id = family["family_id"]
        require(
            family["unique_bytes"] == by_family[family_id],
            f"base family bytes drift: {family_id}",
        )
        expected_modality = (
            "uk" if family["stratum"] == "ua" else family["stratum"]
        )
        require(
            modalities[family_id] == {expected_modality},
            f"base family modality drift: {family_id}",
        )

    totals = vector.get("totals")
    require(isinstance(totals, Mapping), "base totals missing")
    require(
        totals.get("total_unique_bytes") == EXPECTED_BASE_BYTES
        and totals.get("by_stratum") == EXPECTED_BASE_CAPACITY
        and totals.get("family_count") == EXPECTED_BASE_FAMILY_COUNT,
        "base vector totals drift",
    )
    gate.validate_vector(dict(vector))
    return rows, sorted(
        family_rows,
        key=lambda row: row["family_id"],
    )


def verify_lesia(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(
        evidence.get("schema_version")
        == "12-6.d03-wikisource-lesia1892-decontam-g05-g06-execution.v1",
        "Lesia evidence schema drift",
    )
    require(
        evidence.get("execution_head_sha") == LESIA_HEAD,
        "Lesia execution head drift",
    )
    require(
        evidence.get("evidence_identity_sha256") == LESIA_EVIDENCE_ID
        and self_hash(evidence, "evidence_identity_sha256")
        == LESIA_EVIDENCE_ID,
        "Lesia evidence identity drift",
    )
    require(
        evidence.get("execution_profile") == "LOCAL_FREE",
        "Lesia execution profile drift",
    )
    boundary = evidence.get("content_boundary")
    require(
        boundary
        == {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "Lesia content boundary drift",
    )

    truth = evidence.get("truth_boundary")
    require(isinstance(truth, Mapping), "Lesia truth boundary missing")
    require(
        truth.get("global_cross_source_dedup_complete_for_lesia_parent")
        is True
        and truth.get("reserved_evaluation_decontamination_complete_for_lesia")
        is True
        and truth.get("canonical_quality_privacy_complete_for_lesia")
        is True,
        "Lesia prerequisite gates incomplete",
    )
    require(
        truth.get("cross_lineage_rededup_after_post_qp_complete") is False
        and truth.get("balance_diversity_retest_complete") is False
        and truth.get("family_caps_complete") is False,
        "Lesia pre-claims this carrier",
    )
    zero_truth(truth, "Lesia truth")

    authority = evidence.get("lesia_authority")
    require(isinstance(authority, Mapping), "Lesia authority missing")
    require(
        authority.get("source_family") == LESIA_FAMILY,
        "Lesia source family drift",
    )
    require(
        authority.get("source_object_count") == 107
        and authority.get("pre_gate_payload_bytes")
        == EXPECTED_LESIA_PRE_GATE_BYTES,
        "Lesia source authority drift",
    )

    execution = evidence.get("gate_execution")
    require(isinstance(execution, Mapping), "Lesia gate execution missing")
    require(
        execution.get("input_records") == 107
        and execution.get("survivor_records") == EXPECTED_LESIA_RECORDS
        and execution.get("survivor_source_objects")
        == EXPECTED_LESIA_SOURCES
        and execution.get("survivor_payload_bytes")
        == EXPECTED_LESIA_BYTES
        and execution.get("later_gate_loss_bytes")
        == EXPECTED_LESIA_GATE_LOSS_BYTES,
        "Lesia post-QP result drift",
    )

    capacity = evidence.get("capacity_observation")
    require(
        isinstance(capacity, Mapping),
        "Lesia capacity observation missing",
    )
    require(
        capacity.get("parent_marginal_global_unique_capacity_bytes")
        == EXPECTED_LESIA_PARENT_MARGINAL_BYTES
        and capacity.get("qualified_post_qp_survivor_payload_bytes")
        == EXPECTED_LESIA_BYTES
        and capacity.get("safe_incremental_unique_upper_bound_bytes")
        == EXPECTED_LESIA_BYTES
        and capacity.get("exact_incremental_unique_capacity_credited_bytes")
        == 0
        and capacity.get("exact_cross_lineage_rededup_required") is True
        and capacity.get("balance_retest_authorized") is False,
        "Lesia capacity boundary drift",
    )

    rows = verify_inventory(
        evidence["survivor_inventory"],
        label="Lesia survivor inventory",
        expected_records=EXPECTED_LESIA_RECORDS,
        expected_bytes=EXPECTED_LESIA_BYTES,
    )
    require(
        {row["family"] for row in rows} == {LESIA_FAMILY},
        "Lesia survivor family drift",
    )
    require(
        {row["modality"] for row in rows} == {"uk"},
        "Lesia survivor modality drift",
    )
    require(
        len({row["source_id"] for row in rows})
        == EXPECTED_LESIA_SOURCES,
        "Lesia source IDs are not unique",
    )
    require(
        len({row["payload_sha256"] for row in rows})
        == EXPECTED_LESIA_RECORDS,
        "Lesia exact payload hashes are not unique",
    )
    return rows


def build_result(
    *,
    base_composition: Mapping[str, Any],
    base_vector: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    lesia_evidence: Mapping[str, Any],
    execution_head: str,
) -> dict[str, Mapping[str, Any]]:
    lowercase_sha(execution_head, "execution_head", 40)
    base_rows, base_families = verify_base(
        base_composition,
        base_vector,
        base_receipt,
    )
    lesia_rows = verify_lesia(lesia_evidence)

    base_record_ids = {row["record_id"] for row in base_rows}
    base_source_ids = {row["source_id"] for row in base_rows}
    base_payloads = {row["payload_sha256"] for row in base_rows}

    record_collisions = sorted(
        row["record_id"]
        for row in lesia_rows
        if row["record_id"] in base_record_ids
    )
    source_collisions = sorted(
        row["source_id"]
        for row in lesia_rows
        if row["source_id"] in base_source_ids
    )
    require(
        not record_collisions,
        "Lesia record IDs collide with current balance",
    )
    require(
        not source_collisions,
        "Lesia source IDs collide with current balance",
    )

    replay_rows = [
        row
        for row in lesia_rows
        if row["payload_sha256"] in base_payloads
    ]
    novel_rows = [
        dict(row)
        for row in lesia_rows
        if row["payload_sha256"] not in base_payloads
    ]
    replay_bytes = sum(int(row["payload_bytes"]) for row in replay_rows)
    novel_bytes = sum(int(row["payload_bytes"]) for row in novel_rows)
    require(novel_rows, "Lesia contributes zero novel rows")
    require(
        novel_bytes + replay_bytes == EXPECTED_LESIA_BYTES,
        "Lesia replay arithmetic drift",
    )
    novel_source_count = len({row["source_id"] for row in novel_rows})
    require(
        novel_source_count == len(novel_rows),
        "novel Lesia source IDs are not one-to-one",
    )

    combined_rows = sorted(
        [dict(row) for row in base_rows] + novel_rows,
        key=lambda row: row["record_id"],
    )
    combined_records = len(combined_rows)
    combined_sources = len({row["source_id"] for row in combined_rows})
    combined_bytes = sum(int(row["payload_bytes"]) for row in combined_rows)
    require(
        combined_records == EXPECTED_BASE_RECORDS + len(novel_rows),
        "combined record arithmetic drift",
    )
    require(
        combined_sources == EXPECTED_BASE_SOURCES + novel_source_count,
        "combined source arithmetic drift",
    )
    require(
        combined_bytes == EXPECTED_BASE_BYTES + novel_bytes,
        "combined byte arithmetic drift",
    )
    require(
        len({row["payload_sha256"] for row in combined_rows})
        == combined_records,
        "combined exact payload hashes are not unique",
    )

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
        "record_count": combined_records,
        "total_payload_bytes": combined_bytes,
        "records": combined_rows,
        "record_inventory_digest_sha256": sha256(combined_rows),
        "payload_inventory_digest_sha256": sha256(payload_projection),
    }

    family_authority_core = {
        "schema": "12-6.d03-lesia-postqp-family-extension.v1",
        "lesia_execution_head_sha": LESIA_HEAD,
        "lesia_evidence_identity_sha256": LESIA_EVIDENCE_ID,
        "family": LESIA_FAMILY,
        "stratum": "uk",
        "post_qp_records": EXPECTED_LESIA_RECORDS,
        "post_qp_source_objects": EXPECTED_LESIA_SOURCES,
        "post_qp_payload_bytes": EXPECTED_LESIA_BYTES,
        "discounted_exact_replay_records": len(replay_rows),
        "discounted_exact_replay_payload_bytes": replay_bytes,
        "novel_records": len(novel_rows),
        "novel_source_objects": novel_source_count,
        "novel_payload_bytes": novel_bytes,
        "zero_credit": True,
    }
    family_authority = {
        **family_authority_core,
        "authority_identity_sha256": sha256(family_authority_core),
    }

    dedup_core = {
        "schema": DEDUP_SCHEMA,
        "base_head_sha": BASE_HEAD,
        "base_composition_identity_sha256": BASE_COMPOSITION_ID,
        "base_dedup_identity_sha256": BASE_DEDUP_ID,
        "lesia_head_sha": LESIA_HEAD,
        "lesia_evidence_identity_sha256": LESIA_EVIDENCE_ID,
        "lesia_family_authority_identity_sha256": family_authority[
            "authority_identity_sha256"
        ],
        "base_records": EXPECTED_BASE_RECORDS,
        "base_source_objects": EXPECTED_BASE_SOURCES,
        "base_payload_bytes": EXPECTED_BASE_BYTES,
        "lesia_post_qp_records": EXPECTED_LESIA_RECORDS,
        "lesia_post_qp_source_objects": EXPECTED_LESIA_SOURCES,
        "lesia_post_qp_payload_bytes": EXPECTED_LESIA_BYTES,
        "discounted_exact_replay_records": len(replay_rows),
        "discounted_exact_replay_payload_bytes": replay_bytes,
        "novel_lesia_records": len(novel_rows),
        "novel_lesia_source_objects": novel_source_count,
        "novel_lesia_payload_bytes": novel_bytes,
        "cross_inventory_record_id_collision_free": True,
        "cross_inventory_source_id_collision_free": True,
        "cross_inventory_exact_payload_replay_discounted": True,
        "combined_records": combined_records,
        "combined_source_objects": combined_sources,
        "combined_payload_bytes": combined_bytes,
        "combined_record_inventory_digest_sha256": combined_inventory[
            "record_inventory_digest_sha256"
        ],
        "combined_payload_inventory_digest_sha256": combined_inventory[
            "payload_inventory_digest_sha256"
        ],
        "terminal_verdict": "PASS",
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
        "scale_promotion_authorized": False,
    }
    dedup_proof = {
        **dedup_core,
        "evidence_identity_sha256": sha256(dedup_core),
    }

    families = [dict(row) for row in base_families]
    require(
        all(row["family_id"] != LESIA_FAMILY for row in families),
        "Lesia family already exists in current balance",
    )
    families.append(
        {
            "family_id": LESIA_FAMILY,
            "stratum": "ua",
            "unique_bytes": novel_bytes,
        }
    )
    families.sort(key=lambda row: row["family_id"])

    capacity = dict(EXPECTED_BASE_CAPACITY)
    capacity["ua"] += novel_bytes
    family_count = dict(EXPECTED_BASE_FAMILY_COUNT)
    family_count["ua"] += 1
    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-CURRENT-BALANCE-LESIA-POSTQP-UNION-V1",
            "head_sha": execution_head,
            "evidence_identity_sha256": dedup_proof[
                "evidence_identity_sha256"
            ],
            "terminal_verdict": "PASS",
        },
        "families": families,
        "totals": {
            "total_unique_bytes": combined_bytes,
            "by_stratum": capacity,
            "family_count": family_count,
        },
    }

    policy = gate.load_json(gate.POLICY_PATH)
    gate.validate_policy(policy)
    require(
        policy.get("policy_identity_sha256") == POLICY_ID,
        "NEXT100 policy identity drift",
    )
    gate.validate_vector(vector)
    balance = gate.evaluate(policy, vector)
    require(
        balance.get("raw_capacity_by_stratum") == capacity,
        "raw capacity drift",
    )
    expected_gaps = {
        "ua": max(UA_TARGET_BYTES - capacity["ua"], 0),
        "en": 0,
        "code": 0,
    }
    require(
        balance.get("raw_gap_to_target_by_stratum") == expected_gaps,
        "raw target gap drift",
    )
    require(
        balance.get("family_minimum")
        == {
            "required_per_stratum": 2,
            "observed": family_count,
            "pass": True,
        },
        "family minimum drift",
    )
    require(
        balance.get("status") == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA",
        "balance status drift",
    )

    composition_core = {
        "schema": COMPOSITION_SCHEMA,
        "status": "COMPOSED_ZERO_CREDIT_PENDING_LATER_GATES",
        "source_git_sha": execution_head,
        "parents": {
            "base_balance_head_sha": BASE_HEAD,
            "base_composition_identity_sha256": BASE_COMPOSITION_ID,
            "base_receipt_identity_sha256": BASE_RECEIPT_ID,
            "lesia_post_qp_head_sha": LESIA_HEAD,
            "lesia_evidence_identity_sha256": LESIA_EVIDENCE_ID,
        },
        "lesia_family_authority_identity_sha256": family_authority[
            "authority_identity_sha256"
        ],
        "combined_dedup_proof_identity_sha256": dedup_proof[
            "evidence_identity_sha256"
        ],
        "combined_inventory": combined_inventory,
        "stratum_capacity_bytes": {
            "uk": capacity["ua"],
            "en": capacity["en"],
            "code": capacity["code"],
        },
        "stratum_family_counts": {
            "uk": family_count["ua"],
            "en": family_count["en"],
            "code": family_count["code"],
        },
        "next_gate": "ACQUIRE_MORE_DIVERSE_LAWFUL_UA_SOURCE_CAPACITY",
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
        "scale_promotion_authorized": False,
    }
    composition = {
        **composition_core,
        "composition_identity_sha256": sha256(composition_core),
    }

    receipt_core = {
        "schema": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "base_composition_identity_sha256": BASE_COMPOSITION_ID,
        "base_receipt_identity_sha256": BASE_RECEIPT_ID,
        "lesia_evidence_identity_sha256": LESIA_EVIDENCE_ID,
        "lesia_family_authority_identity_sha256": family_authority[
            "authority_identity_sha256"
        ],
        "combined_dedup_proof_identity_sha256": dedup_proof[
            "evidence_identity_sha256"
        ],
        "composition_identity_sha256": composition[
            "composition_identity_sha256"
        ],
        "balance_policy_identity_sha256": POLICY_ID,
        "balance_result_identity_sha256": balance[
            "result_identity_sha256"
        ],
        "discounted_exact_replay_records": len(replay_rows),
        "discounted_exact_replay_payload_bytes": replay_bytes,
        "novel_lesia_records": len(novel_rows),
        "novel_lesia_source_objects": novel_source_count,
        "novel_lesia_payload_bytes": novel_bytes,
        "combined_record_count": combined_records,
        "combined_source_object_count": combined_sources,
        "combined_total_payload_bytes": combined_bytes,
        "raw_capacity_by_stratum": capacity,
        "family_minimum": balance["family_minimum"],
        "maximum_feasible_total_source_bytes": balance[
            "maximum_feasible_total_source_bytes"
        ],
        "maximum_feasible_stratum_bytes": balance[
            "maximum_feasible_stratum_bytes"
        ],
        "raw_gap_to_target_by_stratum": expected_gaps,
        "balance_status": balance["status"],
        "next_scientific_gate": (
            "ACQUIRE_MORE_DIVERSE_LAWFUL_UA_SOURCE_CAPACITY"
        ),
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": sha256(receipt_core),
    }
    return {
        "lesia-family-authority": family_authority,
        "combined-dedup-proof": dedup_proof,
        "composition": composition,
        "next100-input": vector,
        "balance-result": balance,
        "execution-receipt": receipt,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical(value) + b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-composition-json", type=Path, required=True)
    parser.add_argument("--base-next100-json", type=Path, required=True)
    parser.add_argument("--base-receipt-json", type=Path, required=True)
    parser.add_argument("--lesia-evidence-json", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_result(
            base_composition=load(args.base_composition_json),
            base_vector=load(args.base_next100_json),
            base_receipt=load(args.base_receipt_json),
            lesia_evidence=load(args.lesia_evidence_json),
            execution_head=args.execution_head,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            write_json(args.output_dir / f"{name}.json", value)
    except (ExecutionError, gate.GateError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    receipt = result["execution-receipt"]
    print("D03_LESIA_CURRENT_BALANCE=" + str(receipt["balance_status"]))
    print(
        "NOVEL_LESIA_PAYLOAD_BYTES="
        + str(receipt["novel_lesia_payload_bytes"])
    )
    print(
        "DISCOUNTED_LESIA_REPLAY_BYTES="
        + str(receipt["discounted_exact_replay_payload_bytes"])
    )
    print(
        "COMBINED_TOTAL_PAYLOAD_BYTES="
        + str(receipt["combined_total_payload_bytes"])
    )
    print(
        "MAX_FEASIBLE_TOTAL_SOURCE_BYTES="
        + str(receipt["maximum_feasible_total_source_bytes"])
    )
    print(
        "UA_RAW_GAP_BYTES="
        + str(receipt["raw_gap_to_target_by_stratum"]["ua"])
    )
    print(
        "RECEIPT_IDENTITY_SHA256="
        + str(receipt["receipt_identity_sha256"])
    )
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
