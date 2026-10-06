#!/usr/bin/env python3
"""Compose exact #2792 balance with exact post-QP current-NBU survivor authority.

Execution-only bridge. It authenticates the current balanced text-free inventory,
authenticates current NBU post-DATA-232/G05/G06 text-free evidence plus the pinned
NBU intake contract, proves exact record/source/payload collision freedom, and
re-runs the unchanged NEXT100-106 balance/family-cap gate.

No corpus, tokenizer-fit, training, final-test, paid-compute, or scale authority
is created by this runner.
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

BASE_HEAD = "0c7423e3fe9d6bac0c1ba91294d0e6830490e820"
BASE_COMPOSITION_ID = "44a436c831ce1357c79badd4b1ba050e6f5ca09e750ff00aee2d1df70bfa88a6"
BASE_RECEIPT_ID = "359221033ce9fdf850b84ea98d84b1d5fbff47cf91d0d8e678b7a11be534d39f"
BASE_DEDUP_ID = "f79430e4c250a7f9f7b92e87d66f8b227dbb571e70a53d1ccafb51c12fc90bc2"
NBU_HEAD = "dc4822ec1365494507bca1d67bab1b334a3fa143"
NBU_EVIDENCE_ID = "0345668ea4f46896905e88186a79f0093c5d6e17c5a349d04ef0d0b9069a78da"
NBU_FAMILY = "ua.nbu.official-resolutions"
NBU_INTAKE_HEAD = "86a8a2a6f91eb4093bee414aaa8e8bbd3d895761"
NBU_INTAKE_BLOB = "b883cbdb489058e005e83f64ce11a67505e03778"
NBU_INTAKE_ID = "2e04335a2e6e665167b63f392d09756ae52e4f65f5253d9a037c48f5d191d9ec"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"

EXPECTED_BASE_RECORDS = 2711
EXPECTED_BASE_SOURCES = 2630
EXPECTED_BASE_BYTES = 19_769_910
EXPECTED_NBU_RECORDS = 40
EXPECTED_NBU_BYTES = 891_322
EXPECTED_COMBINED_RECORDS = 2751
EXPECTED_COMBINED_SOURCES = 2670
EXPECTED_COMBINED_BYTES = 20_661_232
EXPECTED_CAPACITY = {"ua": 1_096_637, "en": 15_481_867, "code": 4_082_728}
EXPECTED_FAMILY_COUNT = {"ua": 4, "en": 7, "code": 18}
EXPECTED_GAPS = {"ua": 7_903_363, "en": 0, "code": 0}
EXPECTED_MAX = 1_026_500
EXPECTED_MAX_STRATA = {"ua": 461_925, "en": 359_275, "code": 205_300}

HEX = frozenset("0123456789abcdef")
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
COMPOSITION_SCHEMA = "12-6.d03-current-balance-nbu-postqp-composition.v1"
DEDUP_SCHEMA = "12-6.d03-current-balance-nbu-postqp-dedup-proof.v1"
RECEIPT_SCHEMA = "12-6.d03-current-balance-nbu-postqp-balance-execution.v1"
MAX_INPUT_BYTES = 8 * 1024 * 1024


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


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


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
                type(observed) is type(expected_value) and observed == expected_value,
                f"{label} widened truth boundary: {field}",
            )


def verify_inventory(
    inventory: Mapping[str, Any],
    *,
    label: str,
    expected_records: int,
    expected_bytes: int,
) -> list[dict[str, Any]]:
    require(inventory.get("schema_version") == INVENTORY_SCHEMA, f"{label} schema drift")
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
        lowercase_sha(row["payload_sha256"], f"{label}[{index}].payload_sha256")
        positive(row["payload_bytes"], f"{label}[{index}].payload_bytes")
        require(row["record_id"] not in seen_records, f"{label} duplicate record_id")
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
        inventory.get("payload_inventory_digest_sha256") == sha256(payload_projection),
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
        == "12-6.d03-current-code-franko-ubuntu-pep-loc-composition.v1",
        "base composition schema drift",
    )
    require(composition.get("source_git_sha") == BASE_HEAD, "base composition head drift")
    require(
        composition.get("composition_identity_sha256") == BASE_COMPOSITION_ID
        and self_hash(composition, "composition_identity_sha256")
        == BASE_COMPOSITION_ID,
        "base composition identity drift",
    )
    require(
        composition.get("combined_dedup_proof_identity_sha256") == BASE_DEDUP_ID,
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
        receipt.get("schema")
        == "12-6.d03-current-code-franko-ubuntu-pep-loc-balance-execution.v1",
        "base receipt schema drift",
    )
    require(receipt.get("execution_head_sha") == BASE_HEAD, "base receipt head drift")
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
        receipt.get("balance_policy_identity_sha256") == POLICY_ID,
        "base policy identity drift",
    )
    require(
        receipt.get("combined_record_count") == EXPECTED_BASE_RECORDS
        and receipt.get("combined_source_object_count") == EXPECTED_BASE_SOURCES
        and receipt.get("combined_total_payload_bytes") == EXPECTED_BASE_BYTES,
        "base receipt totals drift",
    )
    zero_truth(receipt, "base receipt")

    require(vector.get("schema_version") == gate.INPUT_SCHEMA, "base vector schema drift")
    require(vector.get("terminal") is True, "base vector not terminal")
    families = vector.get("families")
    require(isinstance(families, list) and families, "base families missing")
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
            isinstance(family["family_id"], str) and family["family_id"],
            f"base family[{index}] id invalid",
        )
        require(family["family_id"] not in seen_family, "base family duplicate")
        seen_family.add(family["family_id"])
        require(family["stratum"] in {"ua", "en", "code"}, "base family stratum drift")
        positive(family["unique_bytes"], f"base family[{index}].unique_bytes")
        family_rows.append(family)

    by_family: dict[str, int] = defaultdict(int)
    by_family_modality: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_family[row["family"]] += int(row["payload_bytes"])
        by_family_modality[row["family"]].add(row["modality"])
    require(set(by_family) == seen_family, "base vector/inventory family set drift")
    for family in family_rows:
        require(
            family["unique_bytes"] == by_family[family["family_id"]],
            f"base vector/inventory family bytes drift: {family['family_id']}",
        )
        expected_stratum = family["stratum"]
        modalities = by_family_modality[family["family_id"]]
        if expected_stratum == "ua":
            require(modalities == {"uk"}, "base UA modality drift")
        else:
            require(modalities == {expected_stratum}, "base family modality drift")

    gate.validate_vector(dict(vector))
    return rows, sorted(family_rows, key=lambda row: row["family_id"])


def verify_nbu(
    evidence: Mapping[str, Any],
    family_config: Mapping[str, Any],
    family_config_raw: bytes,
) -> list[dict[str, Any]]:
    require(
        evidence.get("schema_version")
        == "12-6.d03-nbu-current40-decontam-g05-g06-execution.v1",
        "NBU evidence schema drift",
    )
    require(evidence.get("execution_head_sha") == NBU_HEAD, "NBU execution head drift")
    require(
        evidence.get("evidence_identity_sha256") == NBU_EVIDENCE_ID
        and self_hash(evidence, "evidence_identity_sha256") == NBU_EVIDENCE_ID,
        "NBU evidence identity drift",
    )
    require(
        evidence.get("execution_profile") == "LOCAL_FREE",
        "NBU execution profile drift",
    )
    content_boundary = evidence.get("content_boundary")
    require(
        content_boundary
        == {
            "durable_output_text_free": True,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "raw_training_text_persisted": False,
        },
        "NBU content boundary drift",
    )
    truth = evidence.get("truth_boundary")
    require(isinstance(truth, Mapping), "NBU truth boundary missing")
    require(
        truth.get("global_cross_source_dedup_complete_for_nbu_parent") is True
        and truth.get("reserved_evaluation_decontamination_complete_for_nbu") is True
        and truth.get("canonical_quality_privacy_complete_for_nbu") is True,
        "NBU prerequisite gates are incomplete",
    )
    require(
        truth.get("cross_lineage_rededup_after_post_qp_complete") is False
        and truth.get("balance_diversity_retest_complete") is False
        and truth.get("family_caps_complete") is False,
        "NBU evidence unexpectedly pre-claims this carrier's gates",
    )
    zero_truth(truth, "NBU truth")

    nbu = evidence.get("nbu_authority")
    require(isinstance(nbu, Mapping), "NBU authority missing")
    require(nbu.get("source_family") == NBU_FAMILY, "NBU family drift")
    require(nbu.get("source_object_count") == EXPECTED_NBU_RECORDS, "NBU source count drift")
    gate_execution = evidence.get("gate_execution")
    require(isinstance(gate_execution, Mapping), "NBU gate execution missing")
    require(
        gate_execution.get("survivor_records") == EXPECTED_NBU_RECORDS
        and gate_execution.get("survivor_source_objects") == EXPECTED_NBU_RECORDS
        and gate_execution.get("survivor_payload_bytes") == EXPECTED_NBU_BYTES
        and gate_execution.get("later_gate_loss_bytes") == 172,
        "NBU survivor result drift",
    )
    rows = verify_inventory(
        evidence["survivor_inventory"],
        label="NBU survivor inventory",
        expected_records=EXPECTED_NBU_RECORDS,
        expected_bytes=EXPECTED_NBU_BYTES,
    )
    require(
        {row["family"] for row in rows} == {NBU_FAMILY},
        "NBU survivor family drift",
    )
    require({row["modality"] for row in rows} == {"uk"}, "NBU survivor modality drift")
    require(
        len({row["source_id"] for row in rows}) == EXPECTED_NBU_RECORDS,
        "NBU source IDs are not unique",
    )

    require(
        git_blob_sha1(family_config_raw) == NBU_INTAKE_BLOB,
        "NBU intake config blob drift",
    )
    require(
        family_config.get("schema")
        == "12-6.d03-ua-nbu-official-resolutions-intake.v1",
        "NBU intake schema drift",
    )
    require(
        family_config.get("status") == "PREPARED_DISCOVERY_ZERO_CREDIT",
        "NBU intake status drift",
    )
    intake_core = dict(family_config)
    intake_core.pop("contract_identity_sha256", None)
    intake_identity = hashlib.sha256(canonical(intake_core) + b"\n").hexdigest()
    require(
        family_config.get("contract_identity_sha256") == NBU_INTAKE_ID
        and intake_identity == NBU_INTAKE_ID,
        "NBU intake contract identity drift",
    )
    source = family_config.get("source")
    require(isinstance(source, Mapping), "NBU intake source missing")
    require(
        source.get("family_id") == NBU_FAMILY
        and source.get("source_id") == NBU_FAMILY
        and source.get("stratum") == "uk"
        and source.get("one_conservative_family") is True,
        "NBU intake family binding drift",
    )
    claims = family_config.get("claims")
    require(isinstance(claims, Mapping), "NBU intake claims missing")
    require(
        claims.get("canonical_capacity_credit_bytes") == 0
        and claims.get("training_authorized_bytes") == 0
        and claims.get("authorized_unique_loss_positions") == 0
        and claims.get("tokenizer_fit_authorized") is False
        and claims.get("model_training_executed") is False
        and claims.get("optimizer_updates") == 0
        and claims.get("final_test_accessed") is False
        and claims.get("paid_compute_authorized") is False,
        "NBU intake claims widened",
    )
    return rows


def execute(
    *,
    base_composition: Mapping[str, Any],
    base_vector: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    nbu_evidence: Mapping[str, Any],
    nbu_family_config: Mapping[str, Any],
    nbu_family_config_raw: bytes,
    execution_head: str,
) -> dict[str, Mapping[str, Any]]:
    lowercase_sha(execution_head, "execution_head", 40)
    base_rows, base_families = verify_base(base_composition, base_vector, base_receipt)
    nbu_rows = verify_nbu(nbu_evidence, nbu_family_config, nbu_family_config_raw)

    base_record_ids = {row["record_id"] for row in base_rows}
    base_source_ids = {row["source_id"] for row in base_rows}
    base_payloads = {row["payload_sha256"] for row in base_rows}
    require(
        len(base_payloads) == EXPECTED_BASE_RECORDS,
        "base exact payload hashes are not unique",
    )

    record_collisions = sorted(
        row["record_id"] for row in nbu_rows if row["record_id"] in base_record_ids
    )
    source_collisions = sorted(
        row["source_id"] for row in nbu_rows if row["source_id"] in base_source_ids
    )
    payload_collisions = sorted(
        row["payload_sha256"] for row in nbu_rows if row["payload_sha256"] in base_payloads
    )
    require(not record_collisions, "NBU record IDs collide with current balance")
    require(not source_collisions, "NBU source IDs collide with current balance")
    require(not payload_collisions, "NBU payloads replay current balance")

    combined_rows = sorted(
        [dict(row) for row in base_rows] + [dict(row) for row in nbu_rows],
        key=lambda row: row["record_id"],
    )
    require(len(combined_rows) == EXPECTED_COMBINED_RECORDS, "combined record drift")
    require(
        len({row["record_id"] for row in combined_rows}) == EXPECTED_COMBINED_RECORDS,
        "combined record IDs not unique",
    )
    require(
        len({row["source_id"] for row in combined_rows}) == EXPECTED_COMBINED_SOURCES,
        "combined source IDs not unique",
    )
    require(
        len({row["payload_sha256"] for row in combined_rows}) == EXPECTED_COMBINED_RECORDS,
        "combined payload hashes not unique",
    )
    require(
        sum(int(row["payload_bytes"]) for row in combined_rows) == EXPECTED_COMBINED_BYTES,
        "combined payload-byte arithmetic drift",
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
        "record_count": EXPECTED_COMBINED_RECORDS,
        "total_payload_bytes": EXPECTED_COMBINED_BYTES,
        "records": combined_rows,
        "record_inventory_digest_sha256": sha256(combined_rows),
        "payload_inventory_digest_sha256": sha256(payload_projection),
    }

    nbu_family_authority_core = {
        "schema": "12-6.d03-nbu-current40-family-extension.v1",
        "intake_authority_git_sha": NBU_INTAKE_HEAD,
        "intake_authority_git_blob_sha1": NBU_INTAKE_BLOB,
        "intake_contract_identity_sha256": NBU_INTAKE_ID,
        "family": NBU_FAMILY,
        "stratum": "uk",
        "source_objects": EXPECTED_NBU_RECORDS,
        "post_qp_unique_payload_bytes": EXPECTED_NBU_BYTES,
        "nbu_evidence_identity_sha256": NBU_EVIDENCE_ID,
        "zero_credit": True,
    }
    nbu_family_authority = {
        **nbu_family_authority_core,
        "authority_identity_sha256": sha256(nbu_family_authority_core),
    }

    dedup_core = {
        "schema": DEDUP_SCHEMA,
        "base_head_sha": BASE_HEAD,
        "base_composition_identity_sha256": BASE_COMPOSITION_ID,
        "base_dedup_identity_sha256": BASE_DEDUP_ID,
        "nbu_head_sha": NBU_HEAD,
        "nbu_evidence_identity_sha256": NBU_EVIDENCE_ID,
        "nbu_family_authority_identity_sha256": nbu_family_authority[
            "authority_identity_sha256"
        ],
        "base_records": EXPECTED_BASE_RECORDS,
        "base_source_objects": EXPECTED_BASE_SOURCES,
        "base_payload_bytes": EXPECTED_BASE_BYTES,
        "nbu_records": EXPECTED_NBU_RECORDS,
        "nbu_source_objects": EXPECTED_NBU_RECORDS,
        "nbu_payload_bytes": EXPECTED_NBU_BYTES,
        "discounted_payload_replay_records": 0,
        "discounted_payload_replay_bytes": 0,
        "novel_nbu_records": EXPECTED_NBU_RECORDS,
        "novel_nbu_source_objects": EXPECTED_NBU_RECORDS,
        "novel_nbu_payload_bytes": EXPECTED_NBU_BYTES,
        "cross_inventory_record_id_collision_free": True,
        "cross_inventory_source_id_collision_free": True,
        "cross_inventory_exact_payload_collision_free": True,
        "combined_records": EXPECTED_COMBINED_RECORDS,
        "combined_source_objects": EXPECTED_COMBINED_SOURCES,
        "combined_payload_bytes": EXPECTED_COMBINED_BYTES,
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
        all(row["family_id"] != NBU_FAMILY for row in families),
        "NBU family already exists in base vector",
    )
    families.append(
        {
            "family_id": NBU_FAMILY,
            "stratum": "ua",
            "unique_bytes": EXPECTED_NBU_BYTES,
        }
    )
    families.sort(key=lambda row: row["family_id"])
    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-CURRENT-BALANCE-NBU-POSTQP-UNION-V1",
            "head_sha": execution_head,
            "evidence_identity_sha256": dedup_proof["evidence_identity_sha256"],
            "terminal_verdict": "PASS",
        },
        "families": families,
        "totals": {
            "total_unique_bytes": EXPECTED_COMBINED_BYTES,
            "by_stratum": EXPECTED_CAPACITY,
            "family_count": EXPECTED_FAMILY_COUNT,
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
        balance.get("status") == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA",
        "balance status drift",
    )
    require(
        balance.get("maximum_feasible_total_source_bytes") == EXPECTED_MAX,
        "maximum feasible mix drift",
    )
    require(
        balance.get("maximum_feasible_stratum_bytes") == EXPECTED_MAX_STRATA,
        "maximum feasible stratum mix drift",
    )
    require(
        balance.get("raw_capacity_by_stratum") == EXPECTED_CAPACITY,
        "raw capacity drift",
    )
    require(
        balance.get("raw_gap_to_target_by_stratum") == EXPECTED_GAPS,
        "raw target gap drift",
    )
    require(
        balance.get("family_minimum")
        == {
            "required_per_stratum": 2,
            "observed": EXPECTED_FAMILY_COUNT,
            "pass": True,
        },
        "family minimum drift",
    )

    composition_core = {
        "schema": COMPOSITION_SCHEMA,
        "status": "COMPOSED_ZERO_CREDIT_PENDING_LATER_GATES",
        "source_git_sha": execution_head,
        "parents": {
            "base_balance_head_sha": BASE_HEAD,
            "base_composition_identity_sha256": BASE_COMPOSITION_ID,
            "base_receipt_identity_sha256": BASE_RECEIPT_ID,
            "nbu_post_qp_head_sha": NBU_HEAD,
            "nbu_evidence_identity_sha256": NBU_EVIDENCE_ID,
        },
        "nbu_family_authority_identity_sha256": nbu_family_authority[
            "authority_identity_sha256"
        ],
        "combined_dedup_proof_identity_sha256": dedup_proof[
            "evidence_identity_sha256"
        ],
        "combined_inventory": combined_inventory,
        "stratum_capacity_bytes": {
            "uk": EXPECTED_CAPACITY["ua"],
            "en": EXPECTED_CAPACITY["en"],
            "code": EXPECTED_CAPACITY["code"],
        },
        "stratum_family_counts": {
            "uk": EXPECTED_FAMILY_COUNT["ua"],
            "en": EXPECTED_FAMILY_COUNT["en"],
            "code": EXPECTED_FAMILY_COUNT["code"],
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
        "nbu_evidence_identity_sha256": NBU_EVIDENCE_ID,
        "nbu_family_authority_identity_sha256": nbu_family_authority[
            "authority_identity_sha256"
        ],
        "combined_dedup_proof_identity_sha256": dedup_proof[
            "evidence_identity_sha256"
        ],
        "composition_identity_sha256": composition["composition_identity_sha256"],
        "balance_policy_identity_sha256": POLICY_ID,
        "balance_result_identity_sha256": balance["result_identity_sha256"],
        "discounted_exact_replay_records": 0,
        "discounted_exact_replay_payload_bytes": 0,
        "novel_nbu_records": EXPECTED_NBU_RECORDS,
        "novel_nbu_source_objects": EXPECTED_NBU_RECORDS,
        "novel_nbu_payload_bytes": EXPECTED_NBU_BYTES,
        "combined_record_count": EXPECTED_COMBINED_RECORDS,
        "combined_source_object_count": EXPECTED_COMBINED_SOURCES,
        "combined_total_payload_bytes": EXPECTED_COMBINED_BYTES,
        "raw_capacity_by_stratum": EXPECTED_CAPACITY,
        "family_minimum": balance["family_minimum"],
        "maximum_feasible_total_source_bytes": EXPECTED_MAX,
        "maximum_feasible_stratum_bytes": EXPECTED_MAX_STRATA,
        "raw_gap_to_target_by_stratum": EXPECTED_GAPS,
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
    receipt = {**receipt_core, "receipt_identity_sha256": sha256(receipt_core)}
    return {
        "nbu-family-authority": nbu_family_authority,
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
    parser.add_argument("--nbu-evidence-json", type=Path, required=True)
    parser.add_argument("--nbu-family-config-json", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        raw_config = args.nbu_family_config_json.read_bytes()
        result = execute(
            base_composition=load(args.base_composition_json),
            base_vector=load(args.base_next100_json),
            base_receipt=load(args.base_receipt_json),
            nbu_evidence=load(args.nbu_evidence_json),
            nbu_family_config=load(args.nbu_family_config_json),
            nbu_family_config_raw=raw_config,
            execution_head=args.execution_head,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            write_json(args.output_dir / f"{name}.json", value)
    except (ExecutionError, gate.GateError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    receipt = result["execution-receipt"]
    print("D03_NBU_COMBINED_BALANCE=" + str(receipt["balance_status"]))
    print("NOVEL_NBU_PAYLOAD_BYTES=" + str(receipt["novel_nbu_payload_bytes"]))
    print("COMBINED_TOTAL_PAYLOAD_BYTES=" + str(receipt["combined_total_payload_bytes"]))
    print("MAX_FEASIBLE_TOTAL_SOURCE_BYTES=" + str(receipt["maximum_feasible_total_source_bytes"]))
    print("UA_RAW_GAP_BYTES=" + str(receipt["raw_gap_to_target_by_stratum"]["ua"]))
    print("RECEIPT_IDENTITY_SHA256=" + str(receipt["receipt_identity_sha256"]))
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
