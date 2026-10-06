#!/usr/bin/env python3
"""Compose #2816 current balance with #2810 post-QP Lesia evidence.

Execution-only, zero-credit carrier. Reuses the exact incumbent strict-JSON,
inventory and NEXT100-106 mechanics from #2816; it only proves the cross-lineage
union and updates the already-existing Lesia family capacity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import tools.run_d03_current_balance_nbu_postqp_v1 as parent

BASE_HEAD = "64a9cb6b8decca7b045d2cc6cad07e80b0655cec"
BASE_COMPOSITION_ID = "6a0860bc85cb7f2bf81580250c69ced7c15c63c40a622070fa9f2655b97e7f74"
BASE_RECEIPT_ID = "76dc985be9b0debbea9f5691051122fd36ebba3325f600c9a737e99a023d5355"
BASE_DEDUP_ID = "dfadc4165b6406f1b1a6981e71b7e29a7d16c28726e4b7f48963b7c978de2d6a"
LESIA_HEAD = "1d594e048365fd2121433210b83c0dc9069c50bf"
LESIA_EVIDENCE_ID = "dddddf9b99e4050d6b8483a8b18822a1453be5511772668315c800b947c17b8d"
LESIA_FAMILY = "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
BASE_RECORDS, BASE_SOURCES, BASE_BYTES = 2751, 2670, 20_661_232
LESIA_RECORDS, LESIA_BYTES = 105, 159_722
COMBINED_RECORDS, COMBINED_SOURCES, COMBINED_BYTES = 2856, 2775, 20_820_954
CAPACITY = {"ua": 1_256_359, "en": 15_481_867, "code": 4_082_728}
FAMILY_COUNT = {"ua": 4, "en": 7, "code": 18}
GAPS = {"ua": 7_743_641, "en": 0, "code": 0}
MAX_TOTAL = 1_825_100
MAX_STRATA = {"ua": 821_295, "en": 638_785, "code": 365_020}
DEDUP_SCHEMA = "12-6.d03-current-balance-lesia-postqp-dedup-proof.v1"
COMPOSITION_SCHEMA = "12-6.d03-current-balance-lesia-postqp-composition.v1"
RECEIPT_SCHEMA = "12-6.d03-current-balance-lesia-postqp-balance-execution.v1"

ExecutionError = parent.ExecutionError
require = parent.require
load = parent.load
canonical = parent.canonical
sha256 = parent.sha256
self_hash = parent.self_hash
lowercase_sha = parent.lowercase_sha
zero_truth = parent.zero_truth
verify_inventory = parent.verify_inventory
gate = parent.gate


def verify_base(
    composition: Mapping[str, Any],
    vector: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(
        composition.get("schema") == "12-6.d03-current-balance-nbu-postqp-composition.v1"
        and composition.get("source_git_sha") == BASE_HEAD,
        "base composition authority drift",
    )
    require(
        composition.get("composition_identity_sha256") == BASE_COMPOSITION_ID
        and self_hash(composition, "composition_identity_sha256") == BASE_COMPOSITION_ID,
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
        expected_records=BASE_RECORDS,
        expected_bytes=BASE_BYTES,
    )
    require(len({row["source_id"] for row in rows}) == BASE_SOURCES, "base source count drift")

    require(
        receipt.get("schema") == "12-6.d03-current-balance-nbu-postqp-balance-execution.v1"
        and receipt.get("execution_head_sha") == BASE_HEAD,
        "base receipt authority drift",
    )
    require(
        receipt.get("receipt_identity_sha256") == BASE_RECEIPT_ID
        and self_hash(receipt, "receipt_identity_sha256") == BASE_RECEIPT_ID,
        "base receipt identity drift",
    )
    require(
        receipt.get("composition_identity_sha256") == BASE_COMPOSITION_ID
        and receipt.get("combined_dedup_proof_identity_sha256") == BASE_DEDUP_ID
        and receipt.get("balance_policy_identity_sha256") == POLICY_ID,
        "base receipt lineage drift",
    )
    require(
        receipt.get("combined_record_count") == BASE_RECORDS
        and receipt.get("combined_source_object_count") == BASE_SOURCES
        and receipt.get("combined_total_payload_bytes") == BASE_BYTES,
        "base receipt totals drift",
    )
    zero_truth(receipt, "base receipt")

    require(
        vector.get("schema_version") == gate.INPUT_SCHEMA
        and vector.get("terminal") is True,
        "base vector drift",
    )
    authority = vector.get("dedup_authority")
    require(
        isinstance(authority, Mapping)
        and authority.get("head_sha") == BASE_HEAD
        and authority.get("evidence_identity_sha256") == BASE_DEDUP_ID
        and authority.get("terminal_verdict") == "PASS",
        "base vector authority drift",
    )
    gate.validate_vector(dict(vector))
    families = [dict(row) for row in vector["families"]]
    by_family: dict[str, int] = defaultdict(int)
    for row in rows:
        by_family[row["family"]] += int(row["payload_bytes"])
    require(
        {row["family_id"]: row["unique_bytes"] for row in families} == dict(by_family),
        "base vector/inventory family bytes drift",
    )
    lesia = [row for row in families if row["family_id"] == LESIA_FAMILY]
    require(
        len(lesia) == 1
        and lesia[0]
        == {"family_id": LESIA_FAMILY, "stratum": "ua", "unique_bytes": 1479},
        "base Lesia family drift",
    )
    return rows, families


def verify_lesia(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(
        evidence.get("schema_version")
        == "12-6.d03-wikisource-lesia1892-decontam-g05-g06-execution.v1"
        and evidence.get("execution_head_sha") == LESIA_HEAD,
        "Lesia evidence authority drift",
    )
    require(
        evidence.get("evidence_identity_sha256") == LESIA_EVIDENCE_ID
        and self_hash(evidence, "evidence_identity_sha256") == LESIA_EVIDENCE_ID,
        "Lesia evidence identity drift",
    )
    truth = evidence.get("truth_boundary")
    require(isinstance(truth, Mapping), "Lesia truth boundary missing")
    require(
        truth.get("global_cross_source_dedup_complete_for_lesia_parent") is True
        and truth.get("reserved_evaluation_decontamination_complete_for_lesia") is True
        and truth.get("canonical_quality_privacy_complete_for_lesia") is True
        and truth.get("cross_lineage_rededup_after_post_qp_complete") is False
        and truth.get("balance_diversity_retest_complete") is False
        and truth.get("family_caps_complete") is False,
        "Lesia gate truth drift",
    )
    zero_truth(truth, "Lesia truth")
    gate_execution = evidence.get("gate_execution")
    require(
        isinstance(gate_execution, Mapping)
        and gate_execution.get("survivor_records") == LESIA_RECORDS
        and gate_execution.get("survivor_source_objects") == LESIA_RECORDS
        and gate_execution.get("survivor_payload_bytes") == LESIA_BYTES
        and gate_execution.get("later_gate_loss_bytes") == 9677,
        "Lesia survivor result drift",
    )
    rows = verify_inventory(
        evidence["survivor_inventory"],
        label="Lesia survivor inventory",
        expected_records=LESIA_RECORDS,
        expected_bytes=LESIA_BYTES,
    )
    require(
        {row["family"] for row in rows} == {LESIA_FAMILY}
        and {row["modality"] for row in rows} == {"uk"}
        and len({row["source_id"] for row in rows}) == LESIA_RECORDS,
        "Lesia inventory family/source drift",
    )
    capacity = evidence.get("capacity_observation")
    require(
        isinstance(capacity, Mapping)
        and capacity.get("qualified_post_qp_survivor_payload_bytes") == LESIA_BYTES
        and capacity.get("safe_incremental_unique_upper_bound_bytes") == LESIA_BYTES
        and capacity.get("exact_incremental_unique_capacity_credited_bytes") == 0
        and capacity.get("exact_cross_lineage_rededup_required") is True
        and capacity.get("balance_retest_authorized") is False,
        "Lesia capacity boundary drift",
    )
    return rows


def execute(
    *,
    base_composition: Mapping[str, Any],
    base_vector: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    lesia_evidence: Mapping[str, Any],
    execution_head: str,
) -> dict[str, Mapping[str, Any]]:
    lowercase_sha(execution_head, "execution_head", 40)
    base_rows, families = verify_base(base_composition, base_vector, base_receipt)
    lesia_rows = verify_lesia(lesia_evidence)
    record_ids = {row["record_id"] for row in base_rows}
    source_ids = {row["source_id"] for row in base_rows}
    payloads = {row["payload_sha256"] for row in base_rows}
    require(len(payloads) == BASE_RECORDS, "base payload hashes are not unique")
    require(
        not any(row["record_id"] in record_ids for row in lesia_rows),
        "Lesia record collision",
    )
    require(
        not any(row["source_id"] in source_ids for row in lesia_rows),
        "Lesia source collision",
    )
    require(
        not any(row["payload_sha256"] in payloads for row in lesia_rows),
        "Lesia payload replay",
    )

    combined = sorted(
        [dict(row) for row in base_rows] + [dict(row) for row in lesia_rows],
        key=lambda row: row["record_id"],
    )
    require(
        len(combined) == COMBINED_RECORDS
        and len({r["source_id"] for r in combined}) == COMBINED_SOURCES,
        "combined cardinality drift",
    )
    require(
        len({r["payload_sha256"] for r in combined}) == COMBINED_RECORDS
        and sum(r["payload_bytes"] for r in combined) == COMBINED_BYTES,
        "combined payload drift",
    )
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in combined
    ]
    inventory = {
        "schema_version": parent.INVENTORY_SCHEMA,
        "record_count": COMBINED_RECORDS,
        "total_payload_bytes": COMBINED_BYTES,
        "records": combined,
        "record_inventory_digest_sha256": sha256(combined),
        "payload_inventory_digest_sha256": sha256(payload_projection),
    }

    dedup_core = {
        "schema": DEDUP_SCHEMA,
        "base_head_sha": BASE_HEAD,
        "base_composition_identity_sha256": BASE_COMPOSITION_ID,
        "base_dedup_identity_sha256": BASE_DEDUP_ID,
        "lesia_head_sha": LESIA_HEAD,
        "lesia_evidence_identity_sha256": LESIA_EVIDENCE_ID,
        "base_records": BASE_RECORDS,
        "base_source_objects": BASE_SOURCES,
        "base_payload_bytes": BASE_BYTES,
        "lesia_records": LESIA_RECORDS,
        "lesia_source_objects": LESIA_RECORDS,
        "lesia_payload_bytes": LESIA_BYTES,
        "discounted_payload_replay_records": 0,
        "discounted_payload_replay_bytes": 0,
        "novel_lesia_records": LESIA_RECORDS,
        "novel_lesia_source_objects": LESIA_RECORDS,
        "novel_lesia_payload_bytes": LESIA_BYTES,
        "cross_inventory_record_id_collision_free": True,
        "cross_inventory_source_id_collision_free": True,
        "cross_inventory_exact_payload_collision_free": True,
        "combined_records": COMBINED_RECORDS,
        "combined_source_objects": COMBINED_SOURCES,
        "combined_payload_bytes": COMBINED_BYTES,
        "combined_record_inventory_digest_sha256": inventory["record_inventory_digest_sha256"],
        "combined_payload_inventory_digest_sha256": inventory["payload_inventory_digest_sha256"],
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
    proof = {**dedup_core, "evidence_identity_sha256": sha256(dedup_core)}

    lesia_family = next(row for row in families if row["family_id"] == LESIA_FAMILY)
    lesia_family["unique_bytes"] += LESIA_BYTES
    families.sort(key=lambda row: row["family_id"])
    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-CURRENT-BALANCE-LESIA-POSTQP-UNION-V1",
            "head_sha": execution_head,
            "evidence_identity_sha256": proof["evidence_identity_sha256"],
            "terminal_verdict": "PASS",
        },
        "families": families,
        "totals": {
            "total_unique_bytes": COMBINED_BYTES,
            "by_stratum": CAPACITY,
            "family_count": FAMILY_COUNT,
        },
    }
    policy = gate.load_json(gate.POLICY_PATH)
    gate.validate_policy(policy)
    require(policy.get("policy_identity_sha256") == POLICY_ID, "policy identity drift")
    gate.validate_vector(vector)
    balance = gate.evaluate(policy, vector)
    require(
        balance.get("status") == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"
        and balance.get("maximum_feasible_total_source_bytes") == MAX_TOTAL
        and balance.get("maximum_feasible_stratum_bytes") == MAX_STRATA
        and balance.get("raw_capacity_by_stratum") == CAPACITY
        and balance.get("raw_gap_to_target_by_stratum") == GAPS
        and balance.get("family_minimum")
        == {"required_per_stratum": 2, "observed": FAMILY_COUNT, "pass": True},
        "balance result drift",
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
        "combined_dedup_proof_identity_sha256": proof["evidence_identity_sha256"],
        "combined_inventory": inventory,
        "stratum_capacity_bytes": {
            "uk": CAPACITY["ua"],
            "en": CAPACITY["en"],
            "code": CAPACITY["code"],
        },
        "stratum_family_counts": {
            "uk": FAMILY_COUNT["ua"],
            "en": FAMILY_COUNT["en"],
            "code": FAMILY_COUNT["code"],
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
    composition = {**composition_core, "composition_identity_sha256": sha256(composition_core)}
    receipt_core = {
        "schema": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "base_composition_identity_sha256": BASE_COMPOSITION_ID,
        "base_receipt_identity_sha256": BASE_RECEIPT_ID,
        "lesia_evidence_identity_sha256": LESIA_EVIDENCE_ID,
        "combined_dedup_proof_identity_sha256": proof["evidence_identity_sha256"],
        "composition_identity_sha256": composition["composition_identity_sha256"],
        "balance_policy_identity_sha256": POLICY_ID,
        "balance_result_identity_sha256": balance["result_identity_sha256"],
        "discounted_exact_replay_records": 0,
        "discounted_exact_replay_payload_bytes": 0,
        "novel_lesia_records": LESIA_RECORDS,
        "novel_lesia_source_objects": LESIA_RECORDS,
        "novel_lesia_payload_bytes": LESIA_BYTES,
        "combined_record_count": COMBINED_RECORDS,
        "combined_source_object_count": COMBINED_SOURCES,
        "combined_total_payload_bytes": COMBINED_BYTES,
        "raw_capacity_by_stratum": CAPACITY,
        "family_minimum": balance["family_minimum"],
        "maximum_feasible_total_source_bytes": MAX_TOTAL,
        "maximum_feasible_stratum_bytes": MAX_STRATA,
        "raw_gap_to_target_by_stratum": GAPS,
        "balance_status": balance["status"],
        "next_scientific_gate": "ACQUIRE_MORE_DIVERSE_LAWFUL_UA_SOURCE_CAPACITY",
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
        "combined-dedup-proof": proof,
        "composition": composition,
        "next100-input": vector,
        "balance-result": balance,
        "execution-receipt": receipt,
    }


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
        result = execute(
            base_composition=load(args.base_composition_json),
            base_vector=load(args.base_next100_json),
            base_receipt=load(args.base_receipt_json),
            lesia_evidence=load(args.lesia_evidence_json),
            execution_head=args.execution_head,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            (args.output_dir / f"{name}.json").write_bytes(canonical(value) + b"\n")
    except (ExecutionError, gate.GateError, OSError, ValueError, StopIteration) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    receipt = result["execution-receipt"]
    print("D03_LESIA_COMBINED_BALANCE=" + str(receipt["balance_status"]))
    print("NOVEL_LESIA_PAYLOAD_BYTES=" + str(receipt["novel_lesia_payload_bytes"]))
    print("COMBINED_TOTAL_PAYLOAD_BYTES=" + str(receipt["combined_total_payload_bytes"]))
    print(
        "MAX_FEASIBLE_TOTAL_SOURCE_BYTES="
        + str(receipt["maximum_feasible_total_source_bytes"])
    )
    print("UA_RAW_GAP_BYTES=" + str(receipt["raw_gap_to_target_by_stratum"]["ua"]))
    print("RECEIPT_IDENTITY_SHA256=" + str(receipt["receipt_identity_sha256"]))
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
