#!/usr/bin/env python3
"""Compose #2875 current balance with only the terminal current-Rada family.

Execution-only, zero-credit carrier. The full current-Rada post-G06 artifact contains
an older/common baseline that is not byte-identical to #2875 for every shared source.
This runner therefore fails closed on the measured baseline drift and admits only the
independently named Rada family after proving zero record/source/exact-payload
collisions against the exact #2875 inventory. Balance arithmetic is delegated to the
canonical NEXT100-106 gate unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import tools.run_d03_current_balance_lesia_postqp_v1 as parent

BASE_HEAD = "06bfeb25c3620e0f610d28dae47f8f34a1c1bbff"
BASE_COMPOSITION_ID = "019f038d233a280d239731441b66aff49fcc843d4262b002e6d6fa45c8d98ce8"
BASE_RECEIPT_ID = "3292e4c63e6ce1c9450d581742f63f0fb5cb3fb7d0517f901c9fd2350969f21a"
BASE_DEDUP_ID = "5aedfcf07e1950b07ba2bcf3ce2f2dca1e70d6095091a73f4970b19ad98ba1f7"
RADA_HEAD = "a63d7c88ccb5fa380b7af4b44bde32433cf35877"
RADA_EVIDENCE_ID = "8bf576920ade8ca864bc3a00a41888f05c1e16a191a5dd8aec3b95ce903a4163"
RADA_PROOF_ID = "d716a9650ebfce142737188ec447834357e3881227fec43fcc38304964bac635"
RADA_FAMILY = "ua.rada.open-data.laws-texts"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"

BASE_RECORDS, BASE_SOURCES, BASE_BYTES = 2_856, 2_775, 20_820_954
RADA_ALL_RECORDS, RADA_ALL_BYTES = 97_737, 190_586_561
RADA_RECORDS, RADA_BYTES = 97_496, 185_158_183
RADA_EXCLUDED_NONFAMILY_RECORDS = 241
RADA_EXCLUDED_NONFAMILY_BYTES = 5_428_378
FULL_RECORD_ID_OVERLAP = 240
FULL_SOURCE_ID_OVERLAP = 241
FULL_PAYLOAD_OVERLAP = 234
FULL_RECORD_ID_CONFLICTS = 6
FULL_SOURCE_ID_CONFLICTS = 7

COMBINED_RECORDS, COMBINED_SOURCES, COMBINED_BYTES = 100_352, 100_271, 205_979_137
CAPACITY = {"ua": 186_414_542, "en": 15_481_867, "code": 4_082_728}
FAMILY_COUNT = {"ua": 5, "en": 7, "code": 18}
GAPS = {"ua": 0, "en": 0, "code": 0}
MAX_TOTAL = 6_281_700
MAX_STRATA = {"ua": 2_826_765, "en": 2_198_595, "code": 1_256_340}
MIX_SHORTFALL = 13_718_300

DEDUP_SCHEMA = "12-6.d03-current-balance-rada-family-dedup-proof.v1"
COMPOSITION_SCHEMA = "12-6.d03-current-balance-rada-family-composition.v1"
RECEIPT_SCHEMA = "12-6.d03-current-balance-rada-family-balance-execution.v1"

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
        composition.get("schema") == "12-6.d03-current-balance-lesia-postqp-composition.v1"
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
    require(len({row["payload_sha256"] for row in rows}) == BASE_RECORDS, "base payload uniqueness drift")

    require(
        receipt.get("schema") == "12-6.d03-current-balance-lesia-postqp-balance-execution.v1"
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
        vector.get("schema_version") == gate.INPUT_SCHEMA and vector.get("terminal") is True,
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
    require(
        not any(row["family_id"] == RADA_FAMILY for row in families),
        "base already contains Rada family",
    )
    return rows, families


def verify_rada(
    evidence: Mapping[str, Any],
    inventory: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(
        evidence.get("schema_version") == "12-6.d03-rada-current-postdata232-g05-g06-execution.v1"
        and evidence.get("execution_head_sha") == RADA_HEAD,
        "Rada evidence authority drift",
    )
    require(
        evidence.get("evidence_identity_sha256") == RADA_EVIDENCE_ID
        and self_hash(evidence, "evidence_identity_sha256") == RADA_EVIDENCE_ID,
        "Rada evidence identity drift",
    )
    counts = evidence.get("counts")
    require(
        isinstance(counts, Mapping)
        and counts.get("post_g06_records") == RADA_ALL_RECORDS
        and counts.get("post_g06_source_objects") == RADA_ALL_RECORDS
        and counts.get("post_g06_payload_bytes") == RADA_ALL_BYTES,
        "Rada evidence totals drift",
    )
    content_boundary = evidence.get("content_boundary")
    require(
        isinstance(content_boundary, Mapping)
        and content_boundary.get("durable_output_text_free") is True
        and content_boundary.get("raw_evaluation_text_persisted") is False
        and content_boundary.get("raw_survivor_text_persisted") is False
        and content_boundary.get("raw_training_text_persisted") is False,
        "Rada content boundary drift",
    )
    truth = evidence.get("truth_boundary")
    require(
        isinstance(truth, Mapping)
        and truth.get("current_rada_data232_parent_two_clean_complete") is True
        and truth.get("canonical_quality_privacy_executed") is True
        and truth.get("balance_diversity_retest_complete") is False
        and truth.get("family_caps_complete") is False,
        "Rada gate truth drift",
    )
    zero_truth(truth, "Rada truth")

    rows = verify_inventory(
        inventory,
        label="Rada survivor inventory",
        expected_records=RADA_ALL_RECORDS,
        expected_bytes=RADA_ALL_BYTES,
    )
    inv_summary = evidence.get("survivor_inventory")
    require(isinstance(inv_summary, Mapping), "Rada inventory summary missing")
    for field in (
        "record_inventory_digest_sha256",
        "payload_inventory_digest_sha256",
        "record_count",
        "total_payload_bytes",
    ):
        require(
            inv_summary.get(field) == inventory.get(field),
            f"Rada inventory binding drift: {field}",
        )

    require(
        proof.get("schema_version") == "12-6.d03-rada-current-postdata232-g05-g06-two-clean.v1"
        and proof.get("execution_head_sha") == RADA_HEAD,
        "Rada two-clean proof authority drift",
    )
    require(
        proof.get("proof_identity_sha256") == RADA_PROOF_ID
        and self_hash(proof, "proof_identity_sha256") == RADA_PROOF_ID
        and proof.get("evidence_identity_sha256") == RADA_EVIDENCE_ID,
        "Rada two-clean proof identity drift",
    )
    require(
        proof.get("fresh_execution_count") == 2
        and proof.get("independent_runner_jobs") is True
        and proof.get("byte_identical_outputs") is True
        and proof.get("record_count") == RADA_ALL_RECORDS
        and proof.get("total_payload_bytes") == RADA_ALL_BYTES
        and proof.get("record_inventory_digest_sha256")
        == inventory.get("record_inventory_digest_sha256")
        and proof.get("payload_inventory_digest_sha256")
        == inventory.get("payload_inventory_digest_sha256"),
        "Rada two-clean proof result drift",
    )
    zero_truth(proof, "Rada two-clean proof")

    selected = [dict(row) for row in rows if row["family"] == RADA_FAMILY]
    excluded = [dict(row) for row in rows if row["family"] != RADA_FAMILY]
    require(
        len(selected) == RADA_RECORDS
        and sum(row["payload_bytes"] for row in selected) == RADA_BYTES
        and len({row["record_id"] for row in selected}) == RADA_RECORDS
        and len({row["source_id"] for row in selected}) == RADA_RECORDS
        and len({row["payload_sha256"] for row in selected}) == RADA_RECORDS
        and {row["modality"] for row in selected} == {"uk"},
        "Rada selected family drift",
    )
    require(
        len(excluded) == RADA_EXCLUDED_NONFAMILY_RECORDS
        and sum(row["payload_bytes"] for row in excluded) == RADA_EXCLUDED_NONFAMILY_BYTES,
        "Rada excluded baseline drift",
    )
    return rows, selected


def execute(
    *,
    base_composition: Mapping[str, Any],
    base_vector: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    rada_evidence: Mapping[str, Any],
    rada_inventory: Mapping[str, Any],
    rada_proof: Mapping[str, Any],
    execution_head: str,
) -> dict[str, Mapping[str, Any]]:
    lowercase_sha(execution_head, "execution_head", 40)
    base_rows, families = verify_base(base_composition, base_vector, base_receipt)
    rada_all, rada_rows = verify_rada(rada_evidence, rada_inventory, rada_proof)

    base_record = {row["record_id"]: row for row in base_rows}
    base_source = {row["source_id"]: row for row in base_rows}
    base_payload = {row["payload_sha256"]: row for row in base_rows}

    full_record_overlap = [row for row in rada_all if row["record_id"] in base_record]
    full_source_overlap = [row for row in rada_all if row["source_id"] in base_source]
    full_payload_overlap = [row for row in rada_all if row["payload_sha256"] in base_payload]
    full_record_conflicts = [
        row
        for row in full_record_overlap
        if row["payload_sha256"] != base_record[row["record_id"]]["payload_sha256"]
    ]
    full_source_conflicts = [
        row
        for row in full_source_overlap
        if row["payload_sha256"] != base_source[row["source_id"]]["payload_sha256"]
    ]
    require(
        len(full_record_overlap) == FULL_RECORD_ID_OVERLAP
        and len(full_source_overlap) == FULL_SOURCE_ID_OVERLAP
        and len(full_payload_overlap) == FULL_PAYLOAD_OVERLAP
        and len(full_record_conflicts) == FULL_RECORD_ID_CONFLICTS
        and len(full_source_conflicts) == FULL_SOURCE_ID_CONFLICTS,
        "full Rada/common-baseline drift changed; slice decision must be requalified",
    )

    require(
        not any(row["record_id"] in base_record for row in rada_rows),
        "Rada-family record collision",
    )
    require(
        not any(row["source_id"] in base_source for row in rada_rows),
        "Rada-family source collision",
    )
    require(
        not any(row["payload_sha256"] in base_payload for row in rada_rows),
        "Rada-family payload replay",
    )

    combined = sorted(
        [dict(row) for row in base_rows] + [dict(row) for row in rada_rows],
        key=lambda row: row["record_id"],
    )
    require(
        len(combined) == COMBINED_RECORDS
        and len({row["record_id"] for row in combined}) == COMBINED_RECORDS
        and len({row["source_id"] for row in combined}) == COMBINED_SOURCES
        and len({row["payload_sha256"] for row in combined}) == COMBINED_RECORDS,
        "combined cardinality/collision drift",
    )
    require(
        sum(row["payload_bytes"] for row in combined) == COMBINED_BYTES,
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
        "schema_version": INVENTORY_SCHEMA,
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
        "rada_head_sha": RADA_HEAD,
        "rada_evidence_identity_sha256": RADA_EVIDENCE_ID,
        "rada_two_clean_proof_identity_sha256": RADA_PROOF_ID,
        "selection_rule": "ONLY_FAMILY_ua.rada.open-data.laws-texts_FROM_TERMINAL_RADA_ARTIFACT",
        "full_rada_artifact_records": RADA_ALL_RECORDS,
        "full_rada_artifact_payload_bytes": RADA_ALL_BYTES,
        "full_rada_vs_base_record_id_overlap": FULL_RECORD_ID_OVERLAP,
        "full_rada_vs_base_source_id_overlap": FULL_SOURCE_ID_OVERLAP,
        "full_rada_vs_base_exact_payload_overlap": FULL_PAYLOAD_OVERLAP,
        "full_rada_vs_base_record_id_conflicts": FULL_RECORD_ID_CONFLICTS,
        "full_rada_vs_base_source_id_conflicts": FULL_SOURCE_ID_CONFLICTS,
        "excluded_non_rada_records": RADA_EXCLUDED_NONFAMILY_RECORDS,
        "excluded_non_rada_payload_bytes": RADA_EXCLUDED_NONFAMILY_BYTES,
        "selected_rada_records": RADA_RECORDS,
        "selected_rada_source_objects": RADA_RECORDS,
        "selected_rada_payload_bytes": RADA_BYTES,
        "selected_rada_record_id_collision_free": True,
        "selected_rada_source_id_collision_free": True,
        "selected_rada_exact_payload_collision_free": True,
        "combined_records": COMBINED_RECORDS,
        "combined_source_objects": COMBINED_SOURCES,
        "combined_payload_bytes": COMBINED_BYTES,
        "combined_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "combined_payload_inventory_digest_sha256": inventory[
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
    proof = {**dedup_core, "evidence_identity_sha256": sha256(dedup_core)}

    families.append(
        {"family_id": RADA_FAMILY, "stratum": "ua", "unique_bytes": RADA_BYTES}
    )
    families.sort(key=lambda row: row["family_id"])
    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-CURRENT-BALANCE-RADA-FAMILY-UNION-V1",
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
    require(
        policy.get("policy_identity_sha256") == POLICY_ID,
        "policy identity drift",
    )
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
        "status": "COMPOSED_ZERO_CREDIT_PENDING_MORE_UA_FAMILY_CAPACITY",
        "source_git_sha": execution_head,
        "parents": {
            "base_balance_head_sha": BASE_HEAD,
            "base_composition_identity_sha256": BASE_COMPOSITION_ID,
            "base_receipt_identity_sha256": BASE_RECEIPT_ID,
            "rada_post_g06_head_sha": RADA_HEAD,
            "rada_evidence_identity_sha256": RADA_EVIDENCE_ID,
            "rada_two_clean_proof_identity_sha256": RADA_PROOF_ID,
        },
        "combined_dedup_proof_identity_sha256": proof[
            "evidence_identity_sha256"
        ],
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
        "rada_evidence_identity_sha256": RADA_EVIDENCE_ID,
        "rada_two_clean_proof_identity_sha256": RADA_PROOF_ID,
        "combined_dedup_proof_identity_sha256": proof[
            "evidence_identity_sha256"
        ],
        "composition_identity_sha256": composition[
            "composition_identity_sha256"
        ],
        "balance_policy_identity_sha256": POLICY_ID,
        "balance_result_identity_sha256": balance[
            "result_identity_sha256"
        ],
        "selected_rada_records": RADA_RECORDS,
        "selected_rada_source_objects": RADA_RECORDS,
        "selected_rada_payload_bytes": RADA_BYTES,
        "excluded_non_rada_records": RADA_EXCLUDED_NONFAMILY_RECORDS,
        "excluded_non_rada_payload_bytes": RADA_EXCLUDED_NONFAMILY_BYTES,
        "combined_record_count": COMBINED_RECORDS,
        "combined_source_object_count": COMBINED_SOURCES,
        "combined_total_payload_bytes": COMBINED_BYTES,
        "raw_capacity_by_stratum": CAPACITY,
        "family_minimum": balance["family_minimum"],
        "maximum_feasible_total_source_bytes": MAX_TOTAL,
        "maximum_feasible_stratum_bytes": MAX_STRATA,
        "remaining_policy_mix_shortfall_bytes": MIX_SHORTFALL,
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
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": sha256(receipt_core),
    }
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
    parser.add_argument("--rada-evidence-json", type=Path, required=True)
    parser.add_argument("--rada-inventory-json", type=Path, required=True)
    parser.add_argument("--rada-proof-json", type=Path, required=True)
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
            rada_evidence=load(args.rada_evidence_json),
            rada_inventory=load(args.rada_inventory_json),
            rada_proof=load(args.rada_proof_json),
            execution_head=args.execution_head,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            (args.output_dir / f"{name}.json").write_bytes(
                canonical(value) + b"\n"
            )
    except (
        ExecutionError,
        gate.GateError,
        OSError,
        ValueError,
        StopIteration,
        KeyError,
        TypeError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    receipt = result["execution-receipt"]
    print(
        "D03_RADA_FAMILY_COMBINED_BALANCE="
        + str(receipt["balance_status"])
    )
    print(
        "SELECTED_RADA_PAYLOAD_BYTES="
        + str(receipt["selected_rada_payload_bytes"])
    )
    print(
        "COMBINED_TOTAL_PAYLOAD_BYTES="
        + str(receipt["combined_total_payload_bytes"])
    )
    print(
        "MAXIMUM_FEASIBLE_TOTAL_SOURCE_BYTES="
        + str(receipt["maximum_feasible_total_source_bytes"])
    )
    print(
        "NEXT_SCIENTIFIC_GATE="
        + str(receipt["next_scientific_gate"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
