#!/usr/bin/env python3
"""Compose the qualified current balance with the qualified Lang-UK court family.

Execution-only, zero-credit carrier. It authenticates the exact #2954 current-balance
artifact and the exact #3039 post-dedup current-clean Lang-UK evidence, proves zero
record/source/exact-payload collision, then re-runs the unchanged NEXT100-106
balance/family-cap gate. It creates no corpus, tokenizer, training, final-test,
paid-compute, or scale-promotion authority.
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

BASE_HEAD = "61a0cdb2239085dd5fe11af5af9a06e90cad4e33"
BASE_COMPOSITION_ID = "9a16993adbfca998a22fd5d947c5cb430db6eca3b20c2958b84b28109ccb494a"
BASE_RECEIPT_ID = "c3e415822cc7c3bc36119bca640e7c9e762376cd30bd3b3c3c2f64e99d644eb4"
BASE_DEDUP_ID = "e512b4cf742e967fe4d6ceea0422307ad912150578b51a359462e078554e4044"
LANGUK_HEAD = "f6a2a440aacebec6a44a5f2ba955902970ff4d50"
LANGUK_EVIDENCE_ID = "94c317ff01bdf4881fbbd08f4f55271eb6a82b15a02e9e4cd20f17830663383f"
LANGUK_PROOF_ID = "6bc34219d2e4d0c78bd380e2510cf2733198cab0f0165a4c7044b4e6d763c489"
LANGUK_FAMILY = "ua.languk.supreme-court-decisions"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"

BASE_RECORDS, BASE_SOURCES, BASE_BYTES = 100_352, 100_271, 205_979_137
BASE_CAPACITY = {"ua": 186_414_542, "en": 15_481_867, "code": 4_082_728}
BASE_FAMILY_COUNT = {"ua": 5, "en": 7, "code": 18}
LANGUK_RECORDS, LANGUK_SOURCES, LANGUK_BYTES = 256, 256, 2_880_510
COMBINED_RECORDS, COMBINED_SOURCES, COMBINED_BYTES = 100_608, 100_527, 208_859_647
CAPACITY = {"ua": 189_295_052, "en": 15_481_867, "code": 4_082_728}
FAMILY_COUNT = {"ua": 6, "en": 7, "code": 18}
TARGET_TOTAL = 20_000_000
TARGET_STRATA = {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000}

DEDUP_SCHEMA = "12-6.d03-current-balance-languk-family-dedup-proof.v1"
COMPOSITION_SCHEMA = "12-6.d03-current-balance-languk-family-composition.v1"
RECEIPT_SCHEMA = "12-6.d03-current-balance-languk-family-balance-execution.v1"
MAX_INPUT_BYTES = 64 * 1024 * 1024


class ExecutionError(RuntimeError):
    """Fail-closed execution error."""


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
    require(type(value) is dict, f"{path} must contain an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: Any) -> str:
    raw = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(raw).hexdigest()


def self_hash(value: Mapping[str, Any], identity_field: str) -> str:
    claimed = value.get(identity_field)
    require(type(claimed) is str and len(claimed) == 64, f"missing {identity_field}")
    core = dict(value)
    core.pop(identity_field, None)
    require(sha256(core) == claimed, f"{identity_field} self-hash mismatch")
    return claimed


def lowercase_sha(value: Any, field: str, length: int) -> str:
    require(type(value) is str and len(value) == length, f"{field} invalid length")
    require(value.lower() == value, f"{field} must be lowercase")
    require(all(ch in "0123456789abcdef" for ch in value), f"{field} must be hex")
    return value


def zero_truth(value: Mapping[str, Any], label: str) -> None:
    checks = {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    for field, expected in checks.items():
        if field in value:
            require(value.get(field) == expected, f"{label} widened {field}")


def verify_inventory(
    inventory: Mapping[str, Any], *, label: str, expected_records: int, expected_bytes: int
) -> list[dict[str, Any]]:
    require(inventory.get("schema_version") == INVENTORY_SCHEMA, f"{label} schema drift")
    rows = inventory.get("records")
    require(type(rows) is list and len(rows) == expected_records, f"{label} record count drift")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        require(type(row) is dict, f"{label}[{index}] must be object")
        for key in ("record_id", "source_id", "payload_sha256", "family", "modality"):
            require(type(row.get(key)) is str and bool(row[key]), f"{label}[{index}].{key} invalid")
        lowercase_sha(row["payload_sha256"], f"{label}[{index}].payload_sha256", 64)
        require(type(row.get("payload_bytes")) is int and row["payload_bytes"] > 0, f"{label}[{index}].payload_bytes invalid")
        normalized.append(dict(row))
    require(inventory.get("record_count") == expected_records, f"{label} declared count drift")
    require(sum(row["payload_bytes"] for row in normalized) == expected_bytes, f"{label} payload total drift")
    require(inventory.get("total_payload_bytes") == expected_bytes, f"{label} declared bytes drift")
    require(len({row["record_id"] for row in normalized}) == expected_records, f"{label} duplicate record id")
    payload_projection = [
        {"record_id": row["record_id"], "payload_sha256": row["payload_sha256"], "payload_bytes": row["payload_bytes"]}
        for row in normalized
    ]
    require(sha256(normalized) == inventory.get("record_inventory_digest_sha256"), f"{label} record root drift")
    require(sha256(payload_projection) == inventory.get("payload_inventory_digest_sha256"), f"{label} payload root drift")
    return normalized


def verify_base(
    composition: Mapping[str, Any], vector: Mapping[str, Any], receipt: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(
        composition.get("schema") == "12-6.d03-current-balance-rada-family-composition.v1"
        and composition.get("source_git_sha") == BASE_HEAD,
        "base composition authority drift",
    )
    require(
        composition.get("composition_identity_sha256") == BASE_COMPOSITION_ID
        and self_hash(composition, "composition_identity_sha256") == BASE_COMPOSITION_ID,
        "base composition identity drift",
    )
    require(composition.get("combined_dedup_proof_identity_sha256") == BASE_DEDUP_ID, "base dedup identity drift")
    zero_truth(composition, "base composition")
    rows = verify_inventory(composition["combined_inventory"], label="base inventory", expected_records=BASE_RECORDS, expected_bytes=BASE_BYTES)
    require(len({row["source_id"] for row in rows}) == BASE_SOURCES, "base source count drift")
    require(len({row["payload_sha256"] for row in rows}) == BASE_RECORDS, "base payload uniqueness drift")

    require(
        receipt.get("schema") == "12-6.d03-current-balance-rada-family-balance-execution.v1"
        and receipt.get("execution_head_sha") == BASE_HEAD,
        "base receipt authority drift",
    )
    require(receipt.get("receipt_identity_sha256") == BASE_RECEIPT_ID and self_hash(receipt, "receipt_identity_sha256") == BASE_RECEIPT_ID, "base receipt identity drift")
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

    require(vector.get("schema_version") == gate.INPUT_SCHEMA and vector.get("terminal") is True, "base vector drift")
    gate.validate_vector(dict(vector))
    require(vector.get("totals", {}).get("by_stratum") == BASE_CAPACITY, "base stratum capacity drift")
    require(vector.get("totals", {}).get("family_count") == BASE_FAMILY_COUNT, "base family count drift")
    families = [dict(row) for row in vector["families"]]
    by_family: dict[str, int] = defaultdict(int)
    for row in rows:
        by_family[row["family"]] += int(row["payload_bytes"])
    require({row["family_id"]: row["unique_bytes"] for row in families} == dict(by_family), "base vector/inventory family bytes drift")
    require(not any(row["family_id"] == LANGUK_FAMILY for row in families), "base already contains Lang-UK family")
    return rows, families


def verify_languk(evidence: Mapping[str, Any], proof: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(
        evidence.get("schema_version") == "12-6.d03-languk-court-postdedup-clean-execution.v1"
        and evidence.get("execution_head_sha") == LANGUK_HEAD,
        "Lang-UK evidence authority drift",
    )
    require(evidence.get("evidence_identity_sha256") == LANGUK_EVIDENCE_ID and self_hash(evidence, "evidence_identity_sha256") == LANGUK_EVIDENCE_ID, "Lang-UK evidence identity drift")
    truth = evidence.get("truth_boundary")
    require(type(truth) is dict, "Lang-UK truth boundary missing")
    require(
        truth.get("global_cross_source_dedup_complete_for_languk") is True
        and truth.get("reserved_evaluation_decontamination_complete_for_languk") is True
        and truth.get("canonical_quality_privacy_complete_for_languk") is True
        and truth.get("balance_diversity_retest_complete") is False
        and truth.get("family_caps_complete") is False,
        "Lang-UK gate truth drift",
    )
    zero_truth(truth, "Lang-UK truth")
    rows = verify_inventory(evidence["survivor_inventory"], label="Lang-UK survivor inventory", expected_records=LANGUK_RECORDS, expected_bytes=LANGUK_BYTES)
    require(
        len({row["source_id"] for row in rows}) == LANGUK_SOURCES
        and len({row["payload_sha256"] for row in rows}) == LANGUK_RECORDS
        and {row["family"] for row in rows} == {LANGUK_FAMILY}
        and {row["modality"] for row in rows} == {"uk"},
        "Lang-UK survivor family/source drift",
    )
    gate_execution = evidence.get("gate_execution")
    require(
        type(gate_execution) is dict
        and gate_execution.get("survivor_records") == LANGUK_RECORDS
        and gate_execution.get("survivor_source_objects") == LANGUK_SOURCES
        and gate_execution.get("survivor_payload_bytes") == LANGUK_BYTES
        and gate_execution.get("later_gate_loss_bytes") == 0,
        "Lang-UK gate result drift",
    )
    require(
        proof.get("schema_version") == "12-6.d03-languk-court-postdedup-clean-two-clean.v1"
        and proof.get("execution_head_sha") == LANGUK_HEAD,
        "Lang-UK proof authority drift",
    )
    require(proof.get("proof_identity_sha256") == LANGUK_PROOF_ID and self_hash(proof, "proof_identity_sha256") == LANGUK_PROOF_ID, "Lang-UK proof identity drift")
    require(
        proof.get("evidence_identity_sha256") == LANGUK_EVIDENCE_ID
        and proof.get("two_fresh_processes_byte_identical") is True
        and proof.get("languk_survivor_records") == LANGUK_RECORDS
        and proof.get("languk_survivor_source_objects") == LANGUK_SOURCES
        and proof.get("languk_survivor_payload_bytes") == LANGUK_BYTES,
        "Lang-UK proof result drift",
    )
    zero_truth(proof, "Lang-UK proof")
    return rows


def execute(
    *,
    base_composition: Mapping[str, Any],
    base_vector: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    languk_evidence: Mapping[str, Any],
    languk_proof: Mapping[str, Any],
    execution_head: str,
) -> dict[str, Mapping[str, Any]]:
    lowercase_sha(execution_head, "execution_head", 40)
    base_rows, families = verify_base(base_composition, base_vector, base_receipt)
    languk_rows = verify_languk(languk_evidence, languk_proof)

    record_ids = {row["record_id"] for row in base_rows}
    source_ids = {row["source_id"] for row in base_rows}
    payloads = {row["payload_sha256"] for row in base_rows}
    require(not any(row["record_id"] in record_ids for row in languk_rows), "Lang-UK record collision")
    require(not any(row["source_id"] in source_ids for row in languk_rows), "Lang-UK source collision")
    require(not any(row["payload_sha256"] in payloads for row in languk_rows), "Lang-UK payload replay")

    combined = sorted([dict(row) for row in base_rows] + [dict(row) for row in languk_rows], key=lambda row: row["record_id"])
    require(
        len(combined) == COMBINED_RECORDS
        and len({row["record_id"] for row in combined}) == COMBINED_RECORDS
        and len({row["source_id"] for row in combined}) == COMBINED_SOURCES
        and len({row["payload_sha256"] for row in combined}) == COMBINED_RECORDS,
        "combined cardinality/collision drift",
    )
    require(sum(row["payload_bytes"] for row in combined) == COMBINED_BYTES, "combined payload drift")
    payload_projection = [
        {"record_id": row["record_id"], "payload_sha256": row["payload_sha256"], "payload_bytes": row["payload_bytes"]}
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
        "base_receipt_identity_sha256": BASE_RECEIPT_ID,
        "base_dedup_identity_sha256": BASE_DEDUP_ID,
        "languk_head_sha": LANGUK_HEAD,
        "languk_evidence_identity_sha256": LANGUK_EVIDENCE_ID,
        "languk_two_clean_proof_identity_sha256": LANGUK_PROOF_ID,
        "base_records": BASE_RECORDS,
        "base_source_objects": BASE_SOURCES,
        "base_payload_bytes": BASE_BYTES,
        "languk_records": LANGUK_RECORDS,
        "languk_source_objects": LANGUK_SOURCES,
        "languk_payload_bytes": LANGUK_BYTES,
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
    dedup = {**dedup_core, "evidence_identity_sha256": sha256(dedup_core)}

    families.append({"family_id": LANGUK_FAMILY, "stratum": "ua", "unique_bytes": LANGUK_BYTES})
    families.sort(key=lambda row: row["family_id"])
    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-CURRENT-BALANCE-LANGUK-FAMILY-UNION-V1",
            "head_sha": execution_head,
            "evidence_identity_sha256": dedup["evidence_identity_sha256"],
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
        balance.get("status") == "TARGET_20M_SOURCE_MIX_FEASIBLE"
        and balance.get("maximum_feasible_total_source_bytes") == TARGET_TOTAL
        and balance.get("maximum_feasible_stratum_bytes") == TARGET_STRATA
        and balance.get("raw_capacity_by_stratum") == CAPACITY
        and balance.get("raw_gap_to_target_by_stratum") == {"ua": 0, "en": 0, "code": 0}
        and balance.get("family_minimum") == {"required_per_stratum": 2, "observed": FAMILY_COUNT, "pass": True},
        "balance result drift",
    )

    composition_core = {
        "schema": COMPOSITION_SCHEMA,
        "status": "COMPOSED_ZERO_CREDIT_TARGET_20M_SOURCE_MIX_FEASIBLE",
        "source_git_sha": execution_head,
        "parents": {
            "current_balance_rada_head_sha": BASE_HEAD,
            "base_composition_identity_sha256": BASE_COMPOSITION_ID,
            "base_receipt_identity_sha256": BASE_RECEIPT_ID,
            "languk_postdedup_clean_head_sha": LANGUK_HEAD,
            "languk_evidence_identity_sha256": LANGUK_EVIDENCE_ID,
            "languk_two_clean_proof_identity_sha256": LANGUK_PROOF_ID,
        },
        "combined_dedup_proof_identity_sha256": dedup["evidence_identity_sha256"],
        "combined_inventory": inventory,
        "stratum_capacity_bytes": {"uk": CAPACITY["ua"], "en": CAPACITY["en"], "code": CAPACITY["code"]},
        "stratum_family_counts": {"uk": FAMILY_COUNT["ua"], "en": FAMILY_COUNT["en"], "code": FAMILY_COUNT["code"]},
        "next_gate": balance["next_step"],
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
        "languk_evidence_identity_sha256": LANGUK_EVIDENCE_ID,
        "languk_two_clean_proof_identity_sha256": LANGUK_PROOF_ID,
        "combined_dedup_proof_identity_sha256": dedup["evidence_identity_sha256"],
        "composition_identity_sha256": composition["composition_identity_sha256"],
        "balance_policy_identity_sha256": POLICY_ID,
        "balance_result_identity_sha256": balance["result_identity_sha256"],
        "novel_languk_records": LANGUK_RECORDS,
        "novel_languk_source_objects": LANGUK_SOURCES,
        "novel_languk_payload_bytes": LANGUK_BYTES,
        "combined_record_count": COMBINED_RECORDS,
        "combined_source_object_count": COMBINED_SOURCES,
        "combined_total_payload_bytes": COMBINED_BYTES,
        "raw_capacity_by_stratum": CAPACITY,
        "family_minimum": balance["family_minimum"],
        "maximum_feasible_total_source_bytes": TARGET_TOTAL,
        "maximum_feasible_stratum_bytes": TARGET_STRATA,
        "remaining_policy_mix_shortfall_bytes": 0,
        "raw_gap_to_target_by_stratum": {"ua": 0, "en": 0, "code": 0},
        "balance_status": balance["status"],
        "next_scientific_gate": balance["next_step"],
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
        "combined-dedup-proof": dedup,
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
    parser.add_argument("--languk-evidence-json", type=Path, required=True)
    parser.add_argument("--languk-proof-json", type=Path, required=True)
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
            languk_evidence=load(args.languk_evidence_json),
            languk_proof=load(args.languk_proof_json),
            execution_head=args.execution_head,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            (args.output_dir / f"{name}.json").write_bytes(canonical(value) + b"\n")
    except (ExecutionError, gate.GateError, OSError, ValueError, StopIteration, KeyError, TypeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    receipt = result["execution-receipt"]
    print("D03_LANGUK_FAMILY_COMBINED_BALANCE=" + str(receipt["balance_status"]))
    print("COMBINED_TOTAL_PAYLOAD_BYTES=" + str(receipt["combined_total_payload_bytes"]))
    print("MAXIMUM_FEASIBLE_TOTAL_SOURCE_BYTES=" + str(receipt["maximum_feasible_total_source_bytes"]))
    print("REMAINING_POLICY_MIX_SHORTFALL_BYTES=" + str(receipt["remaining_policy_mix_shortfall_bytes"]))
    print("NEXT_SCIENTIFIC_GATE=" + str(receipt["next_scientific_gate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
