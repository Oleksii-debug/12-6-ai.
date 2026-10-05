#!/usr/bin/env python3
"""Execute D03 balance on qualified base+code, Franko and Ubuntu evidence.

Execution-only bridge: authenticate text-free authorities, prove pairwise exact
no-replay across their survivor inventories, project trusted family semantics, and
delegate mixture/family-cap mathematics to the existing NEXT100-106 gate.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tools.next100_106_balance_gate as gate
import tools.run_d03_current_plus_delta_balance_v1 as prior
from twelve_six.data.current_plus_delta_inventory_composition_v1 import (
    verify_current_clean_and_delta_composition,
)
from twelve_six.data import trusted_family_authority_franko_ubuntu as families

INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
BASE_HEAD = "d724db3873d33808361731d1f399f0d9a43079cf"
BASE_COMP = "8552fde3d551f773a82bc71684bf3b07a20c3e68885034c30e0488ee434c85f0"
BASE_DEDUP = "639c38b64ab3525a38bb8bf8eb9b30060c8891f4cf960836e826d1c3382dcb5f"
BASE_RECEIPT = "21bb9ab52212b6e9e4a3a4b6629280fc7d450cdd84de8062a286dfe5ba1f7f01"
BASE_COUNTS = (272, 259, 5_846_839)

FRANKO = {
    "label": "Franko",
    "schema": "12-6.d03-franko1901-decontam-g05-g06-execution.v1",
    "authority": "franko_authority",
    "family": "ua.verba.public-domain.franko1901",
    "modality": "uk",
    "truth": "franko",
    "head": "4280ac3a6b901906b0b38b5ff0ce0e74a90dd154",
    "identity": "09c350a1857fb6e728a6025840964253acbf39a3204c981b760f4373a3b51cb4",
    "parent_head": "5e9a9c27257824faf43915a8a75ef57f9ab6d341",
    "parent_survivor": "2da8566250e168659c280e0261f8e3b42a07618c260835946438f7abeee8156a",
    "parent_proof": "258a447dec49b2bb54d22af7f85458a51f786efe4f37c688ded9ca309614006c",
    "parent_matcher": "f02573a77ab649d03a7fe14353da4d9d67cde28699fc94059d7553d35c51955f",
    "pre_records": 30_636,
    "pre_bytes": 1_760_562,
    "records": 1_364,
    "bytes": 194_683,
}
UBUNTU = {
    "label": "Ubuntu",
    "schema": "12-6.d03-ubuntu-irc-decontam-g05-g06-execution.v1",
    "authority": "ubuntu_authority",
    "family": "common-pile/ubuntu_irc",
    "modality": "en",
    "truth": "ubuntu",
    "head": "fb9c3d9c137b0ccacf40dbee8d6dc75db7a2209f",
    "identity": "b7f7cd4ae377f667cb531ae095a96715a06afb0c24c3fe3c3080f243098fd22d",
    "parent_head": "45c1d93e8b3b6d1ea12d448583ce815f9705832e",
    "parent_survivor": "a4f531cbf2876676f852d8b4eae3816cba59e8b55234457e2ab8a04466fad7a7",
    "parent_proof": "09d58abe8a67f64e412401da7ebda2ec809e49213ace25b675e5924e8dfab54a",
    "parent_matcher": "1ab1c3f6b6ff3efa0e468809c791a46512043d655dc3056da1313433db0cfa65",
    "pre_records": 959,
    "pre_bytes": 4_775_057,
    "records": 897,
    "bytes": 4_651_278,
}
ZERO = {
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
ROW_KEYS = {"record_id", "source_id", "family", "modality", "payload_sha256", "payload_bytes"}


class ExecutionError(ValueError):
    pass


def req(value: bool, message: str) -> None:
    if not value:
        raise ExecutionError(message)


def self_hash(doc: Mapping[str, Any], field: str) -> str:
    body = copy.deepcopy(dict(doc))
    claimed = body.pop(field, None)
    req(isinstance(claimed, str) and len(claimed) == 64, f"{field} malformed")
    req(prior._sha256(body) == claimed, f"{field} self-hash mismatch")
    return claimed


def inventory(doc: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    req(doc.get("schema_version") == INVENTORY_SCHEMA, "inventory schema drift")
    rows = doc.get("records")
    req(isinstance(rows, list), "inventory rows missing")
    normalized: list[dict[str, Any]] = []
    previous = None
    for raw in rows:
        req(type(raw) is dict and set(raw) == ROW_KEYS, "inventory row schema drift")
        row = dict(raw)
        req(isinstance(row["record_id"], str) and row["record_id"], "record_id invalid")
        req(previous is None or row["record_id"] > previous, "record order drift")
        previous = row["record_id"]
        req(isinstance(row["source_id"], str) and row["source_id"], "source_id invalid")
        req(isinstance(row["family"], str) and row["family"], "family invalid")
        req(isinstance(row["modality"], str) and row["modality"], "modality invalid")
        req(isinstance(row["payload_sha256"], str) and len(row["payload_sha256"]) == 64, "payload SHA invalid")
        req(type(row["payload_bytes"]) is int and row["payload_bytes"] > 0, "payload bytes invalid")
        normalized.append(row)
    req(doc.get("record_count") == len(normalized), "inventory record count drift")
    req(doc.get("total_payload_bytes") == sum(r["payload_bytes"] for r in normalized), "inventory byte drift")
    payloads = [{"record_id": r["record_id"], "payload_sha256": r["payload_sha256"], "payload_bytes": r["payload_bytes"]} for r in normalized]
    req(doc.get("record_inventory_digest_sha256") == prior._sha256(normalized), "record root drift")
    req(doc.get("payload_inventory_digest_sha256") == prior._sha256(payloads), "payload root drift")
    req(len({r["record_id"] for r in normalized}) == len(normalized), "record replay")
    req(len({r["payload_sha256"] for r in normalized}) == len(normalized), "payload replay")
    if cfg:
        req(len(normalized) == cfg["records"], f"{cfg['label']} survivor count drift")
        req(doc["total_payload_bytes"] == cfg["bytes"], f"{cfg['label']} survivor bytes drift")
        req(all(r["family"] == cfg["family"] for r in normalized), f"{cfg['label']} family drift")
        req(all(r["modality"] == cfg["modality"] for r in normalized), f"{cfg['label']} modality drift")
        req(len({r["source_id"] for r in normalized}) == len(normalized), f"{cfg['label']} source replay")
    return normalized


def verify_extension(doc: dict[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    label = str(cfg["label"])
    req(doc.get("schema_version") == cfg["schema"], f"{label} schema drift")
    req(self_hash(doc, "evidence_identity_sha256") == cfg["identity"], f"{label} identity drift")
    req(doc.get("execution_head_sha") == cfg["head"], f"{label} head drift")
    req(doc.get("execution_profile") == "LOCAL_FREE", f"{label} profile drift")
    parent = doc.get("parent")
    req(isinstance(parent, Mapping), f"{label} parent missing")
    for field, expected in (
        ("execution_head_sha", cfg["parent_head"]),
        ("survivor_authority_sha256", cfg["parent_survivor"]),
        ("two_clean_proof_identity_sha256", cfg["parent_proof"]),
        ("matcher_report_sha256", cfg["parent_matcher"]),
    ):
        req(parent.get(field) == expected, f"{label} parent lineage drift: {field}")
    authority = doc.get(str(cfg["authority"]))
    req(isinstance(authority, Mapping), f"{label} authority missing")
    req(authority.get("source_family") == cfg["family"], f"{label} family authority drift")
    req(authority.get("source_object_count") == cfg["pre_records"], f"{label} pre-record drift")
    req(authority.get("pre_gate_payload_bytes") == cfg["pre_bytes"], f"{label} pre-byte drift")
    run = doc.get("gate_execution")
    req(isinstance(run, Mapping), f"{label} gate execution missing")
    req(run.get("input_records") == cfg["pre_records"], f"{label} gate input drift")
    req(run.get("survivor_records") == cfg["records"], f"{label} gate survivor drift")
    req(run.get("survivor_source_objects") == cfg["records"], f"{label} source count drift")
    req(run.get("survivor_payload_bytes") == cfg["bytes"], f"{label} survivor byte drift")
    req(run.get("later_gate_loss_bytes") == cfg["pre_bytes"] - cfg["bytes"], f"{label} loss arithmetic drift")
    rows = inventory(doc.get("survivor_inventory", {}), cfg)
    inv = doc["survivor_inventory"]
    req(run.get("survivor_record_inventory_digest_sha256") == inv["record_inventory_digest_sha256"], f"{label} record-root drift")
    req(run.get("survivor_payload_inventory_digest_sha256") == inv["payload_inventory_digest_sha256"], f"{label} payload-root drift")
    truth = doc.get("truth_boundary")
    req(isinstance(truth, Mapping), f"{label} truth missing")
    suffix = cfg["truth"]
    for field in (f"global_cross_source_dedup_complete_for_{suffix}", f"reserved_evaluation_decontamination_complete_for_{suffix}", f"canonical_quality_privacy_complete_for_{suffix}"):
        req(truth.get(field) is True, f"{label} incomplete: {field}")
    for field in ("balance_diversity_retest_complete", "family_caps_complete", "cluster_safe_split_complete", "deterministic_pack_two_clean_complete", "positive_exact_unique_loss_ledger", "tokenizer_fit_authorized", "training_executed", "learned_weights_created", "final_test_outcomes_read", "paid_compute_used"):
        req(truth.get(field) is False, f"{label} truth widened: {field}")
    for field in ("canonical_capacity_credited", "training_authorized_bytes", "authorized_unique_loss_positions", "authorized_optimized_target_exposure", "optimizer_updates_executed_on_real_targets"):
        req(type(truth.get(field)) is int and truth[field] == 0, f"{label} truth widened: {field}")
    return rows


def execute(base_comp: dict[str, Any], base_dedup: dict[str, Any], base_receipt: dict[str, Any], franko: dict[str, Any], ubuntu: dict[str, Any], head: str) -> dict[str, dict[str, Any]]:
    req(len(head) == 40 and all(c in "0123456789abcdef" for c in head), "execution head invalid")
    verify_current_clean_and_delta_composition(base_comp, expected_composition_identity_sha256=BASE_COMP)
    req(base_comp.get("source_git_sha") == BASE_HEAD, "base composition head drift")
    base_rows = inventory(base_comp["combined_inventory"])
    req((len(base_rows), len({r["source_id"] for r in base_rows}), sum(r["payload_bytes"] for r in base_rows)) == BASE_COUNTS, "base physical counts drift")
    req(base_dedup.get("schema") == "12-6.d03-current-plus-delta-global-unique-proof.v1", "base dedup schema drift")
    req(self_hash(base_dedup, "evidence_identity_sha256") == BASE_DEDUP, "base dedup identity drift")
    req(base_dedup.get("terminal_verdict") == "PASS", "base dedup nonterminal")
    req(base_receipt.get("schema") == "12-6.d03-current-plus-delta-balance-execution.v1", "base receipt schema drift")
    req(self_hash(base_receipt, "receipt_identity_sha256") == BASE_RECEIPT, "base receipt identity drift")
    req(base_receipt.get("combined_dedup_proof_identity_sha256") == BASE_DEDUP, "base receipt dedup drift")
    req(base_receipt.get("balance_policy_identity_sha256") == POLICY_ID, "base policy drift")
    franko_rows = verify_extension(franko, FRANKO)
    ubuntu_rows = verify_extension(ubuntu, UBUNTU)

    parts = {"base_code": base_rows, "franko": franko_rows, "ubuntu": ubuntu_rows}
    source_sets: list[set[str]] = []
    all_rows: list[dict[str, Any]] = []
    membership: list[dict[str, str]] = []
    seen_records: set[str] = set()
    seen_payloads: set[str] = set()
    for origin, rows in parts.items():
        sources = {r["source_id"] for r in rows}
        req(all(not (sources & prior_sources) for prior_sources in source_sets), "cross-inventory source collision")
        source_sets.append(sources)
        for row in rows:
            req(row["record_id"] not in seen_records, "cross-inventory record collision")
            req(row["payload_sha256"] not in seen_payloads, "cross-inventory payload collision")
            seen_records.add(row["record_id"])
            seen_payloads.add(row["payload_sha256"])
            all_rows.append(dict(row))
            membership.append({"record_id": row["record_id"], "origin": origin})
    all_rows.sort(key=lambda r: r["record_id"])
    membership.sort(key=lambda r: r["record_id"])
    record_count = len(all_rows)
    source_count = sum(len(s) for s in source_sets)
    total_bytes = sum(r["payload_bytes"] for r in all_rows)
    req((record_count, source_count, total_bytes) == (2533, 2520, 10_692_800), "combined physical arithmetic drift")
    payloads = [{"record_id": r["record_id"], "payload_sha256": r["payload_sha256"], "payload_bytes": r["payload_bytes"]} for r in all_rows]
    combined_inventory = {"schema_version": INVENTORY_SCHEMA, "record_count": record_count, "total_payload_bytes": total_bytes, "record_inventory_digest_sha256": prior._sha256(all_rows), "payload_inventory_digest_sha256": prior._sha256(payloads), "records": all_rows}

    capacities: dict[str, int] = defaultdict(int)
    for row in all_rows:
        capacities[row["family"]] += row["payload_bytes"]
    trusted = {r["family"]: r for r in families.trusted_family_projection(capacities)}
    req(set(trusted) == set(capacities), "trusted family coverage drift")
    family_rows = [{"family": family, "stratum": trusted[family]["stratum"], "source_family_identity_sha256": trusted[family]["source_family_identity_sha256"], "capacity_bytes": capacities[family]} for family in sorted(capacities)]
    family_root = families.trusted_family_authority_root_sha256(capacities)
    strata = {"uk": 0, "en": 0, "code": 0}
    counts = {"uk": 0, "en": 0, "code": 0}
    for row in family_rows:
        strata[row["stratum"]] += row["capacity_bytes"]
        counts[row["stratum"]] += 1

    comp_core = {"schema": "12-6.d03-current-code-franko-ubuntu-composition.v1", "status": "COMPOSED_ZERO_CREDIT_PENDING_BALANCE", "source_git_sha": head, "parents": {"base_composition_identity_sha256": BASE_COMP, "base_dedup_identity_sha256": BASE_DEDUP, "base_balance_receipt_identity_sha256": BASE_RECEIPT, "franko_execution_head_sha": FRANKO["head"], "franko_evidence_identity_sha256": FRANKO["identity"], "ubuntu_execution_head_sha": UBUNTU["head"], "ubuntu_evidence_identity_sha256": UBUNTU["identity"]}, "combined_inventory": combined_inventory, "inventory_membership": membership, "cross_inventory_record_id_collision_free": True, "cross_inventory_source_id_collision_free": True, "cross_inventory_exact_payload_collision_free": True, "record_membership_sha256": prior._sha256(membership), "trusted_family_authority_root_sha256": family_root, "families": family_rows, "stratum_capacity_bytes": strata, "stratum_family_counts": counts, "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP", **ZERO}
    composition = {**comp_core, "composition_identity_sha256": prior._sha256(comp_core)}
    dedup_core = {"schema": "12-6.d03-current-code-franko-ubuntu-global-unique-proof.v1", "base_terminal_dedup_identity_sha256": BASE_DEDUP, "franko_incumbent_global_dedup": {"parent_execution_head_sha": FRANKO["parent_head"], "parent_survivor_authority_sha256": FRANKO["parent_survivor"], "parent_two_clean_proof_identity_sha256": FRANKO["parent_proof"], "parent_matcher_report_sha256": FRANKO["parent_matcher"], "post_qp_evidence_identity_sha256": FRANKO["identity"], "terminal_verdict": "PASS"}, "ubuntu_incumbent_global_dedup": {"parent_execution_head_sha": UBUNTU["parent_head"], "parent_survivor_authority_sha256": UBUNTU["parent_survivor"], "parent_two_clean_proof_identity_sha256": UBUNTU["parent_proof"], "parent_matcher_report_sha256": UBUNTU["parent_matcher"], "post_qp_evidence_identity_sha256": UBUNTU["identity"], "terminal_verdict": "PASS"}, "composition_identity_sha256": composition["composition_identity_sha256"], "cross_inventory_record_id_collision_free": True, "cross_inventory_source_id_collision_free": True, "cross_inventory_exact_payload_collision_free": True, "semantics": "BASE_TERMINAL_PLUS_INDEPENDENT_INCUMBENT_GLOBAL_DEDUP_EXTENSIONS_PLUS_PAIRWISE_EXACT_COLLISION_PROOF_NO_REPLAY", "terminal_verdict": "PASS"}
    dedup = {**dedup_core, "evidence_identity_sha256": prior._sha256(dedup_core)}

    vector_families = []
    by_stratum = {"ua": 0, "en": 0, "code": 0}
    family_count = {"ua": 0, "en": 0, "code": 0}
    for row in family_rows:
        stratum = {"uk": "ua", "en": "en", "code": "code"}[row["stratum"]]
        vector_families.append({"family_id": row["family"], "stratum": stratum, "unique_bytes": row["capacity_bytes"]})
        by_stratum[stratum] += row["capacity_bytes"]
        family_count[stratum] += 1
    vector = {"schema_version": gate.INPUT_SCHEMA, "terminal": True, "dedup_authority": {"worker_id": "D03-CURRENT-CODE-FRANKO-UBUNTU-COMPOSITION-V1", "head_sha": head, "evidence_identity_sha256": dedup["evidence_identity_sha256"], "terminal_verdict": "PASS"}, "families": sorted(vector_families, key=lambda r: r["family_id"]), "totals": {"total_unique_bytes": sum(by_stratum.values()), "by_stratum": by_stratum, "family_count": family_count}, "physical_authority": {"composition_identity_sha256": composition["composition_identity_sha256"], "record_inventory_digest_sha256": combined_inventory["record_inventory_digest_sha256"], "payload_inventory_digest_sha256": combined_inventory["payload_inventory_digest_sha256"], "record_count": record_count, "source_object_count": source_count, "total_payload_bytes": total_bytes, "trusted_family_authority_root_sha256": family_root}}
    gate.validate_vector(vector)
    policy = gate.load_json(gate.POLICY_PATH)
    gate.validate_policy(policy)
    req(policy.get("policy_identity_sha256") == POLICY_ID, "canonical policy drift")
    balance = gate.evaluate(policy, vector)
    status = balance["status"]
    receipt_core = {"schema": "12-6.d03-current-code-franko-ubuntu-balance-execution.v1", "execution_profile": "LOCAL_FREE", "execution_head_sha": head, "composition_identity_sha256": composition["composition_identity_sha256"], "combined_dedup_proof_identity_sha256": dedup["evidence_identity_sha256"], "base_balance_receipt_identity_sha256": BASE_RECEIPT, "franko_evidence_identity_sha256": FRANKO["identity"], "ubuntu_evidence_identity_sha256": UBUNTU["identity"], "balance_policy_identity_sha256": POLICY_ID, "balance_result_identity_sha256": balance["result_identity_sha256"], "balance_status": status, "maximum_feasible_total_source_bytes": balance["maximum_feasible_total_source_bytes"], "maximum_feasible_stratum_bytes": balance["maximum_feasible_stratum_bytes"], "raw_capacity_by_stratum": balance["raw_capacity_by_stratum"], "raw_gap_to_target_by_stratum": balance["raw_gap_to_target_by_stratum"], "family_minimum": balance["family_minimum"], "combined_record_count": record_count, "combined_source_object_count": source_count, "combined_total_payload_bytes": total_bytes, "next_scientific_gate": "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACK" if status == "TARGET_20M_SOURCE_MIX_FEASIBLE" else "ACQUIRE_MORE_DIVERSE_LAWFUL_SOURCE_CAPACITY", **ZERO}
    receipt = {**receipt_core, "receipt_identity_sha256": prior._sha256(receipt_core)}
    return {"composition": composition, "combined-dedup-proof": dedup, "next100-input": vector, "balance-result": balance, "execution-receipt": receipt}


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(prior._canonical(value) + b"\n")


def main() -> int:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--base-composition-json", type=Path, required=True)
    p.add_argument("--base-dedup-json", type=Path, required=True)
    p.add_argument("--base-receipt-json", type=Path, required=True)
    p.add_argument("--franko-evidence-json", type=Path, required=True)
    p.add_argument("--ubuntu-evidence-json", type=Path, required=True)
    p.add_argument("--source-git-sha", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result = execute(prior._load(a.base_composition_json), prior._load(a.base_dedup_json), prior._load(a.base_receipt_json), prior._load(a.franko_evidence_json), prior._load(a.ubuntu_evidence_json), a.source_git_sha)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        write(a.output_dir / f"{name}.json", value)
    r = result["execution-receipt"]
    print("D03_CURRENT_CODE_FRANKO_UBUNTU_BALANCE=" + r["balance_status"])
    print("MAXIMUM_FEASIBLE_TOTAL_SOURCE_BYTES=" + str(r["maximum_feasible_total_source_bytes"]))
    print("COMBINED_TOTAL_PAYLOAD_BYTES=" + str(r["combined_total_payload_bytes"]))
    print("RECEIPT_IDENTITY_SHA256=" + r["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
