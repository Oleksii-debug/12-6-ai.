#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-102 Research Corpus V1 intake."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_AUTHORITY_ID = "6922c8c1b0fcfed032413af0469207d1dcf0a6e30ebda49a79765206a496a193"
EXPECTED_INVENTORY_ID = "c952562878598ab81f28e4c734a0fd24cafb1101859acd8c4a675b069a011fa4"
EXPECTED_PARENT_HEAD = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
EXPECTED_CONTROL_HEAD = "83d92d50a4380636cc7f1cd41fa9ffd4445dd12e"
EXPECTED_EVAL_HEAD = "5e5a1de3b594cee5612e63d3d4c2a70499740ac7"
EXPECTED_UA_AUTHORITY = "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6"
EXPECTED_PY_AUTHORITY = "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d"
EXPECTED_UA_HASH = "65e570c3cd954b595b586554b89a90da6efad0deca6a84d2316937745db17ef2"
EXPECTED_PARENT_HASHES = {
    "154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83",
    "94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a",
    "72c301db0b2539f3f7a73c9c15e2e425700a6b758a1114f1a861e2d60c704c50",
    "2c61b3ac94d1dcebcde0c6f519554d2d7917247fbaa0a97002db4ef69e70ff28",
    "4c7d8d132c9898fc7d715e473f3ac74785ddc4ab96d2c9240f87835dc6d981ff",
}
EXPECTED_PY_HASHES = [
    "604c7243b37696ca7173dbdf9dd2b5663f54075590e599b741243f882739e869",
    "65eff0ae0fbb474ee32bc02f12111f57775474d1b6375c3dd0894ec666d05267",
    "49771a2eedc6cae523e10e7fa04feac4e79713e324d7fe4d37577689ded1b279",
    "1998f1a712c71a4cdec3d6f9693b0de79451004fcc5bdc78149d7d7a180427af",
    "647bda8d594fec2c8be35b0505986f9f5e885d2fba8692838cb6f48494b8bc04",
    "68bb8862c2b25dc702412cc1470410f7a218498dbbcdfb52a4db3269bd973d26",
    "ab11186bb4c1fb0ddc849693cff0b4f230b162dc41d6908d6807f1f0f86ac6c2",
    "aab52bdb384c5503d34d524edb06cacdc0c7ea03385c5413bd1db29cb7f7f388",
    "bb78d8aad6fcd8ccf9408bd64a998aebedff14804aa79f78421cdd1fcdd91fef",
    "a19741280be37f9268367aa31e8c3eedc1876b839b9aca4d13299f2872153b55",
    "c9e8a1dc6709e6e50dadc9f22269f3ba149c2c4ba0f1b53ae32f0811acab79cb",
    "e67ce9871a098147df5caa26900518e61422ce789252a766ae046a6eb4fde742",
    "3b86d261ef94dd7b0deb0c577faaa41b9026f50cd18abee6c5eb84aa5aeb38ee",
    "a09756447fdbd535629939d1bcaf8db5f6fba4b23bdc9468e27625f67c11e470",
]


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-102 validation failed: {message}")


def validate(path: Path) -> dict[str, object]:
    authority = json.loads(path.read_text(encoding="utf-8"))
    claimed = authority.pop("authority_identity_sha256", None)
    require(claimed == EXPECTED_AUTHORITY_ID, "authority identity drift")
    require(hashlib.sha256(canonical(authority)).hexdigest() == claimed, "authority self-hash mismatch")
    require(authority["schema_version"] == "12-6.next100-102-research-corpus-v1-intake.v1", "schema drift")
    require(authority["worker_id"] == "NEXT100-102-RESEARCH-CORPUS-V1-INTAKE", "worker drift")
    require(authority["execution_profile"] == "LOCAL_FREE", "execution profile weakened")

    require(authority["control_authority"]["head_sha"] == EXPECTED_CONTROL_HEAD, "controller head drift")
    require(authority["parent_data301"]["head_sha"] == EXPECTED_PARENT_HEAD, "DATA-301 head drift")
    require(authority["parent_data301"]["state"] == "TERMINAL_BLOCKED", "DATA-301 blocker erased")
    require(authority["parent_data301"]["authorized_balanced_no_replay_capacity"] == 0, "parent capacity fabricated")
    require(authority["selection_validation"]["head_sha"] == EXPECTED_EVAL_HEAD, "EVAL-303 head drift")
    require(authority["selection_validation"]["records"] == {"ua": 8, "en": 2, "code": 0}, "selection vector drift")
    require(authority["selection_validation"]["final_test_consumed"] is False, "final-test isolation broken")

    src = authority["source_authorities"]
    require(src["ua_wikisource"]["authority_identity_sha256"] == EXPECTED_UA_AUTHORITY, "UA authority drift")
    require(src["ua_wikisource"]["training_permission"] == "ALLOWED", "UA training permission drift")
    require(src["python_docs"]["authority_identity_sha256"] == EXPECTED_PY_AUTHORITY, "Python authority drift")
    require(src["python_docs"]["accepted_chunk_count"] == 14, "Python accepted chunk count drift")
    require(src["python_docs"]["rejected_chunk_count"] == 2, "Python rejected chunk count drift")
    require(src["python_docs"]["training_permission"] == "ALLOWED", "Python training permission drift")

    inv = authority["candidate_inventory"]
    require(hashlib.sha256(canonical(inv)).hexdigest() == authority["candidate_inventory_identity_sha256"], "candidate inventory hash mismatch")
    require(authority["candidate_inventory_identity_sha256"] == EXPECTED_INVENTORY_ID, "candidate inventory identity drift")
    require(inv["logical_record_count"] == 20, "record count drift")
    require(inv["source_family_count"] == 6, "source family count drift")
    require(inv["family_counts_by_stratum"] == {"uk": 2, "en": 2, "code": 2}, "stratum family count drift")
    require(inv["exact_training_candidate_bytes"] is None, "unverified exact byte capacity fabricated")

    parent = inv["parent_records"]
    require(len(parent) == 5, "parent record count drift")
    require({r["content_sha256"] for r in parent} == EXPECTED_PARENT_HASHES, "parent content identity drift")
    successor = inv["successor_records"]
    require(len(successor) == 15, "successor record count drift")
    ua = [r for r in successor if r["family"] == "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv"]
    require(len(ua) == 1 and ua[0]["content_sha256"] == EXPECTED_UA_HASH, "UA successor identity drift")
    py = [r for r in successor if r["family"] == "python.cpython.documentation"]
    require([r["content_sha256"] for r in py] == EXPECTED_PY_HASHES, "Python accepted chunk identity/order drift")
    all_hashes = [r["content_sha256"] for r in parent + successor]
    require(len(all_hashes) == len(set(all_hashes)), "exact content hash duplicate introduced")

    material = authority["materialization"]
    require(material["single_tree_exact_content_materialized"] is False, "materialization overclaimed")
    gates = authority["gate_state"]
    require(gates["G03_SOURCE_INVENTORY"] == "PASS_IDENTITY_FROZEN_PRE_DECONTAMINATION", "inventory gate drift")
    require(gates["G08_RESERVED_DECONTAMINATION"] == "BLOCKED_UNTIL_EXACT_CONTENT_MATERIALIZATION", "decontamination overclaimed")
    require(gates["G09_BALANCE_DIVERSITY"] == "STRUCTURAL_MINIMUM_FAMILY_COUNT_REACHED_CAPACITY_AND_CAPS_UNPROVEN", "balance gate overclaimed")
    require(gates["G12_UNIQUE_LOSS"] == "BLOCKED_NO_POST_SPLIT_POST_PACK_LEDGER", "loss ledger overclaimed")

    training = authority["training_authority"]
    require(training["long_training_authorized"] is False, "long training silently authorized")
    require(training["training_executed"] is False, "training falsely recorded")
    require(training["authorized_unique_optimized_targets"] == 0, "optimized target capacity fabricated")
    require(training["paid_compute_used"] is False, "paid compute boundary drift")

    release = authority["release_state"]
    require(release["pre_decontamination_candidate_identity_frozen"] is True, "intake identity not frozen")
    for key in ("corpus_frozen", "release_ready", "terminal_corpus"):
        require(release[key] is False, f"release state overclaimed: {key}")
    require(release["corpus_identity"] is None and release["shard_identity"] is None, "corpus/shard identity fabricated")

    return {
        "status": "PASS",
        "authority_identity_sha256": claimed,
        "candidate_inventory_identity_sha256": authority["candidate_inventory_identity_sha256"],
        "logical_record_count": inv["logical_record_count"],
        "family_counts_by_stratum": inv["family_counts_by_stratum"],
        "authorized_unique_optimized_targets": 0,
        "next_gate": "EXACT_CONTENT_MATERIALIZATION_THEN_DECONTAMINATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="configs/data/next100_102_research_corpus_v1_intake.json")
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.path)), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
