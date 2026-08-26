#!/usr/bin/env python3
"""Fail-closed validator for the live NEXT100-063 -> 20M data convergence delta."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/control/20m_data_convergence_delta_v1.json"

EXPECTED_SCHEMA = "12-6.20m-data-convergence-delta.v1"
EXPECTED_REGISTRY_HEAD = "94dce83cbe611144f961b9f93b3be273345a7f62"
EXPECTED_REGISTRY_ID = "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"
EXPECTED_PARENT_ID = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
EXPECTED_TARGET_BYTES = 20_000_000
EXPECTED_CANDIDATE_BYTES = 565_743
EXPECTED_BY_STRATUM = {
    "ua": {"normalized_bytes": 100_856, "family_count": 4},
    "en": {"normalized_bytes": 168_544, "family_count": 4},
    "code": {"normalized_bytes": 296_343, "family_count": 5},
}


class ConvergenceDeltaError(ValueError):
    """Raised when live 20M data-convergence evidence is inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConvergenceDeltaError(message)


def validate(value: dict[str, Any]) -> None:
    require(value.get("schema") == EXPECTED_SCHEMA, "unexpected schema")
    require(value.get("execution_profile") == "LOCAL_FREE", "execution must remain LOCAL_FREE")

    registry = value["source_registry"]
    require(registry["pr"] == 538, "unexpected source-registry PR")
    require(registry["head_sha"] == EXPECTED_REGISTRY_HEAD, "unexpected source-registry head")
    require(registry["registry_identity_sha256"] == EXPECTED_REGISTRY_ID, "unexpected registry identity")
    require(registry["parent_registry_identity_sha256"] == EXPECTED_PARENT_ID, "unexpected DATA-287 parent identity")
    require(registry["terminal_addition_count"] == 9, "terminal addition count drift")
    require(registry["candidate_source_authority_count"] == 14, "candidate source count drift")
    require(registry["candidate_normalized_bytes"] == EXPECTED_CANDIDATE_BYTES, "candidate-byte accounting drift")
    require(registry["target_normalized_bytes"] == EXPECTED_TARGET_BYTES, "Research Corpus V1 target drift")
    require(
        registry["target_gap_normalized_bytes"]
        == EXPECTED_TARGET_BYTES - EXPECTED_CANDIDATE_BYTES,
        "target-gap arithmetic drift",
    )
    require(registry["by_stratum"] == EXPECTED_BY_STRATUM, "stratum accounting drift")
    require(
        sum(v["normalized_bytes"] for v in registry["by_stratum"].values())
        == EXPECTED_CANDIDATE_BYTES,
        "stratum bytes do not sum to candidate bytes",
    )
    require(
        sum(v["family_count"] for v in registry["by_stratum"].values())
        == registry["independent_family_count"],
        "family accounting drift",
    )
    require(
        all(v["family_count"] >= 2 for v in registry["by_stratum"].values()),
        "minimum family diversity regressed",
    )
    require(registry["family_minimum_gate"] == "PASS_PRE_GLOBAL_DEDUP", "family gate drift")

    inventory = value["record_inventory_gate"]
    require(
        inventory["required_parent_registry_identity_sha256"] == EXPECTED_REGISTRY_ID,
        "record inventory is not bound to NEXT100-063",
    )
    require(inventory["training_authorized"] is False, "record inventory cannot authorize training")
    require(inventory["duplicate_normalized_content_forbidden"] is True, "duplicate content must fail closed")
    require(
        inventory["must_cover_every_counted_authority_or_explicitly_defer_with_reason"] is True,
        "authority coverage rule missing",
    )

    decontam = value["decontamination_gate"]
    require(decontam["next100_066_scan_executed"] is False, "decontamination is prematurely marked executed")
    require(decontam["exact_and_near_match_required"] is True, "near-match decontamination was weakened")
    require(decontam["training_authorized"] is False, "decontamination gate cannot authorize training")

    truth = value["truth_boundary"]
    require(truth["final_corpus_identity"] is None, "final corpus identity fabricated")
    require(truth["shard_identity"] is None, "shard identity fabricated")
    require(truth["authorized_unique_loss_positions"] == 0, "nonzero training exposure fabricated")
    require(truth["long_training_authorized"] is False, "long training unexpectedly authorized")
    require(truth["long_training_executed"] is False, "long training unexpectedly claimed")
    require(truth["paid_compute_authorized"] is False, "paid compute unexpectedly authorized")
    require(truth["paid_compute_used"] is False, "paid compute unexpectedly claimed")
    require(truth["learned_20m_claimed"] is False, "learned 20M claim is premature")

    actions = value["ordered_next_actions"]
    require(actions[0] == "FREEZE_EXACT_RECORD_INVENTORY_BOUND_TO_NEXT100_063_REGISTRY_IDENTITY", "wrong next action")
    require(actions[-1] == "REQUEST_MATERIAL_COMPUTE_AUTHORIZATION_ONLY_AFTER_ALL_DATA_GATES_PASS", "compute authorization moved before data gates")
    require(len(actions) == len(set(actions)), "duplicate ordered next actions")


def main() -> int:
    value = json.loads(CONTROL.read_text(encoding="utf-8"))
    validate(value)
    print("20M_DATA_CONVERGENCE_DELTA=PASS")
    print("SOURCE_REGISTRY_SHA256=" + EXPECTED_REGISTRY_ID)
    print("CANDIDATE_NORMALIZED_BYTES=" + str(EXPECTED_CANDIDATE_BYTES))
    print("TARGET_GAP_NORMALIZED_BYTES=" + str(EXPECTED_TARGET_BYTES - EXPECTED_CANDIDATE_BYTES))
    print("AUTHORIZED_UNIQUE_LOSS_POSITIONS=0")
    print("NEXT=" + value["ordered_next_actions"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
