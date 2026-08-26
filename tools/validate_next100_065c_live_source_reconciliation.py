#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "12-6.next100-065c-live-source-reconciliation.v1"
WORKER_ID = "AUTODEV-NEXT100-065C-LIVE-SOURCE-RECONCILIATION"
REGISTRY_V3_IDENTITY = "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c"
GUTENBERG_IDENTITY = "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b"
EXPECTED_VECTOR = {"uk": 100856, "en": 1838293, "code": 106031, "total": 2045180}


class ReconciliationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def validate(data: Mapping[str, Any]) -> dict[str, Any]:
    _require(data.get("schema_version") == SCHEMA, "schema drift")
    _require(data.get("worker_id") == WORKER_ID, "worker id drift")
    _require(data.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    _require(data.get("local_free_only") is True, "LOCAL_FREE boundary weakened")

    parent = _mapping(data.get("parent_dedup_snapshot"), "parent_dedup_snapshot")
    _require(parent.get("pr") == 632, "dedup parent PR drift")
    _require(
        parent.get("role") == "EXECUTABLE_DEDUP_LANE_NOT_TERMINAL_SOURCE_AUTHORITY",
        "parent role drift",
    )
    _require(parent.get("terminal_promotion_allowed") is False, "stale #632 promotion enabled")

    registry = _mapping(data.get("registry_v3"), "registry_v3")
    _require(registry.get("pr") == 538, "registry PR drift")
    _require(
        registry.get("schema") == "12-6.next100-063-terminal-source-registry.v3",
        "registry schema drift",
    )
    _require(registry.get("identity_sha256") == REGISTRY_V3_IDENTITY, "registry identity drift")
    base = _mapping(registry.get("numeric_capacity_bytes"), "registry numeric capacity")
    _require(
        base == {"uk": 100856, "en": 150643, "code": 106031, "total": 357530},
        "registry capacity drift",
    )
    _require(registry.get("independent_families") == 13, "registry family count drift")
    _require(
        registry.get("authorized_unique_causal_loss_positions") == 0,
        "registry exposure must remain zero",
    )

    cpython = _mapping(data.get("accepted_only_cpython"), "accepted_only_cpython")
    _require(cpython.get("pr") == 567, "CPython authority PR drift")
    _require(cpython.get("dedicated_workflow_run") == 33005689174, "CPython run drift")
    _require(
        cpython.get("dedicated_workflow_conclusion") == "success",
        "CPython authority not green",
    )
    _require(cpython.get("accepted_chunk_count") == 14, "CPython accepted-count drift")
    _require(cpython.get("rejected_chunk_count") == 2, "CPython rejected-count drift")
    _require(
        cpython.get("rejection_reasons") == {"pii_phone": 2},
        "CPython privacy rejection drift",
    )
    _require(cpython.get("eligible_capacity_bytes") == 15540, "CPython accepted capacity drift")
    _require(cpython.get("family_credit_delta") == 0, "CPython family double-credit")

    gutenberg = _mapping(data.get("gutenberg_terminal"), "gutenberg_terminal")
    _require(gutenberg.get("pr") == 627, "Gutenberg authority PR drift")
    _require(
        gutenberg.get("authority_identity_sha256") == GUTENBERG_IDENTITY,
        "Gutenberg identity drift",
    )
    _require(gutenberg.get("parent_workflow_run") == 32998859164, "Gutenberg run drift")
    _require(
        gutenberg.get("parent_workflow_conclusion") == "success",
        "Gutenberg source not green",
    )
    _require(gutenberg.get("normalized_utf8_bytes") == 1672110, "Gutenberg capacity drift")
    _require(gutenberg.get("record_count") == 3, "Gutenberg record count drift")
    _require(gutenberg.get("independent_family_credit") == 1, "Gutenberg family credit drift")
    _require(gutenberg.get("evaluation") == "NOT_AUTHORIZED", "Gutenberg evaluation boundary weakened")

    successor = _mapping(
        data.get("minimum_required_successor_vector_before_global_dedup"),
        "minimum_required_successor_vector_before_global_dedup",
    )
    vector = _mapping(successor.get("numeric_capacity_bytes"), "successor numeric capacity")
    _require(vector == EXPECTED_VECTOR, "successor vector drift")
    _require(
        vector["uk"] + vector["en"] + vector["code"] == vector["total"],
        "successor arithmetic mismatch",
    )
    _require(
        vector["total"]
        == base["total"] + cpython["eligible_capacity_bytes"] + gutenberg["normalized_utf8_bytes"],
        "successor total does not reconcile live authorities",
    )
    _require(
        vector["en"]
        == base["en"] + cpython["eligible_capacity_bytes"] + gutenberg["normalized_utf8_bytes"],
        "successor English capacity does not reconcile live authorities",
    )
    _require(successor.get("independent_families") == 14, "successor family count drift")

    gate = _mapping(data.get("promotion_gate"), "promotion_gate")
    proof_flags = (
        "full_live_object_graph_materialized",
        "successor_global_exact_near_fragment_lineage_dedup_passed",
        "immutable_record_inventory_frozen",
        "reserved_evaluation_decontamination_passed",
        "post_composition_quality_privacy_passed",
        "cluster_safe_split_and_deterministic_pack_passed",
        "two_clean_builds_byte_identical",
        "unique_causal_loss_ledger_published",
        "long_training_authorized",
        "paid_compute_authorized",
    )
    for key in proof_flags:
        _require(gate.get(key) is False, f"unproven promotion flag enabled: {key}")
    _require(
        gate.get("authorized_unique_causal_loss_positions") == 0,
        "training exposure must remain zero",
    )
    _require(
        gate.get("decision") == "BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING",
        "control decision weakened",
    )

    return {
        "status": "PASS",
        "required_pre_global_dedup_bytes": vector["total"],
        "required_independent_families": successor["independent_families"],
        "authorized_unique_causal_loss_positions": 0,
        "decision": gate["decision"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data/next100_065c_live_source_reconciliation_v1.json",
    )
    args = parser.parse_args()
    value = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("config root must be an object")
    try:
        result = validate(value)
    except ReconciliationError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
