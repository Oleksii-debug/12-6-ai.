from __future__ import annotations

import copy
import hashlib
import json

import pytest

import tools.next100_106_balance_gate as next100_gate
from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postmaterialization_balance_projection_v1 import (
    BALANCE_BINDING_SCHEMA,
    FAMILY_VECTOR_SCHEMA,
    adapt_postmaterialization_family_vector_to_next100_106,
    build_balance_result_binding,
    build_postmaterialization_family_vector,
    require_balanced_selection_ready,
    verify_postmaterialization_family_vector,
)

GIT_SHA = "2" * 40
MATERIALIZATION_SHA = "3" * 64
JSONL_SHA = "4" * 64
DEDUP_HEAD_SHA = "5" * 40
DEDUP_EVIDENCE_SHA = "6" * 64
DEDUP_WORKER = "NEXT100-065F-CURRENT-MAIN-GLOBAL-DEDUP-V8"

UA_A = "ua.kmu.portal.secretariat-news"
UA_B = "ua.verba.public-domain.nomis1864"
EN_A = "en.mdn.webdocs.prose"
EN_B = "en.project-gutenberg.public-domain-books"
CODE_A = "github:agronholm/anyio"
CODE_B = "github:pytest-dev/pytest"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _row(
    record_id: str,
    family: str,
    modality: str,
    payload_bytes: int,
    *,
    source_id: str | None = None,
) -> dict:
    return {
        "record_id": record_id,
        "source_id": source_id or record_id,
        "family": family,
        "modality": modality,
        "payload_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
        "payload_bytes": payload_bytes,
    }


def _inventory(rows: list[dict]) -> dict:
    rows = sorted(copy.deepcopy(rows), key=lambda item: item["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    return {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "record_inventory_digest_sha256": hashlib.sha256(
            _canonical(rows)
        ).hexdigest(),
        "payload_inventory_digest_sha256": hashlib.sha256(
            _canonical(payload_projection)
        ).hexdigest(),
        "records": rows,
    }


def _truth() -> dict:
    return {
        "authorized_optimized_target_exposure": 0,
        "authorized_unique_loss_positions": 0,
        "current_corpus_eligible": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
        "final_test_outcomes_read": False,
        "foreign_pretrained_weights": False,
        "learned_weights_created": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "paid_compute_used": False,
        "tokenizer_fit_authorized": False,
        "training_authorized_bytes": 0,
        "training_executed": False,
    }


def _evidence(inventory: dict) -> dict:
    return {
        "schema_version": "12-6.d03-post-g05-g06-materialization.v1",
        "status": "MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": GIT_SHA,
        "materialization_identity_sha256": MATERIALIZATION_SHA,
        "materializer_implementation_git_blob_sha1": "7" * 40,
        "repeat_materialization_byte_identical": True,
        "remaining_materialization_blockers": [],
        "input": {},
        "result": {
            "record_payload_jsonl_sha256": JSONL_SHA,
            "record_count": inventory["record_count"],
            "total_payload_bytes": inventory["total_payload_bytes"],
            "source_object_count": len(
                {row["source_id"] for row in inventory["records"]}
            ),
            "record_inventory_digest_sha256": inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": inventory[
                "payload_inventory_digest_sha256"
            ],
        },
        "transform": {},
        "consumed_blockers": [],
        "truth_boundary": _truth(),
    }


def _build(rows: list[dict]) -> tuple[dict, dict, dict]:
    inventory = _inventory(rows)
    evidence = _evidence(inventory)
    vector = build_postmaterialization_family_vector(
        inventory=inventory,
        materialization_evidence=evidence,
        expected_execution_head_sha=GIT_SHA,
        expected_materialization_identity_sha256=MATERIALIZATION_SHA,
        expected_result_jsonl_sha256=JSONL_SHA,
        expected_record_count=inventory["record_count"],
        expected_total_payload_bytes=inventory["total_payload_bytes"],
        expected_source_object_count=len(
            {row["source_id"] for row in inventory["records"]}
        ),
        expected_record_inventory_digest_sha256=inventory[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=inventory[
            "payload_inventory_digest_sha256"
        ],
        source_git_sha=GIT_SHA,
    )
    return vector, inventory, evidence


def _dedup() -> dict:
    return {
        "worker_id": DEDUP_WORKER,
        "head_sha": DEDUP_HEAD_SHA,
        "evidence_identity_sha256": DEDUP_EVIDENCE_SHA,
        "terminal_verdict": "PASS",
    }


def _adapt(vector: dict) -> dict:
    return adapt_postmaterialization_family_vector_to_next100_106(
        vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        dedup_authority=_dedup(),
        expected_dedup_worker_id=DEDUP_WORKER,
        expected_dedup_head_sha=DEDUP_HEAD_SHA,
        expected_dedup_evidence_identity_sha256=DEDUP_EVIDENCE_SHA,
    )


def _partial_rows() -> list[dict]:
    return [
        _row("ua-a", UA_A, "uk", 1_000),
        _row("ua-b", UA_B, "uk", 1_000),
        _row("en-a", EN_A, "en", 1_000),
        _row("en-b", EN_B, "en", 1_000),
        _row("code-a", CODE_A, "code", 1_000),
        _row("code-b", CODE_B, "code", 1_000),
    ]


def _target_rows() -> list[dict]:
    return [
        _row("ua-a", UA_A, "uk", 4_500_000),
        _row("ua-b", UA_B, "uk", 4_500_000),
        _row("en-a", EN_A, "en", 3_500_000),
        _row("en-b", EN_B, "en", 3_500_000),
        _row("code-a", CODE_A, "code", 2_000_000),
        _row("code-b", CODE_B, "code", 2_000_000),
    ]


def test_postmaterialization_vector_binds_machine_inventory() -> None:
    vector, inventory, _ = _build(_partial_rows())
    assert vector["schema"] == FAMILY_VECTOR_SCHEMA
    assert vector["record_count"] == 6
    assert vector["total_payload_bytes"] == 6_000
    assert vector["source_object_count"] == 6
    assert vector["record_inventory_digest_sha256"] == inventory[
        "record_inventory_digest_sha256"
    ]
    assert vector["stratum_capacity_bytes"] == {
        "code": 2_000,
        "en": 2_000,
        "uk": 2_000,
    }
    assert verify_postmaterialization_family_vector(
        vector,
        expected_identity_sha256=vector["family_vector_identity_sha256"],
    ) == vector["family_vector_identity_sha256"]


def test_inventory_rows_are_rehashed_not_trusted() -> None:
    _, inventory, evidence = _build(_partial_rows())
    tampered = copy.deepcopy(inventory)
    tampered["records"][0]["payload_bytes"] += 1
    with pytest.raises(ProjectionError, match="inventory .* drift"):
        build_postmaterialization_family_vector(
            inventory=tampered,
            materialization_evidence=evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_coherently_resealed_wrong_inventory_still_fails_external_roots() -> None:
    _, inventory, _ = _build(_partial_rows())
    tampered = copy.deepcopy(inventory)
    tampered["records"][0]["payload_bytes"] += 1
    tampered = _inventory(tampered["records"])
    forged_evidence = _evidence(tampered)
    with pytest.raises(ProjectionError, match="independently expected value"):
        build_postmaterialization_family_vector(
            inventory=tampered,
            materialization_evidence=forged_evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_materialization_truth_drift_fails_closed() -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence["truth_boundary"]["tokenizer_fit_authorized"] = True
    with pytest.raises(ProjectionError, match="truth boundary drift"):
        build_postmaterialization_family_vector(
            inventory=inventory,
            materialization_evidence=evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_adapter_preserves_physical_authority_for_external_binding() -> None:
    vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    next100_gate.validate_vector(adapted)
    assert adapted["physical_authority"]["family_vector_identity_sha256"] == vector[
        "family_vector_identity_sha256"
    ]
    assert adapted["physical_authority"]["materialization_identity_sha256"] == (
        MATERIALIZATION_SHA
    )


def test_partial_balance_is_bound_but_cannot_authorize_selection() -> None:
    vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"

    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    assert binding["schema"] == BALANCE_BINDING_SCHEMA
    assert binding["balanced_selection_authorized"] is False
    with pytest.raises(
        ProjectionError,
        match="balanced selection blocked",
    ):
        require_balanced_selection_ready(
            binding,
            expected_binding_identity_sha256=binding[
                "binding_identity_sha256"
            ],
        )


def test_target_feasible_balance_can_cross_selection_readiness_gate() -> None:
    vector, _, _ = _build(_target_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "TARGET_20M_SOURCE_MIX_FEASIBLE"
    assert balance["maximum_feasible_total_source_bytes"] == 20_000_000

    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    require_balanced_selection_ready(
        binding,
        expected_binding_identity_sha256=binding["binding_identity_sha256"],
    )


def test_balance_result_substitution_fails_binding() -> None:
    vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    tampered = copy.deepcopy(balance)
    tampered["maximum_feasible_total_source_bytes"] += 100
    with pytest.raises(ProjectionError, match="self-hash mismatch"):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            next100_input=adapted,
            balance_result=tampered,
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )
