from __future__ import annotations

import copy
import hashlib
import json

import pytest

import tools.next100_106_balance_gate as next100_gate
from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postmaterialization_balance_projection_v1 import (
    BALANCE_BINDING_SCHEMA,
    CURRENT_CLEAN_BALANCE_BINDING_SCHEMA,
    CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA,
    FAMILY_VECTOR_SCHEMA,
    adapt_postmaterialization_family_vector_to_next100_106,
    build_balance_result_binding,
    build_current_clean_family_vector,
    build_postmaterialization_family_vector,
    load_strict_json_object,
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
        "whole_corpus_external_llm_cleanliness_claimed": False,
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
    vector, inventory, evidence = _build(_partial_rows())
    assert vector["schema"] == FAMILY_VECTOR_SCHEMA
    assert vector["record_count"] == 6
    assert vector["total_payload_bytes"] == 6_000
    assert vector["source_object_count"] == 6
    assert vector["record_inventory_digest_sha256"] == inventory[
        "record_inventory_digest_sha256"
    ]
    assert evidence["truth_boundary"][
        "whole_corpus_external_llm_cleanliness_claimed"
    ] is False
    assert (
        "external_llm_or_api_used_for_data_or_intelligence"
        not in evidence["truth_boundary"]
    )
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


def test_legacy_global_external_llm_negative_is_rejected() -> None:
    _, inventory, evidence = _build(_partial_rows())
    del evidence["truth_boundary"]["whole_corpus_external_llm_cleanliness_claimed"]
    evidence["truth_boundary"][
        "external_llm_or_api_used_for_data_or_intelligence"
    ] = False
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


def test_positive_whole_corpus_external_llm_cleanliness_claim_is_rejected() -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence["truth_boundary"]["whole_corpus_external_llm_cleanliness_claimed"] = True
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


def _rebuild_documents(inventory: dict, evidence: dict) -> dict:
    expected = _inventory(inventory["records"])
    return build_postmaterialization_family_vector(
        inventory=inventory,
        materialization_evidence=evidence,
        expected_execution_head_sha=GIT_SHA,
        expected_materialization_identity_sha256=MATERIALIZATION_SHA,
        expected_result_jsonl_sha256=JSONL_SHA,
        expected_record_count=expected["record_count"],
        expected_total_payload_bytes=expected["total_payload_bytes"],
        expected_source_object_count=len(
            {row["source_id"] for row in expected["records"]}
        ),
        expected_record_inventory_digest_sha256=expected[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=expected[
            "payload_inventory_digest_sha256"
        ],
        source_git_sha=GIT_SHA,
    )


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("whole_corpus_external_llm_cleanliness_claimed", 0),
        ("authorized_optimized_target_exposure", False),
        ("authorized_optimized_target_exposure", 0.0),
    ],
)
def test_zero_credit_truth_rejects_scalar_type_aliases(
    field: str,
    alias: object,
) -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence = copy.deepcopy(evidence)
    evidence["truth_boundary"][field] = alias
    with pytest.raises(ProjectionError, match="truth boundary drift"):
        _rebuild_documents(inventory, evidence)


@pytest.mark.parametrize(
    "field",
    ["record_count", "total_payload_bytes", "source_object_count"],
)
def test_materialization_result_rejects_integer_to_float_aliases(field: str) -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence = copy.deepcopy(evidence)
    evidence["result"][field] = float(evidence["result"][field])
    with pytest.raises(
        ProjectionError,
        match=rf"materialization result\.{field} does not match independently expected value",
    ):
        _rebuild_documents(inventory, evidence)


@pytest.mark.parametrize("field", ["record_count", "total_payload_bytes"])
def test_inventory_totals_reject_integer_to_float_aliases(field: str) -> None:
    _, inventory, evidence = _build(_partial_rows())
    inventory = copy.deepcopy(inventory)
    inventory[field] = float(inventory[field])
    with pytest.raises(
        ProjectionError,
        match=rf"post-materialization inventory {field} drift",
    ):
        _rebuild_documents(inventory, evidence)


def _current_clean_bytes(
    rows: list[dict],
) -> tuple[dict, dict[str, bytes], dict[str, object]]:
    survivors: list[dict[str, str]] = []
    inventory_rows: list[dict] = []
    for index, row in enumerate(rows):
        payload = chr(ord("a") + index) * row["payload_bytes"]
        survivor = {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "family": row["family"],
            "modality": row["modality"],
            "normalized_payload": payload,
        }
        survivors.append(survivor)
        payload_raw = payload.encode()
        inventory_rows.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "family": row["family"],
                "modality": row["modality"],
                "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
                "payload_bytes": len(payload_raw),
            }
        )
    inventory_rows.sort(key=lambda row: row["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in inventory_rows
    ]
    inventory = {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(inventory_rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in inventory_rows),
        "record_inventory_digest_sha256": hashlib.sha256(
            _canonical(inventory_rows)
        ).hexdigest(),
        "payload_inventory_digest_sha256": hashlib.sha256(
            _canonical(payload_projection)
        ).hexdigest(),
        "records": inventory_rows,
    }
    survivor_raw = b"".join(_canonical(record) + b"\n" for record in survivors)
    survivor_sha = hashlib.sha256(survivor_raw).hexdigest()
    source_objects = len({row["source_id"] for row in inventory["records"]})
    count = inventory["record_count"]
    receipt: dict[str, object] = {
        "schema_version": "12-6.current-clean-decontam-quality-privacy-survivor.v2",
        "status": "CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION",
        "clean_training_records_sha256": "1" * 64,
        "clean_training_handoff_sha256": "2" * 64,
        "data232_report_sha256": "3" * 64,
        "decontamination_execution_identity_sha256": "4" * 64,
        "eval647_execution_receipt_identity_sha256": "5" * 64,
        "quality_execution_identity_sha256": "6" * 64,
        "privacy_execution_identity_sha256": "7" * 64,
        "post_decontamination_input_rows_sha256": "8" * 64,
        "post_quality_input_rows_sha256": "9" * 64,
        "survivor_jsonl_sha256": survivor_sha,
        "survivor_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "survivor_payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "input_training_records": count,
        "post_decontamination_records": count,
        "post_quality_records": count,
        "survivor_records": count,
        "survivor_payload_bytes": inventory["total_payload_bytes"],
        "survivor_source_objects": source_objects,
        "rejection_counts": {},
        "privacy_detector_counts": {},
        "dependency_git_blobs": {"src/twelve_six/data/example.py": "a" * 40},
        "durable_evidence_hash_only": True,
        "terminal_post_g05_g06_authority": False,
        "independent_qualification_required": True,
        "current_corpus_launch_authority_promoted": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    receipt["receipt_identity_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    receipt_raw = _canonical(receipt) + b"\n"
    inventory_raw = _canonical(inventory) + b"\n"
    roots = {
        "composition_receipt.json": hashlib.sha256(receipt_raw).hexdigest(),
        "data232_report.json": "b" * 64,
        "decontamination_execution.json": "c" * 64,
        "eval647_execution_receipt.json": "d" * 64,
        "privacy_execution.json": "e" * 64,
        "quality_execution.json": "f" * 64,
        "survivor_inventory.json": hashlib.sha256(inventory_raw).hexdigest(),
        "survivor_records.jsonl": survivor_sha,
    }
    repeat: dict[str, object] = {
        "schema_version": "12-6.d03-current-clean-composition-physical-repeat.v1",
        "status": "PHYSICAL_EXECUTION_COMPLETE_PENDING_INDEPENDENT_QUALIFICATION",
        "execution_head_sha": GIT_SHA,
        "execution_profile": "LOCAL_FREE",
        "output_files_sha256": roots,
        "survivor_jsonl_sha256": survivor_sha,
        "survivor_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "survivor_payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "survivor_records": count,
        "survivor_payload_bytes": inventory["total_payload_bytes"],
        "two_fresh_executions_byte_identical": True,
        "terminal_post_g05_g06_authority": False,
        "independent_qualification_required": True,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    repeat["proof_identity_sha256"] = hashlib.sha256(_canonical(repeat)).hexdigest()
    repeat_raw = _canonical(repeat) + b"\n"
    raw = {
        "composition_receipt": receipt_raw,
        "survivor_inventory": inventory_raw,
        "repeat_proof": repeat_raw,
        "survivor_records": survivor_raw,
    }
    expected = {
        "expected_composition_receipt_json_sha256": hashlib.sha256(
            receipt_raw
        ).hexdigest(),
        "expected_survivor_inventory_json_sha256": hashlib.sha256(
            inventory_raw
        ).hexdigest(),
        "expected_repeat_proof_json_sha256": hashlib.sha256(repeat_raw).hexdigest(),
        "expected_survivor_records_jsonl_sha256": survivor_sha,
        "expected_receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "expected_repeat_proof_identity_sha256": repeat["proof_identity_sha256"],
        "expected_execution_head_sha": GIT_SHA,
        "expected_record_count": count,
        "expected_total_payload_bytes": inventory["total_payload_bytes"],
        "expected_source_object_count": source_objects,
        "expected_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "expected_payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "source_git_sha": GIT_SHA,
    }
    return inventory, raw, expected


def _build_current_clean(rows: list[dict]) -> tuple[dict, dict[str, bytes], dict[str, object]]:
    _, raw, expected = _current_clean_bytes(rows)
    vector = build_current_clean_family_vector(
        composition_receipt_raw=raw["composition_receipt"],
        survivor_inventory_raw=raw["survivor_inventory"],
        repeat_proof_raw=raw["repeat_proof"],
        survivor_records_raw=raw["survivor_records"],
        **expected,
    )
    return vector, raw, expected


def _reseal(document: dict, identity_field: str) -> bytes:
    document = copy.deepcopy(document)
    document.pop(identity_field, None)
    document[identity_field] = hashlib.sha256(_canonical(document)).hexdigest()
    return _canonical(document) + b"\n"


def test_current_clean_vector_binds_receipt_repeat_and_raw_files() -> None:
    vector, raw, expected = _build_current_clean(_partial_rows())
    assert vector["schema"] == CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA
    assert vector["current_clean_receipt_identity_sha256"] == (
        expected["expected_receipt_identity_sha256"]
    )
    assert vector["current_clean_repeat_proof_identity_sha256"] == (
        expected["expected_repeat_proof_identity_sha256"]
    )
    assert vector["composition_receipt_json_sha256"] == hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    assert vector["repeat_proof_json_sha256"] == hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    assert verify_postmaterialization_family_vector(
        vector,
        expected_identity_sha256=vector["family_vector_identity_sha256"],
    ) == vector["family_vector_identity_sha256"]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_strict_current_clean_json_rejects_nonfinite_numbers(literal: str) -> None:
    with pytest.raises(ProjectionError):
        load_strict_json_object(
            ('{"outer":{"value":' + literal + "}}").encode(),
            label="adversarial",
        )


def test_strict_current_clean_json_rejects_nested_duplicate_keys() -> None:
    with pytest.raises(ProjectionError):
        load_strict_json_object(
            b'{"outer":{"identity":"a","identity":"b"}}',
            label="adversarial",
        )


def test_current_clean_rejects_raw_receipt_byte_substitution() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    raw["composition_receipt"] += b" "
    with pytest.raises(ProjectionError):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_coherent_receipt_reseal_under_fixed_identity() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_payload_bytes"] += 1
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="not independently expected"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_repeat_raw_root_substitution_even_if_resealed() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    repeat = json.loads(raw["repeat_proof"])
    repeat["output_files_sha256"]["survivor_inventory.json"] = "0" * 64
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    resealed = json.loads(raw["repeat_proof"])
    expected["expected_repeat_proof_identity_sha256"] = resealed[
        "proof_identity_sha256"
    ]
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="raw root"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_source_object_count_drift() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_source_objects"] -= 1
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed = json.loads(raw["composition_receipt"])
    expected["expected_receipt_identity_sha256"] = resealed[
        "receipt_identity_sha256"
    ]
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    expected["expected_source_object_count"] = receipt["survivor_source_objects"]
    with pytest.raises(ProjectionError, match="source_object_count"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_boolean_alias_for_zero_exposure() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["authorized_optimized_target_exposure"] = False
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed = json.loads(raw["composition_receipt"])
    expected["expected_receipt_identity_sha256"] = resealed[
        "receipt_identity_sha256"
    ]
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="zero-credit"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_legacy_schema_impersonation() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["schema_version"] = "12-6.d03-post-g05-g06-materialization.v1"
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed = json.loads(raw["composition_receipt"])
    expected["expected_receipt_identity_sha256"] = resealed[
        "receipt_identity_sha256"
    ]
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="current-clean receipt schema"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_balance_binding_preserves_explicit_physical_authority() -> None:
    vector, _, _ = _build_current_clean(_partial_rows())
    adapted = _adapt(vector)
    physical = adapted["physical_authority"]
    assert "materialization_identity_sha256" not in physical
    assert physical["current_clean_receipt_identity_sha256"] == vector[
        "current_clean_receipt_identity_sha256"
    ]
    assert physical["current_clean_repeat_proof_identity_sha256"] == vector[
        "current_clean_repeat_proof_identity_sha256"
    ]
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
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
    assert binding["schema"] == CURRENT_CLEAN_BALANCE_BINDING_SCHEMA
    assert binding["current_clean_physical_authority"] == physical
    assert binding["authorized_optimized_target_exposure"] == 0


def test_current_clean_rebuilds_inventory_from_survivor_jsonl() -> None:
    inventory, raw, expected = _current_clean_bytes(_partial_rows())
    parsed = [
        json.loads(line)
        for line in raw["survivor_records"].decode("utf-8").splitlines()
    ]
    assert len(parsed) == inventory["record_count"]
    assert sum(len(row["normalized_payload"].encode()) for row in parsed) == (
        inventory["total_payload_bytes"]
    )
    vector = build_current_clean_family_vector(
        composition_receipt_raw=raw["composition_receipt"],
        survivor_inventory_raw=raw["survivor_inventory"],
        repeat_proof_raw=raw["repeat_proof"],
        survivor_records_raw=raw["survivor_records"],
        **expected,
    )
    assert vector["record_inventory_digest_sha256"] == inventory[
        "record_inventory_digest_sha256"
    ]


def test_current_clean_rejects_coherently_resealed_record_inventory_mismatch() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    records = [
        json.loads(line)
        for line in raw["survivor_records"].decode("utf-8").splitlines()
    ]
    records[0]["normalized_payload"] = "z" + records[0]["normalized_payload"][1:]
    raw["survivor_records"] = b"".join(
        _canonical(record) + b"\n" for record in records
    )
    survivor_sha = hashlib.sha256(raw["survivor_records"]).hexdigest()

    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_jsonl_sha256"] = survivor_sha
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed_receipt = json.loads(raw["composition_receipt"])

    repeat = json.loads(raw["repeat_proof"])
    repeat["survivor_jsonl_sha256"] = survivor_sha
    repeat["output_files_sha256"]["survivor_records.jsonl"] = survivor_sha
    repeat["output_files_sha256"]["composition_receipt.json"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    resealed_repeat = json.loads(raw["repeat_proof"])

    expected["expected_survivor_records_jsonl_sha256"] = survivor_sha
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    expected["expected_receipt_identity_sha256"] = resealed_receipt[
        "receipt_identity_sha256"
    ]
    expected["expected_repeat_proof_identity_sha256"] = resealed_repeat[
        "proof_identity_sha256"
    ]

    with pytest.raises(ProjectionError, match="differs from rebuilt survivor JSONL"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_noncanonical_survivor_jsonl() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    first, *rest = raw["survivor_records"].splitlines(keepends=True)
    record = json.loads(first)
    noncanonical = json.dumps(record, ensure_ascii=False).encode() + b"\n"
    raw["survivor_records"] = noncanonical + b"".join(rest)
    survivor_sha = hashlib.sha256(raw["survivor_records"]).hexdigest()

    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_jsonl_sha256"] = survivor_sha
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed_receipt = json.loads(raw["composition_receipt"])

    repeat = json.loads(raw["repeat_proof"])
    repeat["survivor_jsonl_sha256"] = survivor_sha
    repeat["output_files_sha256"]["survivor_records.jsonl"] = survivor_sha
    repeat["output_files_sha256"]["composition_receipt.json"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    resealed_repeat = json.loads(raw["repeat_proof"])

    expected["expected_survivor_records_jsonl_sha256"] = survivor_sha
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    expected["expected_receipt_identity_sha256"] = resealed_receipt[
        "receipt_identity_sha256"
    ]
    expected["expected_repeat_proof_identity_sha256"] = resealed_repeat[
        "proof_identity_sha256"
    ]

    with pytest.raises(ProjectionError, match="not canonical JSON"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )
