from __future__ import annotations

import json
from pathlib import Path

from twelve_six.learned20m_readiness import assess_learned20m_readiness

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _authority(kind: str, *, workflow: bool = True, independent: bool = False) -> dict:
    ref = {
        "repository": "Oleksii-debug/12-6-ai.",
        "evidence_kind": kind,
        "git_sha": SHA40,
        "observed_head_sha": SHA40,
        "evidence_sha256": SHA64,
        "terminal": True,
        "self_asserted": False,
    }
    if workflow:
        ref.update(
            workflow_run_id=123,
            workflow_conclusion="success",
            workflow_run_head_sha=SHA40,
        )
    if independent:
        ref.update(
            independent=True,
            producer_identity="producer-lane",
            verifier_identity="independent-auditor",
        )
    return ref


def _local_ready() -> dict:
    data = _load()
    e = data["evidence"]
    e["code"].update(git_sha=SHA40, authority=_authority("qualified_integration_head"))
    e["corpus"].update(
        manifest_sha256=SHA64,
        split_sha256=SHA64,
        packing_sha256=SHA64,
        two_clean_builds_identical=True,
        authority=_authority("research_corpus_v1"),
    )
    e["tokenizer"].update(
        identity_sha256=SHA64,
        decision="BYTE_BASELINE_RETAINED",
        authority=_authority("tokenizer_decision"),
    )
    e["loss_ledger"].update(
        identity_sha256=SHA64,
        unique_causal_loss_positions=412_268_800,
        authority=_authority("unique_post_pack_loss_ledger"),
        data_budget_authority=_authority("terminal_data_budget_qualification"),
        data_budget_status="QUALIFIED",
    )
    e["checkpoint_integrity"].update(
        authority=_authority("d05_checkpoint_integrity"), status="PASS"
    )
    e["evaluation"].update(
        firewall_authority=_authority("evaluation_firewall"),
        selection_validation_authority=_authority("selection_validation"),
        status="PASS",
    )
    e["training_recipe"].update(
        authority=_authority("learned_20m_training_recipe"),
        status="QUALIFIED",
        seed_count=2,
        config_sha256=SHA64,
        stopping_policy_sha256=SHA64,
        requested_unique_loss_positions=103_067_200,
    )
    return data


def _request_ready() -> dict:
    data = _local_ready()
    e = data["evidence"]
    e["bounded_pilot"].update(
        authority=_authority("bounded_learned_20m_pilot"),
        status="PASS",
        numerics_finite=True,
        resume_equivalent=True,
        loss_trajectory_acceptable=True,
    )
    e["cost_envelope"].update(
        authority=_authority("material_compute_cost_envelope", workflow=False),
        status="ESTIMATED",
        maximum_cost_usd=50.0,
    )
    e["independent_audit"].update(
        authority=_authority("independent_launch_audit", independent=True),
        status="PASS",
    )
    return data


def _add_authorizations(data: dict, *, budget: float = 50.0) -> None:
    e = data["evidence"]
    compute_ref = _authority("material_compute_authorization", workflow=False)
    e["compute_authorization"].update(
        authority=compute_ref,
        status="COMPUTE_AUTHORIZED",
        scope="LEARNED_20M_MATERIAL_COMPUTE",
        authorized_by_owner=True,
        maximum_cost_usd=budget,
    )
    e["training_authorization"].update(
        authority=_authority("learned_20m_training_authorization", workflow=False),
        status="TRAINING_AUTHORIZED",
        scope="LEARNED_20M_MATERIAL_TRAINING",
        authorized_by_owner=True,
        training_config_sha256=e["training_recipe"]["config_sha256"],
        compute_authorization_evidence_sha256=compute_ref["evidence_sha256"],
    )


def test_checked_in_packet_remains_blocked() -> None:
    result = assess_learned20m_readiness(_load())
    assert not result.ready_for_local_free_pilot
    assert not result.ready_for_compute_authorization_request
    assert not result.material_training_authorized
    assert "data_budget_not_qualified" in result.local_free_pilot_blockers
    assert "training_not_explicitly_authorized" in result.material_training_blockers


def test_readiness_phases_remain_separate() -> None:
    local = assess_learned20m_readiness(_local_ready())
    assert local.ready_for_local_free_pilot
    assert not local.ready_for_compute_authorization_request

    request = assess_learned20m_readiness(_request_ready())
    assert request.ready_for_compute_authorization_request
    assert not request.material_training_authorized


def test_requested_unique_loss_budget_is_bounded_and_well_typed() -> None:
    for bad in (0, -1, True, 1.5, "100", 412_268_801):
        data = _local_ready()
        data["evidence"]["training_recipe"]["requested_unique_loss_positions"] = bad
        result = assess_learned20m_readiness(data)
        assert not result.ready_for_local_free_pilot


def test_compute_approval_alone_does_not_authorize_training() -> None:
    data = _request_ready()
    e = data["evidence"]
    e["compute_authorization"].update(
        authority=_authority("material_compute_authorization", workflow=False),
        status="COMPUTE_AUTHORIZED",
        scope="LEARNED_20M_MATERIAL_COMPUTE",
        authorized_by_owner=True,
        maximum_cost_usd=50.0,
    )
    result = assess_learned20m_readiness(data)
    assert not result.material_training_authorized
    assert "training_not_explicitly_authorized" in result.material_training_blockers


def test_full_authorization_must_cover_budget_and_exact_recipe() -> None:
    data = _request_ready()
    _add_authorizations(data, budget=49.0)
    result = assess_learned20m_readiness(data)
    assert "authorized_cost_below_estimated_maximum" in result.material_training_blockers

    data["evidence"]["compute_authorization"]["maximum_cost_usd"] = 50.0
    result = assess_learned20m_readiness(data)
    assert result.material_training_authorized

    data["evidence"]["training_authorization"]["training_config_sha256"] = "c" * 64
    result = assess_learned20m_readiness(data)
    assert not result.material_training_authorized
    assert "training_authorization_config_mismatch" in result.material_training_blockers


def test_authority_type_head_and_self_assertion_fail_closed() -> None:
    data = _local_ready()
    data["evidence"]["checkpoint_integrity"]["authority"]["evidence_kind"] = (
        "research_corpus_v1"
    )
    result = assess_learned20m_readiness(data)
    assert "checkpoint_integrity_authority_missing" in result.local_free_pilot_blockers

    data = _local_ready()
    data["evidence"]["tokenizer"]["authority"]["observed_head_sha"] = "c" * 40
    result = assess_learned20m_readiness(data)
    assert "terminal_tokenizer_authority_missing" in result.local_free_pilot_blockers

    data = _local_ready()
    data["evidence"]["corpus"]["authority"]["self_asserted"] = True
    result = assess_learned20m_readiness(data)
    assert "terminal_corpus_authority_missing" in result.local_free_pilot_blockers


def test_code_and_audit_authorities_must_match_their_subjects() -> None:
    data = _local_ready()
    code_ref = data["evidence"]["code"]["authority"]
    code_ref["git_sha"] = "c" * 40
    code_ref["observed_head_sha"] = "c" * 40
    code_ref["workflow_run_head_sha"] = "c" * 40
    result = assess_learned20m_readiness(data)
    assert "qualified_integration_head_sha_mismatch" in result.local_free_pilot_blockers

    data = _request_ready()
    audit = data["evidence"]["independent_audit"]["authority"]
    audit["verifier_identity"] = audit["producer_identity"]
    result = assess_learned20m_readiness(data)
    assert "independent_audit_authority_missing" in result.compute_request_blockers


def test_model_r01_and_truth_boundary_drift_block_readiness() -> None:
    data = _request_ready()
    data["model_authority"]["parameter_count"] += 1
    data["r01_campaign_commit_sha"] = "c" * 40
    data["truth_boundary"]["paid_compute_executed_by_this_package"] = True
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "model_authority_parameter_count_mismatch" in result.local_free_pilot_blockers
    assert "r01_campaign_commit_authority_drift" in result.local_free_pilot_blockers
