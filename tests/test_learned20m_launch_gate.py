from __future__ import annotations

import copy

import pytest

from twelve_six.learned20m_launch_gate import (
    LaunchGateError,
    STATE_AUTHORIZED,
    STATE_BLOCKED,
    STATE_READY,
    assess_launch,
    request_identity,
)

H40 = "1" * 40
H64 = "2" * 64


def terminal(kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "decision": "PASS",
        "source_sha": H40,
        "artifact_sha256": H64,
        "reference": f"evidence:{kind}",
    }


def ready_packet() -> dict[str, object]:
    return {
        "schema_version": "12-6.learned20m-launch-gate.v1",
        "authority": {
            "repository": "Oleksii-debug/12-6-ai.",
            "model341_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
            "modelspec_sha256": (
                "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
            ),
            "parameter_count": 20_613_440,
            "r01_merge_sha": "a73ab38026cb7849f478cc13ad58b93534a76e2f",
            "r01_config_path": "configs/research/r01_20m_to_100m_scaling_campaign_v1.json",
            "r01_config_git_blob_sha1": "c50154db609d41eceb2ffc97912360df567bcc04",
            "base_lineage": "RANDOM_INIT_PRETRAINING_ONLY",
        },
        "evidence": {
            "code_sha": H40,
            "init_spec_sha256": H64,
            "tokenizer_identity_sha256": "3" * 64,
            "corpus_identity_sha256": "4" * 64,
            "split_identity_sha256": "5" * 64,
            "packing_identity_sha256": "6" * 64,
            "post_pack_loss_ledger_sha256": "7" * 64,
            "unique_post_pack_causal_loss_positions": 412_268_800,
            "no_replay_proven": True,
            "evaluation_decontamination": terminal("EVALUATION_DECONTAMINATION"),
            "checkpoint_integrity": terminal("D05_CHECKPOINT_INTEGRITY"),
            "learned_ladder": terminal("INDEPENDENT_LEARNED_LADDER"),
            "integration_ci": terminal("QUALIFIED_INTEGRATION_CI"),
        },
        "training_recipe": {
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "scheduler": "cosine",
            "warmup_steps": 100,
            "precision": "bf16",
            "seeds": [17, 29],
            "gradient_clip_norm": 1.0,
            "target_unique_causal_loss_positions": 412_268_800,
            "budget_policy_sha256": "8" * 64,
            "stopping_rules_sha256": "9" * 64,
            "checkpoint_policy_sha256": "a" * 64,
        },
        "resource_envelope": {
            "compute_envelope_sha256": "b" * 64,
            "hardware_profile_id": "H100-SXM-80GB-x1-measured",
            "accelerator_count": 1,
            "estimated_training_flops": 5.1e16,
            "projected_wall_hours": 2.5,
            "projected_cost_eur": 12.5,
            "max_cost_eur": 20.0,
            "throughput_measurement": terminal("THROUGHPUT_MEASUREMENT"),
        },
        "compute_authorization": None,
        "training_authorization": None,
    }


def authorize(packet: dict[str, object], *, max_cost_eur: float = 20.0) -> None:
    identity = request_identity(packet)
    packet["compute_authorization"] = {
        "decision": "COMPUTE_AUTHORIZED",
        "request_identity_sha256": identity,
        "authorization_id": "compute-auth-001",
        "approver_reference": "issue:authorization-compute",
        "max_cost_eur": max_cost_eur,
    }
    packet["training_authorization"] = {
        "decision": "TRAINING_AUTHORIZED",
        "request_identity_sha256": identity,
        "authorization_id": "training-auth-001",
        "approver_reference": "issue:authorization-training",
    }


def test_scientific_readiness_is_not_training_authorization() -> None:
    result = assess_launch(ready_packet())
    assert result.state == STATE_READY
    assert result.training_authorized is False
    assert result.blockers == ()


def test_both_bound_authorizations_are_required_for_final_state() -> None:
    packet = ready_packet()
    authorize(packet)
    result = assess_launch(packet)
    assert result.state == STATE_AUTHORIZED
    assert result.training_authorized is True


def test_partial_authorization_fails_closed() -> None:
    packet = ready_packet()
    identity = request_identity(packet)
    packet["compute_authorization"] = {
        "decision": "COMPUTE_AUTHORIZED",
        "request_identity_sha256": identity,
        "authorization_id": "compute-auth-001",
        "approver_reference": "issue:authorization-compute",
        "max_cost_eur": 20.0,
    }
    result = assess_launch(packet)
    assert result.state == STATE_BLOCKED
    assert result.blockers == ("partial_authorization",)


def test_stale_authorization_cannot_survive_packet_change() -> None:
    packet = ready_packet()
    authorize(packet)
    packet["training_recipe"]["learning_rate"] = 0.0002  # type: ignore[index]
    result = assess_launch(packet)
    assert result.state == STATE_BLOCKED
    assert result.blockers == ("compute_authorization_invalid",)


def test_zero_unique_positions_blocks_even_if_everything_else_is_present() -> None:
    packet = ready_packet()
    packet["evidence"]["unique_post_pack_causal_loss_positions"] = 0  # type: ignore[index]
    packet["training_recipe"]["target_unique_causal_loss_positions"] = 0  # type: ignore[index]
    result = assess_launch(packet)
    assert result.state == STATE_BLOCKED
    assert "unique_post_pack_causal_loss_positions_must_be_nonzero" in result.blockers


def test_no_replay_must_be_proven() -> None:
    packet = ready_packet()
    packet["evidence"]["no_replay_proven"] = False  # type: ignore[index]
    assert "no_replay_not_proven" in assess_launch(packet).blockers


def test_resource_projection_must_fit_packet_cost_ceiling() -> None:
    packet = ready_packet()
    packet["resource_envelope"]["projected_cost_eur"] = 21.0  # type: ignore[index]
    result = assess_launch(packet)
    assert result.state == STATE_BLOCKED
    assert "projected_cost_exceeds_max_cost" in result.blockers


def test_authorization_cost_cap_cannot_be_lower_than_packet_envelope() -> None:
    packet = ready_packet()
    authorize(packet, max_cost_eur=15.0)
    result = assess_launch(packet)
    assert result.state == STATE_BLOCKED
    assert "packet_max_cost_exceeds_authorization" in result.blockers


def test_request_identity_is_deterministic_and_excludes_authorization_objects() -> None:
    packet = ready_packet()
    before = request_identity(packet)
    authorize(packet)
    assert request_identity(packet) == before
    reordered = copy.deepcopy(packet)
    authority = reordered["authority"]
    assert isinstance(authority, dict)
    reordered["authority"] = dict(reversed(list(authority.items())))
    assert request_identity(reordered) == before


def test_packet_cannot_self_declare_authorization() -> None:
    packet = ready_packet()
    packet["training_authorized"] = True
    with pytest.raises(LaunchGateError, match="derived launch state"):
        assess_launch(packet)


def test_model341_authority_drift_is_rejected() -> None:
    packet = ready_packet()
    packet["authority"]["parameter_count"] = 20_000_000  # type: ignore[index]
    with pytest.raises(LaunchGateError, match="parameter_count"):
        assess_launch(packet)


def test_nonterminal_evidence_is_not_accepted_as_pass() -> None:
    packet = ready_packet()
    packet["evidence"]["checkpoint_integrity"]["decision"] = "QUEUED"  # type: ignore[index]
    result = assess_launch(packet)
    assert result.state == STATE_BLOCKED
    assert "checkpoint_integrity_not_terminal_pass" in result.blockers
