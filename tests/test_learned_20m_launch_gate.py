from copy import deepcopy

import pytest

from twelve_six.learned_20m_launch_gate import (
    BLOCKED,
    EXPECTED_AUTHORITY,
    READY_FOR_AUTHORIZATION_REQUEST,
    TRAINING_AUTHORIZED,
    assess_learned_20m_launch,
)


def _ready_packet() -> dict:
    return {
        "schema_version": 1,
        "packet_id": "LEARNED-20M-LAUNCH-V1",
        "authority": dict(EXPECTED_AUTHORITY),
        "artifacts": {
            "code_sha": "1" * 40,
            "tokenizer_sha256": "2" * 64,
            "corpus_sha256": "3" * 64,
            "split_sha256": "4" * 64,
            "packing_sha256": "5" * 64,
            "unique_loss_ledger_sha256": "6" * 64,
            "unique_post_pack_causal_loss_positions": 500_000_000,
        },
        "gates": {
            "no_replay_proven": True,
            "no_replay_accounting_ref": "ledger:unique-loss-v1",
            "evaluation_decontamination_terminal": True,
            "evaluation_decontamination_ref": "eval:terminal:sha256",
            "checkpoint_integrity_terminal": True,
            "checkpoint_integrity_ref": "github:pr/504@terminal-head",
            "learned_3m_independent_terminal": True,
            "learned_3m_independent_ref": "verify:3m:terminal",
            "learned_10m_independent_terminal": True,
            "learned_10m_independent_ref": "verify:10m:terminal",
        },
        "recipe": {
            "optimizer": "AdamW",
            "scheduler": "preregistered-schedule-v1",
            "precision": "bf16",
            "seeds": [17, 29],
            "target_causal_loss_positions": 412_268_800,
            "stop_rule": "stop-on-budget-or-preregistered-safety-threshold",
            "checkpoint_policy_ref": "checkpoint:policy:v1",
        },
        "resources": {
            "hardware_profile": "qualified-profile-id",
            "estimated_flops": 1.0e17,
            "estimated_wall_clock_hours": 12.0,
            "maximum_cost_usd": 250.0,
            "output_destination": "artifact-store:learned20m/run-id",
            "cancellation_rule": "cancel-on-numerical-or-integrity-gate-failure",
        },
        "authorizations": {
            "compute_authorization_ref": None,
            "training_authorization_ref": None,
        },
    }


def test_scientific_readiness_is_not_training_authorization() -> None:
    result = assess_learned_20m_launch(_ready_packet())
    assert result["state"] == READY_FOR_AUTHORIZATION_REQUEST
    assert result["ready_for_authorization_request"] is True
    assert result["training_authorized"] is False
    assert set(result["authorization_blockers"]) == {
        "compute_authorization_ref_missing",
        "training_authorization_ref_missing",
    }


def test_two_separate_explicit_authorizations_are_required() -> None:
    packet = _ready_packet()
    packet["authorizations"] = {
        "compute_authorization_ref": "owner-approval:compute:learned20m-v1",
        "training_authorization_ref": "owner-approval:training:learned20m-v1",
    }
    result = assess_learned_20m_launch(packet)
    assert result["state"] == TRAINING_AUTHORIZED
    assert result["training_authorized"] is True
    assert result["blockers"] == []
    assert result["authorization_blockers"] == []


def test_parameter_count_does_not_bypass_data_gates() -> None:
    packet = _ready_packet()
    packet["artifacts"]["corpus_sha256"] = None
    packet["artifacts"]["unique_post_pack_causal_loss_positions"] = 0
    result = assess_learned_20m_launch(packet)
    assert result["state"] == BLOCKED
    assert result["training_authorized"] is False
    assert "artifacts_corpus_sha256_must_be_lowercase_hex_64" in result["blockers"]
    expected = "unique_post_pack_causal_loss_positions_must_be_positive_integer"
    assert expected in result["blockers"]


def test_hidden_replay_budget_is_blocked() -> None:
    packet = _ready_packet()
    packet["artifacts"]["unique_post_pack_causal_loss_positions"] = 100
    packet["recipe"]["target_causal_loss_positions"] = 101
    result = assess_learned_20m_launch(packet)
    assert result["state"] == BLOCKED
    assert "target_causal_loss_positions_exceed_unique_authorized_positions" in result[
        "blockers"
    ]


@pytest.mark.parametrize(
    ("terminal_key", "expected_blocker"),
    [
        ("evaluation_decontamination_terminal", "evaluation_decontamination_not_terminal"),
        ("checkpoint_integrity_terminal", "checkpoint_integrity_not_terminal"),
        ("learned_3m_independent_terminal", "learned_3m_independent_not_terminal"),
        ("learned_10m_independent_terminal", "learned_10m_independent_not_terminal"),
    ],
)
def test_every_terminal_scientific_gate_is_independent(
    terminal_key: str, expected_blocker: str
) -> None:
    packet = _ready_packet()
    packet["gates"][terminal_key] = False
    result = assess_learned_20m_launch(packet)
    assert result["state"] == BLOCKED
    assert expected_blocker in result["blockers"]


def test_same_reference_cannot_authorize_compute_and_training() -> None:
    packet = _ready_packet()
    packet["authorizations"] = {
        "compute_authorization_ref": "approval:same",
        "training_authorization_ref": "approval:same",
    }
    result = assess_learned_20m_launch(packet)
    assert result["state"] == READY_FOR_AUTHORIZATION_REQUEST
    assert result["training_authorized"] is False
    expected = "compute_and_training_authorization_refs_must_be_distinct"
    assert expected in result["authorization_blockers"]


def test_manual_state_field_is_rejected_instead_of_trusted() -> None:
    packet = _ready_packet()
    packet["state"] = TRAINING_AUTHORIZED
    result = assess_learned_20m_launch(packet)
    assert result["state"] == BLOCKED
    assert "packet_unknown_state" in result["blockers"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_flop_envelope_blocks_launch(value: float) -> None:
    packet = _ready_packet()
    packet["resources"]["estimated_flops"] = value
    result = assess_learned_20m_launch(packet)
    assert result["state"] == BLOCKED
    assert "estimated_flops_must_be_positive_finite" in result["blockers"]


def test_authority_drift_fails_closed() -> None:
    packet = _ready_packet()
    packet["authority"]["model341_sha"] = "0" * 40
    result = assess_learned_20m_launch(packet)
    assert result["state"] == BLOCKED
    assert "authority_model341_sha_mismatch" in result["blockers"]


def test_assessment_does_not_mutate_input() -> None:
    packet = _ready_packet()
    original = deepcopy(packet)
    assess_learned_20m_launch(packet)
    assert packet == original
