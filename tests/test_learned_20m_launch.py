from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.learned_20m_launch import AUTHORIZED, BLOCKED, READY, evaluate_packet

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = json.loads(
    (ROOT / "configs/launch/learned_20m_authorization_v1.json").read_text(encoding="utf-8")
)


def _scientifically_ready() -> dict[str, object]:
    packet = copy.deepcopy(TEMPLATE)
    packet["authority"]["source_git_sha"] = "a" * 40

    for index, key in enumerate(packet["identities"], 1):
        packet["identities"][key] = format(index, "064x")

    exposure = packet["post_pack_exposure"]
    exposure["unique_causal_loss_positions"] = 20_000_000
    exposure["ledger_sha256"] = "6" * 64

    for record in packet["terminal_evidence"].values():
        record["terminal"] = True
        record["evidence_ref"] = "issue:548#terminal-evidence-fixture"

    pilot = packet["terminal_evidence"]["bounded_pilot"]
    pilot["max_steps"] = 16
    for key in (
        "finite_loss",
        "loss_decreased",
        "gradient_health_passed",
        "checkpoint_resume_passed",
        "throughput_measured",
        "evaluation_isolation_passed",
    ):
        pilot[key] = True

    recipe = packet["training_recipe"]
    names = {
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "precision": "bf16",
        "gradient_policy": "global_norm_clip",
    }
    for index, section_name in enumerate(
        ("optimizer", "scheduler", "precision", "gradient_policy"), 7
    ):
        recipe[section_name]["name"] = names[section_name]
        recipe[section_name]["config_sha256"] = format(index, "064x")
    recipe["seeds"] = [1234, 5678]
    recipe["budget"] = {
        "target_optimized_tokens": 20_000_000,
        "max_steps": 10_000,
        "max_wall_minutes": 720,
    }
    recipe["stop_rules"] = [
        "STOP_ON_NAN_INF",
        "STOP_ON_CHECKPOINT_INTEGRITY_FAILURE",
        "STOP_ON_EVALUATION_FIREWALL_BREACH",
    ]

    packet["resource_envelope"] = {
        "estimated_flops": 2.5e15,
        "estimated_wall_clock_hours": 8.0,
        "device_type": "qualified-accelerator",
        "device_count": 1,
        "max_cost_usd": 100.0,
        "estimate_evidence_ref": "issue:653#resource-envelope-fixture",
    }
    return packet


def _authorize(packet: dict[str, object]) -> None:
    packet["authorizations"]["compute"] = {
        "decision": "COMPUTE_AUTHORIZED",
        "reference": "owner-decision:compute-001",
        "max_cost_usd": 100.0,
    }
    packet["authorizations"]["training"] = {
        "decision": "TRAINING_AUTHORIZED",
        "reference": "owner-decision:training-001",
    }


def test_repository_template_is_fail_closed() -> None:
    result = evaluate_packet(copy.deepcopy(TEMPLATE))
    assert result["state"] == BLOCKED
    assert result["training_may_start"] is False
    assert result["blockers"]


def test_scientific_readiness_only_never_authorizes_training() -> None:
    packet = _scientifically_ready()
    packet["status"] = "TRAINING_AUTHORIZED"
    result = evaluate_packet(packet)
    assert result["state"] == READY
    assert result["training_may_start"] is False
    assert result["blockers"] == []
    assert result["authorization_missing"] == ["compute", "training"]


def test_compute_authorization_alone_is_not_training_authorization() -> None:
    packet = _scientifically_ready()
    packet["authorizations"]["compute"] = {
        "decision": "COMPUTE_AUTHORIZED",
        "reference": "owner-decision:compute-001",
        "max_cost_usd": 100.0,
    }
    result = evaluate_packet(packet)
    assert result["state"] == READY
    assert result["training_may_start"] is False
    assert result["authorization_missing"] == ["training"]


def test_separate_compute_and_training_authorizations_unlock_final_state() -> None:
    packet = _scientifically_ready()
    _authorize(packet)
    result = evaluate_packet(packet)
    assert result["state"] == AUTHORIZED
    assert result["training_may_start"] is True
    assert result["blockers"] == []
    assert result["authorization_missing"] == []


def test_replay_blocks_even_with_both_authorizations() -> None:
    packet = _scientifically_ready()
    packet["post_pack_exposure"]["replayed_loss_positions"] = 1
    _authorize(packet)
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert result["training_may_start"] is False
    assert any(item["code"] == "NON_UNIQUE_EXPOSURE" for item in result["blockers"])


def test_terminal_boolean_without_evidence_reference_is_not_enough() -> None:
    packet = _scientifically_ready()
    packet["terminal_evidence"]["checkpoint_integrity_d05"]["evidence_ref"] = None
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert any(
        item["code"] == "GATE_EVIDENCE_REF_MISSING" for item in result["blockers"]
    )


def test_model341_authority_drift_blocks_launch() -> None:
    packet = _scientifically_ready()
    packet["authority"]["model341"]["parameter_count"] += 1
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert any(
        item["code"] == "MODEL341_AUTHORITY_MISMATCH" for item in result["blockers"]
    )


def test_compute_authorization_must_cover_declared_cost_envelope() -> None:
    packet = _scientifically_ready()
    _authorize(packet)
    packet["authorizations"]["compute"]["max_cost_usd"] = 99.0
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert any(item["code"] == "COMPUTE_BUDGET_TOO_SMALL" for item in result["blockers"])


def test_both_authorizations_cannot_bypass_failed_bounded_pilot() -> None:
    packet = _scientifically_ready()
    _authorize(packet)
    packet["terminal_evidence"]["bounded_pilot"]["finite_loss"] = False
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert result["training_may_start"] is False
    assert any(
        item["code"] == "BOUNDED_PILOT_NOT_QUALIFIED" for item in result["blockers"]
    )


def test_run_budget_cannot_exceed_unique_post_pack_exposure() -> None:
    packet = _scientifically_ready()
    _authorize(packet)
    packet["training_recipe"]["budget"]["target_optimized_tokens"] = 20_000_001
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert result["training_may_start"] is False
    assert any(
        item["code"] == "RUN_BUDGET_EXCEEDS_UNIQUE_EXPOSURE"
        for item in result["blockers"]
    )


def test_gradient_policy_is_mandatory_recipe_authority() -> None:
    packet = _scientifically_ready()
    packet["training_recipe"]["gradient_policy"]["name"] = None
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert any(item["code"] == "RECIPE_VALUE_MISSING" for item in result["blockers"])


def test_nonfinite_cost_envelope_is_rejected() -> None:
    packet = _scientifically_ready()
    packet["resource_envelope"]["max_cost_usd"] = float("inf")
    result = evaluate_packet(packet)
    assert result["state"] == BLOCKED
    assert any(item["code"] == "COST_ENVELOPE_INVALID" for item in result["blockers"])
