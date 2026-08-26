from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/twelve_six/learned20_launch_gate.py"
CONTRACT_PATH = ROOT / "configs/training/learned_20m_launch_gate_v1.json"

spec = importlib.util.spec_from_file_location("learned20_launch_gate", MODULE_PATH)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _pilot_ready_evidence() -> dict:
    return {
        "corpus": {
            "terminal": True,
            "identity": "corpus-v1",
            "corpus_identity": "corpus-v1",
            "split_identity": "split-v1",
            "packing_identity": "packing-v1",
            "two_clean_builds_identical": True,
        },
        "unique_loss_ledger": {
            "terminal": True,
            "identity": "loss-ledger-v1",
            "authorized_unique_loss_positions": 120_000_000,
            "replay_used": False,
        },
        "tokenizer": {
            "terminal": True,
            "identity": "tokenizer-v1",
            "mode": "trained_tokenizer",
            "roundtrip_passed": True,
        },
        "checkpoint": {
            "terminal": True,
            "identity": "checkpoint-v1",
            "corruption_matrix_passed": True,
            "fresh_resume_equivalence": True,
        },
        "evaluation_firewall": {
            "terminal": True,
            "identity": "eval-firewall-v1",
            "selection_validation_identity": "selection-v1",
            "final_test_identity": "final-v1",
            "decontamination_identity": "decontam-v1",
            "training_overlap_count": 0,
            "tokenizer_fit_overlap_count": 0,
        },
        "training_recipe": {
            "terminal": True,
            "identity": "recipe-v1",
            "optimizer_family": "AdamW",
            "learning_rate": 0.0003,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "scheduler": "cosine",
            "warmup_steps": 100,
            "precision": "bf16",
            "gradient_clip_norm": 1.0,
            "seed": 17,
            "requested_unique_loss_positions": 100_000_000,
            "max_optimizer_updates": 10_000,
            "stopping_rule": "stop at preregistered budget or numerical failure",
        },
    }


def _successful_pilot() -> dict:
    return {
        "terminal": True,
        "identity": "bounded-pilot-v1",
        "finite_loss": True,
        "loss_decreased": True,
        "gradient_health_passed": True,
        "checkpoint_resume_passed": True,
        "evaluation_isolation_passed": True,
    }


def test_frozen_contract_is_valid_and_empty_evidence_blocks() -> None:
    contract = _contract()
    assert gate.validate_contract(contract) == []
    result = gate.assess_launch(contract, {}, material_cost=False)
    assert result["pilot_ready"] is False
    assert result["long_training_ready"] is False


def test_terminal_prerequisites_make_bounded_pilot_ready_only() -> None:
    result = gate.assess_launch(
        _contract(),
        _pilot_ready_evidence(),
        material_cost=False,
    )
    assert result["pilot_ready"] is True
    assert result["long_training_ready"] is False
    assert "bounded_pilot_not_terminal" in result["long_training_blockers"]


def test_successful_bounded_pilot_makes_local_free_long_training_ready() -> None:
    evidence = _pilot_ready_evidence()
    evidence["bounded_pilot"] = _successful_pilot()
    result = gate.assess_launch(_contract(), evidence, material_cost=False)
    assert result["pilot_ready"] is True
    assert result["long_training_ready"] is True


def test_material_cost_requires_explicit_terminal_compute_authorization() -> None:
    evidence = _pilot_ready_evidence()
    evidence["bounded_pilot"] = _successful_pilot()

    blocked = gate.assess_launch(_contract(), evidence, material_cost=True)
    assert blocked["long_training_ready"] is False
    assert "compute_authorization_not_terminal" in blocked["long_training_blockers"]

    evidence["compute_authorization"] = {
        "terminal": True,
        "identity": "compute-v1",
        "compute_authorized": True,
        "max_budget_usd": 100.0,
    }
    ready = gate.assess_launch(_contract(), evidence, material_cost=True)
    assert ready["long_training_ready"] is True


def test_recipe_cannot_request_more_unique_loss_than_ledger() -> None:
    evidence = _pilot_ready_evidence()
    evidence["training_recipe"]["requested_unique_loss_positions"] = 130_000_000
    result = gate.assess_launch(_contract(), evidence, material_cost=False)
    assert result["pilot_ready"] is False
    assert (
        "training_recipe.requests_more_unique_loss_than_authorized"
        in result["pilot_blockers"]
    )


def test_evaluation_overlap_fails_closed() -> None:
    evidence = _pilot_ready_evidence()
    evidence["evaluation_firewall"]["training_overlap_count"] = 1
    result = gate.assess_launch(_contract(), evidence, material_cost=False)
    assert result["pilot_ready"] is False
    assert "evaluation_firewall.training_overlap_nonzero" in result["pilot_blockers"]


def test_invalid_learning_rate_and_precision_fail_closed() -> None:
    evidence = _pilot_ready_evidence()
    evidence["training_recipe"]["learning_rate"] = 0.0
    evidence["training_recipe"]["precision"] = "fp8"
    result = gate.assess_launch(_contract(), evidence, material_cost=False)
    assert "training_recipe.learning_rate_invalid" in result["pilot_blockers"]
    assert "training_recipe.precision_invalid" in result["pilot_blockers"]


def test_contract_cannot_silently_authorize_long_training() -> None:
    contract = copy.deepcopy(_contract())
    contract["hard_boundaries"]["long_training_authorized"] = True
    errors = gate.validate_contract(contract)
    assert any("long_training_authorized" in error for error in errors)
