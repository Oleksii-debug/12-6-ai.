from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/learned_20m_launch_packet_v1.json"
VALIDATOR = ROOT / "tools/validate_learned_20m_launch_packet.py"

spec = importlib.util.spec_from_file_location("launch_packet_validator", VALIDATOR)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _fill_scientific_readiness(data: dict) -> None:
    evidence = data["launch_evidence"]
    evidence["code_commit_sha"] = "1" * 40
    evidence["tokenizer"]["identity_sha256"] = "2" * 64
    evidence["tokenizer"]["fit_corpus_identity_sha256"] = "3" * 64
    evidence["corpus"].update(
        {
            "corpus_identity_sha256": "4" * 64,
            "split_identity_sha256": "5" * 64,
            "packing_identity_sha256": "6" * 64,
            "unique_causal_loss_positions": 1_000_000,
            "no_replay_proven": True,
        }
    )
    evidence["evaluation"].update(
        {
            "reservation_identity_sha256": "7" * 64,
            "decontamination_identity_sha256": "8" * 64,
            "selection_validation_authority_ref": "VERIFY-SELECTION@exact-sha",
            "final_test_firewall_preregistered": True,
        }
    )
    evidence["checkpoint_integrity"].update(
        {
            "authority_ref": "D05@exact-sha",
            "terminal_retest_passed": True,
            "fresh_process_resume_passed": True,
        }
    )
    evidence["learned_ladder"].update(
        {
            "learned_3m_authority_ref": "VERIFY-219@exact-sha",
            "learned_10m_authority_ref": "VERIFY-218@exact-sha",
            "independently_verified": True,
        }
    )
    evidence["training_recipe"].update(
        {
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "precision": "bf16",
            "learning_rate": 3e-4,
            "warmup": "preregistered warmup",
            "gradient_policy": "finite checks plus clipping",
            "seeds": [1234],
            "total_unique_loss_positions": 1_000_000,
            "budget_matches_corpus_ledger": True,
            "stopping_rules": ["nonfinite numerics", "preregistered heldout regression"],
        }
    )
    evidence["resource_envelope"].update(
        {
            "accelerator_profile": "C01-qualified-profile",
            "estimated_flops": 1_000_000_000_000_000,
            "estimated_wall_clock_hours": 2.0,
            "max_cost_usd": 10.0,
            "cost_estimate_authority_ref": "C01@exact-sha",
        }
    )


def test_committed_packet_is_valid_and_blocked() -> None:
    data = _load()
    assert validator.validate_packet(data, repo_root=ROOT) == []
    state, blockers = validator.derive_state(data)
    assert state == "BLOCKED"
    assert "unique_causal_loss_positions_zero" in blockers
    assert "checkpoint_terminal_retest_missing" in blockers


def test_scientific_readiness_is_not_training_authorization() -> None:
    data = _load()
    _fill_scientific_readiness(data)
    data["declared_state"] = "READY_FOR_AUTHORIZATION_REQUEST"
    assert validator.validate_packet(data) == []
    assert validator.derive_state(data) == ("READY_FOR_AUTHORIZATION_REQUEST", [])


def test_explicit_compute_and_training_authorities_are_both_required() -> None:
    data = _load()
    _fill_scientific_readiness(data)
    data["authorizations"]["compute"] = {
        "status": "AUTHORIZED",
        "authority_ref": "COMPUTE_AUTHORIZED@exact-authority",
    }
    data["authorizations"]["training"] = {
        "status": "AUTHORIZED",
        "authority_ref": "TRAINING_AUTHORIZED@exact-authority",
    }
    data["declared_state"] = "TRAINING_AUTHORIZED"
    assert validator.validate_packet(data) == []
    assert validator.derive_state(data) == ("TRAINING_AUTHORIZED", [])


def test_partial_authorization_fails_closed() -> None:
    data = _load()
    _fill_scientific_readiness(data)
    data["authorizations"]["compute"] = {
        "status": "AUTHORIZED",
        "authority_ref": "COMPUTE_AUTHORIZED@exact-authority",
    }
    data["declared_state"] = "BLOCKED"
    assert validator.validate_packet(data) == []
    state, blockers = validator.derive_state(data)
    assert state == "BLOCKED"
    assert blockers == ["authorization_state_inconsistent"]


def test_training_budget_cannot_exceed_unique_corpus_ledger() -> None:
    data = _load()
    _fill_scientific_readiness(data)
    data["launch_evidence"]["training_recipe"]["total_unique_loss_positions"] = 1_000_001
    data["declared_state"] = "BLOCKED"
    assert validator.validate_packet(data) == []
    state, blockers = validator.derive_state(data)
    assert state == "BLOCKED"
    assert "training_budget_exceeds_unique_corpus_ledger" in blockers


def test_modelspec_and_truth_boundary_drift_are_rejected() -> None:
    data = _load()
    mutated = copy.deepcopy(data)
    mutated["launch_evidence"]["model_spec_sha256"] = "0" * 64
    mutated["truth_boundary"]["parameter_count_is_authorization"] = True
    errors = validator.validate_packet(mutated)
    assert any("model_spec_sha256 mismatch" in error for error in errors)
    assert any("parameter_count_is_authorization" in error for error in errors)
