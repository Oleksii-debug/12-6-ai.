from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.training.budget import (
    TrainingBudgetError,
    evaluate_budget_readiness,
    load_training_budget_contract,
    validate_corpus_capacity,
    validate_training_budget_contract,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "configs" / "training" / "model341_20m_training_budget.v1.json"
MODEL_IDENTITY = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
DATASET_IDENTITY = "a" * 64


def _terminal_capacity(loss_positions: int) -> dict[str, object]:
    return {
        "postpack_loss_positions": loss_positions,
        "dataset_manifest_sha256": DATASET_IDENTITY,
        "rights_status": "PASS",
        "dedup_status": "PASS",
        "contamination_status": "PASS",
        "split_status": "PASS",
        "replay_policy": "NO_REPLAY",
    }


def test_model341_budget_is_exact_and_model_bound() -> None:
    contract = load_training_budget_contract(BUDGET_PATH)

    assert contract.candidate == "MODEL-341-20M-CANDIDATE-A"
    assert contract.parameter_count == 20_613_440
    assert contract.model_identity_sha256 == MODEL_IDENTITY
    assert contract.point("engineering_1x").loss_positions == 20_613_440
    assert contract.point("calibration_5x").loss_positions == 103_067_200
    assert contract.point("calibration_10x").loss_positions == 206_134_400
    assert contract.point("chinchilla_reference_20x").loss_positions == 412_268_800


def test_budget_contract_rejects_arithmetic_drift() -> None:
    payload = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    payload["budget_points"][-1]["loss_positions"] += 1

    with pytest.raises(TrainingBudgetError, match="arithmetic mismatch"):
        validate_training_budget_contract(payload)


def test_budget_contract_rejects_model_identity_drift() -> None:
    payload = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    payload["candidate"]["model_identity_sha256"] = "not-a-digest"

    with pytest.raises(TrainingBudgetError, match="SHA-256"):
        validate_training_budget_contract(payload)


def test_raw_or_nonterminal_capacity_cannot_be_substituted() -> None:
    capacity = _terminal_capacity(20_613_440)
    del capacity["postpack_loss_positions"]
    capacity["raw_bytes"] = 1_000_000_000

    with pytest.raises(TrainingBudgetError, match="postpack_loss_positions"):
        validate_corpus_capacity(capacity)

    capacity = _terminal_capacity(20_613_440)
    capacity["contamination_status"] = "NOT_TESTED"
    with pytest.raises(TrainingBudgetError, match="contamination_status"):
        validate_corpus_capacity(capacity)


def test_capacity_does_not_authorize_training() -> None:
    contract = load_training_budget_contract(BUDGET_PATH)
    capacity = validate_corpus_capacity(_terminal_capacity(500_000_000))

    readiness = evaluate_budget_readiness(
        contract,
        capacity,
        budget_name="chinchilla_reference_20x",
    )

    assert readiness.capacity_satisfied is True
    assert readiness.available_loss_positions == 500_000_000
    assert readiness.required_loss_positions == 412_268_800
    assert readiness.training_authorized is False


def test_insufficient_terminal_capacity_fails_only_capacity_gate() -> None:
    contract = load_training_budget_contract(BUDGET_PATH)
    capacity = validate_corpus_capacity(_terminal_capacity(20_000_000))

    readiness = evaluate_budget_readiness(
        contract,
        capacity,
        budget_name="engineering_1x",
    )

    assert readiness.capacity_satisfied is False
    assert readiness.training_authorized is False
