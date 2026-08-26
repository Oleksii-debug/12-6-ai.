from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from twelve_six.data_budget import (
    evaluate_policy_stage,
    required_training_token_exposures,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/scaling/data_budget_policy_v1.json"


def load_policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def _evaluate(
    *,
    unique: int = 20_000_000,
    exposures: int = 412_268_800,
    multiplier: int = 20,
):
    return evaluate_policy_stage(
        policy=load_policy(),
        stage_name="20M_PRIMARY",
        unique_loss_positions=unique,
        planned_training_token_exposures=exposures,
        multiplier=multiplier,
    )


def test_exact_model_341_20x_exposure_reference() -> None:
    assert required_training_token_exposures(20_613_440, 20) == 412_268_800


def test_unique_positions_do_not_have_to_equal_exposure_reference() -> None:
    result = _evaluate(unique=20_000_000, exposures=412_268_800)
    assert result.reference_exposure_match is True
    assert result.unique_loss_positions == 20_000_000
    assert result.reference_training_token_exposures == 412_268_800
    assert result.implied_exposures_per_unique_position == pytest.approx(20.61344)
    assert result.training_authorized is False
    assert result.paid_compute_authorized is False


def test_zero_unique_positions_do_not_turn_exposure_reference_into_unique_requirement() -> None:
    result = _evaluate(unique=0, exposures=412_268_800)
    assert result.reference_exposure_match is True
    assert result.implied_exposures_per_unique_position is None
    assert result.training_authorized is False


def test_below_reference_is_descriptive_not_training_verdict() -> None:
    result = _evaluate(unique=20_000_000, exposures=400_000_000)
    assert result.exposure_relation == "BELOW_REFERENCE"
    assert result.exposure_delta == -12_268_800
    assert result.reference_exposure_match is False
    assert result.training_authorized is False


def test_above_reference_is_descriptive_not_training_verdict() -> None:
    result = _evaluate(unique=20_000_000, exposures=500_000_000)
    assert result.exposure_relation == "ABOVE_REFERENCE"
    assert result.exposure_delta == 87_731_200
    assert result.training_authorized is False


def test_future_100m_50x_stage_uses_total_exposures() -> None:
    result = evaluate_policy_stage(
        policy=load_policy(),
        stage_name="100M_TARGET",
        unique_loss_positions=250_000_000,
        planned_training_token_exposures=5_000_000_000,
        multiplier=50,
    )
    assert result.reference_exposure_match is True
    assert result.reference_training_token_exposures == 5_000_000_000
    assert result.approximate_dense_training_flops_at_reference == 3_000_000_000_000_000_000


def test_unpreregistered_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="not preregistered"):
        _evaluate(multiplier=5)


def test_policy_reference_drift_is_rejected() -> None:
    policy = copy.deepcopy(load_policy())
    policy["stages"][0]["reference_training_token_exposures"]["20x"] -= 1
    with pytest.raises(ValueError, match="reference drift"):
        evaluate_policy_stage(
            policy=policy,
            stage_name="20M_PRIMARY",
            unique_loss_positions=20_000_000,
            planned_training_token_exposures=412_268_800,
            multiplier=20,
        )


def test_exposure_unique_conflation_truth_boundary_cannot_be_weakened() -> None:
    policy = copy.deepcopy(load_policy())
    policy["truth_boundary"]["training_token_exposures_are_unique_loss_positions"] = True
    with pytest.raises(ValueError, match="must be false"):
        evaluate_policy_stage(
            policy=policy,
            stage_name="20M_PRIMARY",
            unique_loss_positions=20_000_000,
            planned_training_token_exposures=412_268_800,
            multiplier=20,
        )


def test_paid_compute_truth_boundary_cannot_be_weakened() -> None:
    policy = copy.deepcopy(load_policy())
    policy["truth_boundary"]["paid_compute_authorized"] = True
    with pytest.raises(ValueError, match="must be false"):
        evaluate_policy_stage(
            policy=policy,
            stage_name="20M_PRIMARY",
            unique_loss_positions=20_000_000,
            planned_training_token_exposures=412_268_800,
            multiplier=20,
        )


@pytest.mark.parametrize("bad", [True, -1])
def test_invalid_unique_position_counts_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="unique_loss_positions"):
        evaluate_policy_stage(
            policy=load_policy(),
            stage_name="20M_PRIMARY",
            unique_loss_positions=bad,  # type: ignore[arg-type]
            planned_training_token_exposures=412_268_800,
            multiplier=20,
        )


@pytest.mark.parametrize("bad", [True, -1])
def test_invalid_exposure_counts_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="planned_training_token_exposures"):
        evaluate_policy_stage(
            policy=load_policy(),
            stage_name="20M_PRIMARY",
            unique_loss_positions=20_000_000,
            planned_training_token_exposures=bad,  # type: ignore[arg-type]
            multiplier=20,
        )
