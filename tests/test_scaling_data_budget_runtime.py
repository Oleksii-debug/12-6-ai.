from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from twelve_six.data_budget import (
    evaluate_policy_stage,
    required_unique_loss_positions,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/scaling/data_budget_policy_v1.json"


def load_policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def test_exact_model_341_20x_reference() -> None:
    assert required_unique_loss_positions(20_613_440, 20) == 412_268_800


def test_zero_materialized_positions_fail_closed() -> None:
    result = evaluate_policy_stage(
        policy=load_policy(),
        stage_name="20M_PRIMARY",
        unique_loss_positions=0,
        multiplier=20,
    )
    assert result.data_budget_tier_met is False
    assert result.shortfall_positions == 412_268_800
    assert result.progress_fraction == 0.0
    assert result.training_authorized is False
    assert result.paid_compute_authorized is False


def test_one_position_short_remains_blocked() -> None:
    result = evaluate_policy_stage(
        policy=load_policy(),
        stage_name="20M_PRIMARY",
        unique_loss_positions=412_268_799,
        multiplier=20,
    )
    assert result.data_budget_tier_met is False
    assert result.shortfall_positions == 1


def test_exact_tier_can_be_met_without_authorizing_training() -> None:
    result = evaluate_policy_stage(
        policy=load_policy(),
        stage_name="20M_PRIMARY",
        unique_loss_positions=412_268_800,
        multiplier=20,
    )
    assert result.data_budget_tier_met is True
    assert result.shortfall_positions == 0
    assert result.observed_positions_per_parameter == 20.0
    assert result.training_authorized is False
    assert result.paid_compute_authorized is False


def test_future_100m_50x_stage_is_derived_from_policy() -> None:
    result = evaluate_policy_stage(
        policy=load_policy(),
        stage_name="100M_TARGET",
        unique_loss_positions=5_000_000_000,
        multiplier=50,
    )
    assert result.data_budget_tier_met is True
    assert result.required_unique_loss_positions == 5_000_000_000
    expected_flops = 3_000_000_000_000_000_000
    assert result.approximate_dense_training_flops_at_tier == expected_flops


def test_unpreregistered_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="not preregistered"):
        evaluate_policy_stage(
            policy=load_policy(),
            stage_name="20M_PRIMARY",
            unique_loss_positions=0,
            multiplier=5,
        )


def test_policy_reference_drift_is_rejected() -> None:
    policy = copy.deepcopy(load_policy())
    policy["stages"][0]["reference_unique_loss_tokens"]["20x"] -= 1
    with pytest.raises(ValueError, match="reference drift"):
        evaluate_policy_stage(
            policy=policy,
            stage_name="20M_PRIMARY",
            unique_loss_positions=412_268_800,
            multiplier=20,
        )


def test_paid_compute_truth_boundary_cannot_be_weakened() -> None:
    policy = copy.deepcopy(load_policy())
    policy["truth_boundary"]["paid_compute_authorized"] = True
    with pytest.raises(ValueError, match="must be false"):
        evaluate_policy_stage(
            policy=policy,
            stage_name="20M_PRIMARY",
            unique_loss_positions=412_268_800,
            multiplier=20,
        )


@pytest.mark.parametrize("bad", [True, -1])
def test_invalid_unique_position_counts_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="unique_loss_positions"):
        evaluate_policy_stage(
            policy=load_policy(),
            stage_name="20M_PRIMARY",
            unique_loss_positions=bad,  # type: ignore[arg-type]
            multiplier=20,
        )
