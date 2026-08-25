from __future__ import annotations

from pathlib import Path

from twelve_six.training.warmup_ablation import (
    EXPECTED_COUNTS,
    load_plan,
    materialized_warmup_steps,
    provisional_rule,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/runs/train43_warmup.experimental.json"


def _row(
    fraction: float,
    *,
    validation_spike: float,
    grad_max: float,
    clip_frequency: float,
    update_max: float,
    recovery_tokens: int,
    final_validation: float,
) -> dict:
    return {
        "warmup_fraction": fraction,
        "status": "PASS",
        "early_validation_loss_spike_nats": validation_spike,
        "early_gradient_norm_max": grad_max,
        "early_clip_frequency": clip_frequency,
        "early_relative_update_l2_max": update_max,
        "recovery_tokens_from_initial_validation": recovery_tokens,
        "final_validation_loss": final_validation,
    }


def test_plan_holds_lr_optimizer_semantics_and_long_horizon_fixed() -> None:
    plan = load_plan(PLAN)
    assert plan["learning_rate"] == 3e-4
    assert plan["optimizer"]["weight_decay"] == 0.0
    assert plan["optimizer"]["betas"] == [0.9, 0.95]
    assert plan["optimizer"]["scheduler"] == "linear_warmup"
    assert [row["stage"] for row in plan["stages"]] == ["S1", "S2"]
    assert [row["expected_parameters"] for row in plan["stages"]] == [
        EXPECTED_COUNTS["S1"],
        EXPECTED_COUNTS["S2"],
    ]
    assert all(
        plan["schedule_horizon_steps"] >= 4 * row["execution_steps"]
        for row in plan["stages"]
    )


def test_warmup_grid_materializes_to_zero_eight_and_twenty_steps() -> None:
    plan = load_plan(PLAN)
    assert materialized_warmup_steps(plan) == {0.0: 0, 0.02: 8, 0.05: 20}


def test_rule_selects_smallest_warmup_only_when_both_scales_materially_improve() -> None:
    baseline = _row(
        0.0,
        validation_spike=0.20,
        grad_max=10.0,
        clip_frequency=0.8,
        update_max=0.020,
        recovery_tokens=1000,
        final_validation=4.0,
    )
    two_percent = _row(
        0.02,
        validation_spike=0.15,
        grad_max=8.0,
        clip_frequency=0.6,
        update_max=0.016,
        recovery_tokens=850,
        final_validation=4.01,
    )
    five_percent = _row(
        0.05,
        validation_spike=0.10,
        grad_max=7.0,
        clip_frequency=0.5,
        update_max=0.014,
        recovery_tokens=800,
        final_validation=4.02,
    )
    stages = [
        {"stage": stage, "results": [dict(baseline), dict(two_percent), dict(five_percent)]}
        for stage in ("S1", "S2")
    ]
    result = provisional_rule(stages)
    assert result["selected_warmup_fraction"] == 0.02


def test_rule_keeps_zero_when_warmup_does_not_help_both_scales() -> None:
    baseline = _row(
        0.0,
        validation_spike=0.0,
        grad_max=10.0,
        clip_frequency=0.8,
        update_max=0.020,
        recovery_tokens=1000,
        final_validation=4.0,
    )
    weak = _row(
        0.02,
        validation_spike=0.0,
        grad_max=9.8,
        clip_frequency=0.79,
        update_max=0.0198,
        recovery_tokens=980,
        final_validation=4.0,
    )
    weaker = dict(weak, warmup_fraction=0.05)
    stages = [
        {"stage": stage, "results": [dict(baseline), dict(weak), dict(weaker)]}
        for stage in ("S1", "S2")
    ]
    result = provisional_rule(stages)
    assert result["selected_warmup_fraction"] == 0.0
