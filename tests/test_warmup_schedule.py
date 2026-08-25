from __future__ import annotations

import math

import pytest

from twelve_six.training.warmup_schedule import (
    WarmupScheduleConfig,
    learning_rate_for_step,
    warmup_cosine_factor,
)


def test_short_probe_does_not_change_cosine_horizon() -> None:
    short = WarmupScheduleConfig(
        base_learning_rate=0.01,
        warmup_steps=5,
        experiment_steps=200,
        schedule_horizon_steps=2000,
    )
    longer = WarmupScheduleConfig(
        base_learning_rate=0.01,
        warmup_steps=5,
        experiment_steps=500,
        schedule_horizon_steps=2000,
    )
    for step in (0, 4, 5, 50, 199):
        assert warmup_cosine_factor(step, short) == pytest.approx(
            warmup_cosine_factor(step, longer), rel=0.0, abs=1e-15
        )


def test_linear_warmup_reaches_base_lr_on_last_warmup_step() -> None:
    config = WarmupScheduleConfig(
        base_learning_rate=0.01,
        warmup_steps=5,
        experiment_steps=200,
        schedule_horizon_steps=2000,
    )
    assert learning_rate_for_step(0, config) == pytest.approx(0.002)
    assert learning_rate_for_step(1, config) == pytest.approx(0.004)
    assert learning_rate_for_step(4, config) == pytest.approx(0.01)


def test_no_warmup_starts_at_base_lr() -> None:
    config = WarmupScheduleConfig(
        base_learning_rate=0.01,
        warmup_steps=0,
        experiment_steps=200,
        schedule_horizon_steps=2000,
    )
    assert learning_rate_for_step(0, config) == pytest.approx(0.01)
    assert 0.0 < learning_rate_for_step(199, config) < 0.01


def test_probe_cannot_collapse_schedule_horizon() -> None:
    with pytest.raises(ValueError, match="must be >= experiment_steps"):
        WarmupScheduleConfig(
            base_learning_rate=0.01,
            warmup_steps=5,
            experiment_steps=200,
            schedule_horizon_steps=100,
        )


def test_cosine_is_zero_at_or_beyond_horizon() -> None:
    config = WarmupScheduleConfig(
        base_learning_rate=0.01,
        warmup_steps=5,
        experiment_steps=200,
        schedule_horizon_steps=2000,
    )
    assert warmup_cosine_factor(2000, config) == 0.0
    assert math.isclose(warmup_cosine_factor(2500, config), 0.0)
