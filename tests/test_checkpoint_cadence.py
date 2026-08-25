from __future__ import annotations

import pytest

from twelve_six.checkpoint.cadence import (
    checkpoint_overhead_percent,
    choose_cadence,
    estimate_cadence,
    interval_for_recompute_target,
    robust_seconds,
)


def test_interval_respects_wall_time_target() -> None:
    assert interval_for_recompute_target(step_seconds=0.25, max_recompute_seconds=5.0) == 20
    assert interval_for_recompute_target(step_seconds=2.0, max_recompute_seconds=0.5) == 1


def test_overhead_uses_train_plus_checkpoint_denominator() -> None:
    value = checkpoint_overhead_percent(
        step_seconds=0.5,
        checkpoint_seconds=0.5,
        interval_steps=9,
    )
    assert value == pytest.approx(10.0)


def test_estimate_reports_conservative_lost_work_envelope() -> None:
    estimate = estimate_cadence(
        step_seconds=0.2,
        checkpoint_seconds=0.1,
        max_recompute_seconds=1.0,
    )
    assert estimate.interval_steps == 5
    assert estimate.lost_work_steps == 5
    assert estimate.lost_work_seconds == pytest.approx(1.0)
    assert estimate.checkpoint_overhead_percent == pytest.approx(100.0 * 0.1 / 1.1)


def test_choose_cadence_prefers_tightest_target_under_overhead_budget() -> None:
    estimates = [
        estimate_cadence(
            step_seconds=0.1,
            checkpoint_seconds=0.2,
            max_recompute_seconds=target,
        )
        for target in (1.0, 5.0, 30.0)
    ]
    selected = choose_cadence(estimates, max_overhead_percent=5.0)
    assert selected.max_recompute_seconds == 5.0


def test_robust_seconds_is_median_and_rejects_invalid_samples() -> None:
    assert robust_seconds([0.3, 0.1, 0.2]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        robust_seconds([])
    with pytest.raises(ValueError):
        robust_seconds([0.1, 0.0])
