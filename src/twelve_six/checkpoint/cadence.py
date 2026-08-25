"""Measured checkpoint-cadence economics for training reliability.

This module deliberately does not define a checkpoint format.  It only turns
measured trainer-step/checkpoint latencies into bounded lost-work and overhead
estimates for the incumbent checkpoint implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CadenceEstimate:
    max_recompute_seconds: float
    interval_steps: int
    lost_work_steps: int
    lost_work_seconds: float
    checkpoint_overhead_percent: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def robust_seconds(samples: Iterable[float]) -> float:
    """Return a positive median latency from one or more measurements."""

    values = [float(value) for value in samples]
    if not values:
        raise ValueError("at least one latency sample is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("latency samples must be finite and > 0")
    return float(median(values))


def interval_for_recompute_target(
    *,
    step_seconds: float,
    max_recompute_seconds: float,
) -> int:
    """Largest checkpoint interval whose pre-checkpoint work fits the target.

    If one optimizer step itself exceeds the requested target, checkpoint every
    step.  The resulting wall-time envelope then necessarily exceeds the target
    by one step duration; callers can detect that from :func:`estimate_cadence`.
    """

    if not math.isfinite(step_seconds) or step_seconds <= 0.0:
        raise ValueError("step_seconds must be finite and > 0")
    if not math.isfinite(max_recompute_seconds) or max_recompute_seconds <= 0.0:
        raise ValueError("max_recompute_seconds must be finite and > 0")
    return max(1, int(math.floor(max_recompute_seconds / step_seconds)))


def checkpoint_overhead_percent(
    *,
    step_seconds: float,
    checkpoint_seconds: float,
    interval_steps: int,
) -> float:
    """Synchronous checkpoint share of train+checkpoint wall time."""

    if not math.isfinite(step_seconds) or step_seconds <= 0.0:
        raise ValueError("step_seconds must be finite and > 0")
    if not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0.0:
        raise ValueError("checkpoint_seconds must be finite and > 0")
    if not isinstance(interval_steps, int) or isinstance(interval_steps, bool) or interval_steps <= 0:
        raise ValueError("interval_steps must be a positive integer")
    training = interval_steps * step_seconds
    return 100.0 * checkpoint_seconds / (training + checkpoint_seconds)


def estimate_cadence(
    *,
    step_seconds: float,
    checkpoint_seconds: float,
    max_recompute_seconds: float,
) -> CadenceEstimate:
    interval = interval_for_recompute_target(
        step_seconds=step_seconds,
        max_recompute_seconds=max_recompute_seconds,
    )
    # A failure immediately before the next checkpoint can discard every step
    # since the prior checkpoint, so the conservative envelope is interval steps.
    lost_steps = interval
    lost_seconds = interval * step_seconds
    return CadenceEstimate(
        max_recompute_seconds=float(max_recompute_seconds),
        interval_steps=interval,
        lost_work_steps=lost_steps,
        lost_work_seconds=lost_seconds,
        checkpoint_overhead_percent=checkpoint_overhead_percent(
            step_seconds=step_seconds,
            checkpoint_seconds=checkpoint_seconds,
            interval_steps=interval,
        ),
    )


def choose_cadence(
    estimates: Iterable[CadenceEstimate],
    *,
    max_overhead_percent: float = 5.0,
) -> CadenceEstimate:
    """Choose the tightest lost-work target that stays within overhead budget.

    If no candidate meets the requested overhead budget, return the lowest
    overhead candidate rather than silently inventing a new target.
    """

    values = list(estimates)
    if not values:
        raise ValueError("at least one cadence estimate is required")
    if not math.isfinite(max_overhead_percent) or max_overhead_percent <= 0.0:
        raise ValueError("max_overhead_percent must be finite and > 0")
    ordered = sorted(values, key=lambda item: item.max_recompute_seconds)
    for item in ordered:
        if item.checkpoint_overhead_percent <= max_overhead_percent:
            return item
    return min(ordered, key=lambda item: item.checkpoint_overhead_percent)
