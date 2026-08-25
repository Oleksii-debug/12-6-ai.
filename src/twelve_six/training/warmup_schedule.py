"""TRAIN-43 warmup schedule utilities.

This module is an experiment-side schedule contract around the live AdamW training
harness.  It deliberately keeps the decay horizon independent from a shortened
probe length so warmup comparisons do not silently change cosine decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class WarmupScheduleConfig:
    """Schedule-only settings for controlled warmup experiments."""

    base_learning_rate: float
    warmup_steps: int
    experiment_steps: int
    schedule_horizon_steps: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_learning_rate) or self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be finite and > 0")
        if not isinstance(self.warmup_steps, int) or isinstance(self.warmup_steps, bool):
            raise TypeError("warmup_steps must be an integer")
        if not isinstance(self.experiment_steps, int) or isinstance(self.experiment_steps, bool):
            raise TypeError("experiment_steps must be an integer")
        if not isinstance(self.schedule_horizon_steps, int) or isinstance(
            self.schedule_horizon_steps, bool
        ):
            raise TypeError("schedule_horizon_steps must be an integer")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.experiment_steps <= 0:
            raise ValueError("experiment_steps must be > 0")
        if self.schedule_horizon_steps <= 0:
            raise ValueError("schedule_horizon_steps must be > 0")
        if self.warmup_steps > self.experiment_steps:
            raise ValueError("warmup_steps must be <= experiment_steps")
        if self.schedule_horizon_steps < self.experiment_steps:
            raise ValueError(
                "schedule_horizon_steps must be >= experiment_steps; shortened probes "
                "must not collapse the intended scheduler horizon"
            )
        if self.warmup_steps >= self.schedule_horizon_steps:
            raise ValueError("warmup_steps must be < schedule_horizon_steps")


def warmup_cosine_factor(step_index: int, config: WarmupScheduleConfig) -> float:
    """Return the LR multiplier used before optimizer step ``step_index``.

    ``step_index`` is zero-based.  Warmup reaches exactly 1.0 on its last warmup
    optimizer step.  Cosine progress is measured against ``schedule_horizon_steps``
    rather than ``experiment_steps``.
    """

    if not isinstance(step_index, int) or isinstance(step_index, bool):
        raise TypeError("step_index must be an integer")
    if step_index < 0:
        raise ValueError("step_index must be >= 0")
    if step_index >= config.schedule_horizon_steps:
        return 0.0
    if config.warmup_steps and step_index < config.warmup_steps:
        return (step_index + 1) / config.warmup_steps

    denominator = max(config.schedule_horizon_steps - config.warmup_steps, 1)
    progress = (step_index - config.warmup_steps) / denominator
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def learning_rate_for_step(step_index: int, config: WarmupScheduleConfig) -> float:
    return config.base_learning_rate * warmup_cosine_factor(step_index, config)


def apply_learning_rate(
    optimizer: Optimizer,
    step_index: int,
    config: WarmupScheduleConfig,
) -> float:
    """Set every optimizer parameter group to the controlled experiment LR."""

    learning_rate = learning_rate_for_step(step_index, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate
