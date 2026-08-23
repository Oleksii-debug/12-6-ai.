"""Configuration contracts for the model-agnostic 12-6 trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

PrecisionMode = Literal["fp32", "bf16", "fp16"]
SchedulerKind = Literal["constant", "linear_warmup", "cosine"]

_PRECISION_MODES = frozenset({"fp32", "bf16", "fp16"})
_SCHEDULER_KINDS = frozenset({"constant", "linear_warmup", "cosine"})


def _require_finite(name: str, value: float, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if not positive and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Numerical and optimization settings owned by D02.

    Dataset semantics and model architecture stay outside this configuration.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_steps: int = 100
    warmup_steps: int = 0
    scheduler: SchedulerKind = "constant"
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    precision: PrecisionMode = "fp32"
    seed: int = 1337
    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = False

    def __post_init__(self) -> None:
        _require_finite("learning_rate", self.learning_rate, positive=True)
        _require_finite("weight_decay", self.weight_decay, positive=False)
        _require_finite("eps", self.eps, positive=True)

        if not isinstance(self.betas, tuple) or len(self.betas) != 2:
            raise ValueError("betas must contain two values in [0, 1)")
        for beta in self.betas:
            if (
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(beta)
                or not 0.0 <= beta < 1.0
            ):
                raise ValueError("betas must contain two finite values in [0, 1)")

        _require_int("max_steps", self.max_steps, minimum=1)
        _require_int("warmup_steps", self.warmup_steps, minimum=0)
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be <= max_steps")
        _require_int(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
            minimum=1,
        )
        _require_int("seed", self.seed, minimum=0)

        if self.gradient_clip_norm is not None:
            _require_finite("gradient_clip_norm", self.gradient_clip_norm, positive=True)
        if self.scheduler not in _SCHEDULER_KINDS:
            raise ValueError(f"unsupported scheduler: {self.scheduler!r}")
        if self.precision not in _PRECISION_MODES:
            raise ValueError(f"unsupported precision: {self.precision!r}")
        if not isinstance(self.deterministic_algorithms, bool):
            raise ValueError("deterministic_algorithms must be bool")
        if not isinstance(self.deterministic_warn_only, bool):
            raise ValueError("deterministic_warn_only must be bool")
