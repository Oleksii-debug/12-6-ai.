"""Configuration contracts for the model-agnostic 12-6 trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PrecisionMode = Literal["fp32", "bf16", "fp16"]
SchedulerKind = Literal["constant", "linear_warmup", "cosine"]


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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if len(self.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps]")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0 when enabled")
