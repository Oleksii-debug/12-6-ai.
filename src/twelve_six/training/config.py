"""Configuration contracts for the model-agnostic 12-6 trainer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

PrecisionMode = Literal["fp32", "bf16", "fp16"]
SchedulerKind = Literal["constant", "linear_warmup", "cosine"]

_PRECISION_MODES = frozenset({"fp32", "bf16", "fp16"})
_SCHEDULER_KINDS = frozenset({"constant", "linear_warmup", "cosine"})


def _require_finite(name: str, value: float, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if not positive and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


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

        if not isinstance(self.betas, tuple):
            raise TypeError("betas must be a tuple")
        if len(self.betas) != 2:
            raise ValueError("betas must contain two values in [0, 1)")
        for beta in self.betas:
            if isinstance(beta, bool) or not isinstance(beta, (int, float)):
                raise TypeError("betas must contain numeric values")
            if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
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
            raise TypeError("deterministic_algorithms must be bool")
        if not isinstance(self.deterministic_warn_only, bool):
            raise TypeError("deterministic_warn_only must be bool")


@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    """Versioned scientific inputs, independent of device and provider settings."""

    trainer: TrainerConfig
    microbatch_size: int
    sequence_length: int
    target_unique_loss_positions: int
    maximum_total_exposures: int
    max_exposures_per_unique_position: int

    def __post_init__(self) -> None:
        if type(self.trainer) is not TrainerConfig:
            raise TypeError("trainer must be a TrainerConfig")
        _require_int("microbatch_size", self.microbatch_size, minimum=1)
        _require_int("sequence_length", self.sequence_length, minimum=2)
        _require_int("target_unique_loss_positions", self.target_unique_loss_positions, minimum=1)
        _require_int("maximum_total_exposures", self.maximum_total_exposures, minimum=1)
        _require_int(
            "max_exposures_per_unique_position", self.max_exposures_per_unique_position, minimum=1
        )
        if not (
            self.target_unique_loss_positions <= self.maximum_total_exposures
            <= self.target_unique_loss_positions * self.max_exposures_per_unique_position
        ):
            raise ValueError("recipe exposure budget disagrees with unique positions and replay cap")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trainer"]["betas"] = list(self.trainer.betas)
        return {"schema_version": 1, "optimizer": "AdamW", **payload}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingRecipe:
        recipe_keys = {field.name for field in fields(cls)}
        if type(value) is not dict or value.keys() != recipe_keys | {"schema_version", "optimizer"}:
            raise ValueError("recipe fields must match schema v1 exactly")
        version = value["schema_version"]
        if type(version) is not int or version != 1 or value["optimizer"] != "AdamW":
            raise ValueError("unsupported recipe version or optimizer")
        trainer = value["trainer"]
        if type(trainer) is not dict or trainer.keys() != {f.name for f in fields(TrainerConfig)}:
            raise ValueError("recipe trainer settings must specify every TrainerConfig field")
        if type(trainer["betas"]) is not list:
            raise ValueError("recipe trainer betas must be a JSON array")
        config = TrainerConfig(**{**trainer, "betas": tuple(trainer["betas"])})
        return cls(**{**{key: value[key] for key in recipe_keys}, "trainer": config})

    def identity_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def run_training_fields(self) -> dict[str, Any]:
        """Build the scientific portion of the incumbent run manifest's training block."""
        config = self.trainer
        return {
            "recipe": self.to_dict(),
            "recipe_sha256": self.identity_sha256(),
            "seed": config.seed,
            "precision": config.precision,
            "optimizer": {
                "name": "AdamW",
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "betas": list(config.betas),
                "eps": config.eps,
            },
            "scheduler": {"name": config.scheduler, "warmup_steps": config.warmup_steps},
            "context_length": self.sequence_length,
            "global_batch_tokens": (
                self.microbatch_size * self.sequence_length * config.gradient_accumulation_steps
            ),
            "target_steps": config.max_steps,
            "target_tokens": self.maximum_total_exposures,
        }
