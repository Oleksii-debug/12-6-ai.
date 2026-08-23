"""Training, optimization, and numerical-safety primitives for 12-6 AI."""

from .config import PrecisionMode, SchedulerKind, TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss
from .trainer import (
    CheckpointHookError,
    NonFiniteTrainingError,
    StepMetrics,
    Trainer,
    TrainerState,
    TrainingRunResult,
    build_optimizer,
    build_scheduler,
)

__all__ = [
    "CheckpointHookError",
    "NonFiniteTrainingError",
    "PrecisionMode",
    "SchedulerKind",
    "StepMetrics",
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "TrainingRunResult",
    "build_optimizer",
    "build_scheduler",
    "causal_lm_loss",
    "causal_pair_loss",
]
