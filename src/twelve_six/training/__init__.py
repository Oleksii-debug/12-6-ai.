"""Training, optimization, and numerical-safety primitives for 12-6 AI."""

from .config import PrecisionMode, SchedulerKind, TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss
from .trainer import (
    NonFiniteTrainingError,
    StepMetrics,
    Trainer,
    TrainerState,
    build_optimizer,
    build_scheduler,
)

__all__ = [
    "NonFiniteTrainingError",
    "PrecisionMode",
    "SchedulerKind",
    "StepMetrics",
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "build_optimizer",
    "build_scheduler",
    "causal_lm_loss",
    "causal_pair_loss",
]
