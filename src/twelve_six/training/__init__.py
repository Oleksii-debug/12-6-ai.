"""Training, optimization, numerical-safety, and observability primitives."""

from .config import PrecisionMode, SchedulerKind, TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss
from .observability import (
    TrainingObserver,
    UpdateMagnitude,
    UpdateObservation,
    summarize_parameter_update,
)
from .trainer import (
    CheckpointHookError,
    NonFiniteTrainingError,
    StepMetrics,
    Trainer,
    TrainerState,
    TrainingRunResult,
    TrainingStateInvalidError,
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
    "TrainingStateInvalidError",
    "TrainingObserver",
    "UpdateMagnitude",
    "UpdateObservation",
    "build_optimizer",
    "build_scheduler",
    "causal_lm_loss",
    "causal_pair_loss",
    "summarize_parameter_update",
]
