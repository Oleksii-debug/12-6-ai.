"""Training, optimization, and numerical-safety primitives for 12-6 AI."""

from .config import PrecisionMode, SchedulerKind, TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss
from .precision import PrecisionRuntime, autocast_dtype, resolve_precision_runtime
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
    "PrecisionRuntime",
    "SchedulerKind",
    "StepMetrics",
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "TrainingRunResult",
    "TrainingStateInvalidError",
    "autocast_dtype",
    "build_optimizer",
    "build_scheduler",
    "causal_lm_loss",
    "causal_pair_loss",
    "resolve_precision_runtime",
]
