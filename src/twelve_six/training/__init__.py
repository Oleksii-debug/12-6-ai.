"""Training, optimization, and numerical-safety primitives for 12-6 AI."""

from .config import PrecisionMode, SchedulerKind, TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss
from .numeric_forensics import NumericFailureDiagnostics, batch_identity_sha256
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
    "NumericFailureDiagnostics",
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
    "batch_identity_sha256",
    "build_optimizer",
    "build_scheduler",
    "causal_lm_loss",
    "causal_pair_loss",
    "resolve_precision_runtime",
]
