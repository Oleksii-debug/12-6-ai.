"""Training, optimization, and numerical-safety primitives for 12-6 AI."""

from .config import PrecisionMode, SchedulerKind, TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss
from .memory import (
    TrainingTensorMemory,
    gradient_tensor_bytes,
    measure_training_tensor_memory,
    optimizer_tensor_bytes,
    parameter_tensor_bytes,
    process_rss_bytes,
    scaler_state_metadata,
    scaler_tensor_bytes,
    tensor_nbytes,
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
    "TrainingTensorMemory",
    "build_optimizer",
    "build_scheduler",
    "causal_lm_loss",
    "causal_pair_loss",
    "gradient_tensor_bytes",
    "measure_training_tensor_memory",
    "optimizer_tensor_bytes",
    "parameter_tensor_bytes",
    "process_rss_bytes",
    "scaler_state_metadata",
    "scaler_tensor_bytes",
    "tensor_nbytes",
]
