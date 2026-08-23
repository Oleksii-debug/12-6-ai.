"""12-6 AI model package."""

from .model import (
    CausalLMOutput,
    ModelSpec,
    StageConfig,
    TwelveSixDecoder,
    count_trainable_parameters,
    load_stage_config,
)

__all__ = [
    "CausalLMOutput",
    "ModelSpec",
    "StageConfig",
    "TwelveSixDecoder",
    "count_trainable_parameters",
    "load_stage_config",
]
