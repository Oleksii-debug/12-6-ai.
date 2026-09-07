"""12-6 AI model package."""

__version__ = "0.2.0-dev"

from .model import (
    CausalLMOutput,
    InitSpec,
    ModelSpec,
    StageConfig,
    TwelveSixDecoder,
    canonical_json_sha256,
    count_trainable_parameters,
    load_stage_config,
)

__all__ = [
    "CausalLMOutput",
    "InitSpec",
    "ModelSpec",
    "StageConfig",
    "TwelveSixDecoder",
    "canonical_json_sha256",
    "count_trainable_parameters",
    "load_stage_config",
]
