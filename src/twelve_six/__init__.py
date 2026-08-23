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
from .scaling import (
    DenseScalingCandidate,
    DenseScalingTemplate,
    solve_dense_scaling_candidates,
)

__all__ = [
    "CausalLMOutput",
    "DenseScalingCandidate",
    "DenseScalingTemplate",
    "InitSpec",
    "ModelSpec",
    "StageConfig",
    "TwelveSixDecoder",
    "canonical_json_sha256",
    "count_trainable_parameters",
    "load_stage_config",
    "solve_dense_scaling_candidates",
]
