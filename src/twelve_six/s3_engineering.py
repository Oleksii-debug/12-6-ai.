"""Execution-only bindings for live S3 scale candidates.

SCALE-03 does not own canonical stage configuration. The current executable S3
geometry is owned by SCALE-02 PR #144 and is reproduced here only as an exact
semantic binding so the ~10M vertical can run on the stronger #89 integrated
base without editing #144 files. D11 PR #67 remains the future-tokenizer
architecture alternative. Every binding fails closed on semantic/count drift.
"""

from __future__ import annotations

from typing import Any

from twelve_six.model import InitSpec, ModelSpec

S3_CURRENT_SOURCE_PR = 144
S3_CURRENT_SOURCE_SHA = "0721728cc40cf736205ae11a998ca177cc1e5ed9"
S3_CURRENT_CANDIDATE_ID = "S3-SCALE02-BYTE-GQA-v1"
S3_CURRENT_EXPECTED_PARAMETERS = 10_000_640
S3_CURRENT_MODEL_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
S3_INIT_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"

S3_D11_SOURCE_PR = 67
S3_D11_SOURCE_SHA = "2728b762cd998ca99403e84c6e4b33e6ae8ed29b"
S3_D11_CANDIDATE_ID = "S3-D11-EXPLICIT-Q-GQA-v1"
S3_D11_EXPECTED_PARAMETERS = 9_999_680
S3_D11_MODEL_SHA256 = "ebf3a73851c273211ff9f5f242d28afe22b109e22aacb998e5c0e86d5ff09a55"

S4_D11_EXPECTED_PARAMETERS = 99_797_760
S4_D11_MODEL_SHA256 = "d6ce8b0f44d5601c56fa0b39bfe77cc8863203d3c6ee32701cf897b5a80ab979"

_S3_CURRENT_MODEL_PAYLOAD: dict[str, Any] = {
    "schema_version": 1,
    "vocab_size": 256,
    "max_seq_len": 1024,
    "d_model": 256,
    "n_layers": 12,
    "n_heads": 8,
    "n_kv_heads": 2,
    "head_dim": 32,
    "d_ff": 864,
    "activation": "swiglu",
    "norm_kind": "rmsnorm",
    "norm_placement": "pre",
    "norm_eps": 1e-5,
    "position_embedding": "rope",
    "rope_theta": 10_000.0,
    "rope_rotary_dim": 32,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "final_norm": True,
    "tie_word_embeddings": True,
    "lm_head_bias": False,
}

_S3_D11_MODEL_PAYLOAD: dict[str, Any] = {
    "schema_version": 1,
    "vocab_size": 8192,
    "max_seq_len": 2048,
    "d_model": 320,
    "n_layers": 8,
    "n_heads": 6,
    "n_kv_heads": 2,
    "head_dim": 48,
    "d_ff": 704,
    "activation": "swiglu",
    "norm_kind": "rmsnorm",
    "norm_placement": "pre",
    "norm_eps": 1e-5,
    "position_embedding": "rope",
    "rope_theta": 10_000.0,
    "rope_rotary_dim": 48,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "final_norm": True,
    "tie_word_embeddings": True,
    "lm_head_bias": False,
}

_S4_MODEL_PAYLOAD: dict[str, Any] = {
    "schema_version": 1,
    "vocab_size": 32768,
    "max_seq_len": 4096,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "n_kv_heads": 4,
    "head_dim": 64,
    "d_ff": 2016,
    "activation": "swiglu",
    "norm_kind": "rmsnorm",
    "norm_placement": "pre",
    "norm_eps": 1e-5,
    "position_embedding": "rope",
    "rope_theta": 10_000.0,
    "rope_rotary_dim": 64,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "final_norm": True,
    "tie_word_embeddings": True,
    "lm_head_bias": False,
}


def s3_current_model_spec() -> ModelSpec:
    """Return the exact live byte-compatible S3 execution geometry from #144."""

    spec = ModelSpec.from_dict(_S3_CURRENT_MODEL_PAYLOAD)
    if spec.identity_sha256() != S3_CURRENT_MODEL_SHA256:
        raise RuntimeError("current S3 ModelSpec identity drifted")
    if spec.parameter_count() != S3_CURRENT_EXPECTED_PARAMETERS:
        raise RuntimeError("current S3 analytic parameter count drifted")
    return spec


def s3_init_spec() -> InitSpec:
    """Return the shared scratch InitSpec and verify its semantic identity."""

    init = InitSpec()
    if init.identity_sha256() != S3_INIT_SHA256:
        raise RuntimeError("S3 InitSpec identity drifted")
    return init


def s3_d11_model_spec() -> ModelSpec:
    """Return D11's future-tokenizer S3 alternative for explicit comparison."""

    spec = ModelSpec.from_dict(_S3_D11_MODEL_PAYLOAD)
    if spec.identity_sha256() != S3_D11_MODEL_SHA256:
        raise RuntimeError("D11 S3 ModelSpec identity drifted")
    if spec.parameter_count() != S3_D11_EXPECTED_PARAMETERS:
        raise RuntimeError("D11 S3 analytic parameter count drifted")
    return spec


def s4_d11_model_spec() -> ModelSpec:
    """Return D11's ~100M handoff geometry without instantiating it."""

    spec = ModelSpec.from_dict(_S4_MODEL_PAYLOAD)
    if spec.identity_sha256() != S4_D11_MODEL_SHA256:
        raise RuntimeError("D11 S4 ModelSpec identity drifted")
    if spec.parameter_count() != S4_D11_EXPECTED_PARAMETERS:
        raise RuntimeError("D11 S4 analytic parameter count drifted")
    return spec


def parameter_storage_bytes(spec: ModelSpec, *, bytes_per_parameter: int) -> int:
    """Exact dense parameter storage for one materialized parameter copy."""

    if bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be positive")
    return spec.parameter_count() * bytes_per_parameter


def kv_cache_bytes(
    spec: ModelSpec,
    *,
    batch_size: int,
    sequence_length: int,
    bytes_per_element: int,
) -> int:
    """Exact unexpanded GQA K/V payload size for a model-native cache."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= sequence_length <= spec.max_seq_len:
        raise ValueError("sequence_length must be within model context")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")
    return (
        2
        * spec.n_layers
        * batch_size
        * spec.n_kv_heads
        * sequence_length
        * spec.head_dim
        * bytes_per_element
    )
