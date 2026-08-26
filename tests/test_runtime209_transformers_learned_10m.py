from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from twelve_six.inference.transformers_llama import (
    TransformersInteropError,
    build_llama_interop_plan,
    convert_state_dict_to_llama,
    rope_pairwise_to_llama_permutation,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder

EXPECTED_MODEL_SPEC_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_PARAMETER_COUNT = 10_000_640


def _spec() -> ModelSpec:
    return ModelSpec.from_dict(
        {
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
            "rope_theta": 10000.0,
            "rope_rotary_dim": 32,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "final_norm": True,
            "tie_word_embeddings": True,
            "lm_head_bias": False,
        }
    )


def test_exact_10m_modelspec_maps_without_approximation() -> None:
    spec = _spec()
    assert spec.parameter_count() == EXPECTED_PARAMETER_COUNT
    assert spec.identity_sha256() == EXPECTED_MODEL_SPEC_SHA256

    plan = build_llama_interop_plan(spec)
    config = plan.target_config
    assert plan.source_parameter_count == EXPECTED_PARAMETER_COUNT
    assert plan.source_model_spec_sha256 == EXPECTED_MODEL_SPEC_SHA256
    assert plan.target_architecture == "LlamaForCausalLM"
    assert plan.rope_transform == "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT"
    assert config["vocab_size"] == 256
    assert config["hidden_size"] == 256
    assert config["intermediate_size"] == 864
    assert config["num_hidden_layers"] == 12
    assert config["num_attention_heads"] == 8
    assert config["num_key_value_heads"] == 2
    assert config["head_dim"] == 32
    assert config["max_position_embeddings"] == 1024
    assert config["rms_norm_eps"] == 1e-5
    assert config["rope_parameters"] == {"rope_type": "default", "rope_theta": 10000.0}
    assert config["tie_word_embeddings"] is True


def test_exact_10m_qk_rope_weight_basis_conversion() -> None:
    spec = _spec()
    torch.manual_seed(209)
    model = TwelveSixDecoder(spec)
    source = model.state_dict()
    converted = convert_state_dict_to_llama(spec, source)

    q_permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(heads=spec.n_heads, head_dim=spec.head_dim)
    )
    k_permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(heads=spec.n_kv_heads, head_dim=spec.head_dim)
    )
    assert q_permutation.numel() == 256
    assert k_permutation.numel() == 64

    for layer in range(spec.n_layers):
        torch.testing.assert_close(
            converted[f"model.layers.{layer}.self_attn.q_proj.weight"],
            source[f"blocks.{layer}.attn.q_proj.weight"].index_select(0, q_permutation),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            converted[f"model.layers.{layer}.self_attn.k_proj.weight"],
            source[f"blocks.{layer}.attn.k_proj.weight"].index_select(0, k_permutation),
            atol=0,
            rtol=0,
        )


def test_runtime209_rejects_unsupported_10m_variants_instead_of_approximating() -> None:
    spec = _spec()
    unsupported = (
        replace(spec, rope_rotary_dim=16),
        replace(spec, attention_bias=True),
        replace(spec, mlp_bias=True),
        replace(spec, lm_head_bias=True),
        replace(spec, final_norm=False),
    )
    for candidate in unsupported:
        with pytest.raises(TransformersInteropError):
            build_llama_interop_plan(candidate)
