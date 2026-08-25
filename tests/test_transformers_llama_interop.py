from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from twelve_six.inference.transformers_llama import (
    TransformersInteropError,
    build_llama_interop_plan,
    convert_state_dict_to_llama,
    llama_config_dict,
    rope_pairwise_to_llama_permutation,
)
from twelve_six.model import RotaryEmbedding, TwelveSixDecoder, apply_rope, load_stage_config

ROOT = Path(__file__).resolve().parents[1]


def _s0():
    return load_stage_config(ROOT / "configs/stages/s0_10k.json")


def _llama_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def test_s0_llama_plan_is_exact_and_raw_base_safe() -> None:
    stage = _s0()
    plan = build_llama_interop_plan(stage.model)
    config = plan.target_config

    assert plan.schema == "12-6.transformers-llama-interop-plan.v2"
    assert plan.source_model_spec_sha256 == stage.expected_model_identity_sha256
    assert plan.source_parameter_count == 10_140
    assert plan.target_architecture == "LlamaForCausalLM"
    assert plan.rope_transform == "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT"
    assert plan.runtime_status == "LOCKED_RUNTIME_REQUIRED"
    assert plan.runtime_parity_required is True
    assert len(plan.identity_sha256()) == 64

    assert config["model_type"] == "llama"
    assert config["vocab_size"] == 256
    assert config["hidden_size"] == 20
    assert config["intermediate_size"] == 56
    assert config["num_hidden_layers"] == 1
    assert config["num_attention_heads"] == 2
    assert config["num_key_value_heads"] == 2
    assert config["head_dim"] == 10
    assert config["max_position_embeddings"] == 128
    assert config["hidden_act"] == "silu"
    assert config["rms_norm_eps"] == 1e-5
    assert config["rope_parameters"] == {"rope_type": "default", "rope_theta": 10_000.0}
    assert "rope_theta" not in config
    assert config["bos_token_id"] is None
    assert config["eos_token_id"] is None
    assert config["pad_token_id"] is None

    q_rows = [row for row in plan.tensor_map if row["source"].endswith("q_proj.weight")]
    k_rows = [row for row in plan.tensor_map if row["source"].endswith("k_proj.weight")]
    assert len(q_rows) == len(k_rows) == stage.model.n_layers
    assert all(row["transform"] == plan.rope_transform for row in q_rows + k_rows)


def test_rope_permutation_proves_pairwise_to_llama_basis_equivalence_exactly() -> None:
    stage = _s0()
    spec = stage.model
    torch.manual_seed(17)
    x = torch.randn(2, spec.n_heads, 5, spec.head_dim)
    rope = RotaryEmbedding(spec.rope_rotary_dim, spec.rope_theta)
    cos, sin = rope.cos_sin(5, device=x.device, dtype=x.dtype)

    source_rotated = apply_rope(x, cos, sin, spec.rope_rotary_dim)
    per_head = torch.tensor(
        rope_pairwise_to_llama_permutation(heads=1, head_dim=spec.head_dim),
        dtype=torch.long,
    )
    llama_basis = x.index_select(-1, per_head)
    llama_cos = cos.index_select(-1, per_head).view(1, 1, 5, spec.head_dim)
    llama_sin = sin.index_select(-1, per_head).view(1, 1, 5, spec.head_dim)
    llama_rotated = llama_basis * llama_cos + _llama_rotate_half(llama_basis) * llama_sin

    expected = source_rotated.index_select(-1, per_head)
    assert torch.equal(llama_rotated, expected)


def test_state_dict_conversion_is_complete_and_non_mutating() -> None:
    stage = _s0()
    torch.manual_seed(23)
    model = TwelveSixDecoder(stage.model, stage.init)
    source = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    before = {name: tensor.clone() for name, tensor in source.items()}

    converted = convert_state_dict_to_llama(stage.model, source)

    assert set(converted) == {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    assert torch.equal(converted["model.embed_tokens.weight"], source["token_embedding.weight"])
    assert torch.equal(converted["lm_head.weight"], source["lm_head.weight"])
    assert torch.equal(
        converted["model.layers.0.self_attn.v_proj.weight"],
        source["blocks.0.attn.v_proj.weight"],
    )
    assert torch.equal(
        converted["model.layers.0.self_attn.o_proj.weight"],
        source["blocks.0.attn.out_proj.weight"],
    )
    assert not torch.equal(
        converted["model.layers.0.self_attn.q_proj.weight"],
        source["blocks.0.attn.q_proj.weight"],
    )
    assert not torch.equal(
        converted["model.layers.0.self_attn.k_proj.weight"],
        source["blocks.0.attn.k_proj.weight"],
    )
    assert all(torch.equal(source[name], before[name]) for name in source)


def test_qk_weight_permutation_matches_declared_head_layout() -> None:
    stage = _s0()
    model = TwelveSixDecoder(stage.model, stage.init)
    source = model.state_dict()
    converted = convert_state_dict_to_llama(stage.model, source)

    q_permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(
            heads=stage.model.n_heads,
            head_dim=stage.model.head_dim,
        )
    )
    k_permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(
            heads=stage.model.n_kv_heads,
            head_dim=stage.model.head_dim,
        )
    )
    torch.testing.assert_close(
        converted["model.layers.0.self_attn.q_proj.weight"],
        source["blocks.0.attn.q_proj.weight"].index_select(0, q_permutation),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        converted["model.layers.0.self_attn.k_proj.weight"],
        source["blocks.0.attn.k_proj.weight"].index_select(0, k_permutation),
        atol=0,
        rtol=0,
    )


def test_bridge_fails_closed_on_unrepresented_model_semantics() -> None:
    spec = _s0().model
    cases = (
        replace(spec, rope_rotary_dim=8),
        replace(spec, head_dim=12, rope_rotary_dim=12),
        replace(spec, attention_bias=True),
        replace(spec, mlp_bias=True),
        replace(spec, lm_head_bias=True),
        replace(spec, final_norm=False),
    )
    for incompatible in cases:
        with pytest.raises(TransformersInteropError):
            llama_config_dict(incompatible)


def test_state_inventory_and_shape_drift_fail_closed() -> None:
    stage = _s0()
    model = TwelveSixDecoder(stage.model, stage.init)
    source = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    missing = dict(source)
    missing.pop("blocks.0.attn.q_proj.weight")
    with pytest.raises(TransformersInteropError, match="inventory mismatch"):
        convert_state_dict_to_llama(stage.model, missing)

    extra = dict(source)
    extra["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(TransformersInteropError, match="inventory mismatch"):
        convert_state_dict_to_llama(stage.model, extra)

    wrong_shape = dict(source)
    wrong_shape["blocks.0.mlp.up_proj.weight"] = torch.zeros(1, 1)
    with pytest.raises(TransformersInteropError, match="shape mismatch"):
        convert_state_dict_to_llama(stage.model, wrong_shape)
