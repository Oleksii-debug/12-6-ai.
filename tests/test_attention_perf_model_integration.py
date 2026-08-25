from __future__ import annotations

import math
from unittest.mock import patch

import torch
import torch.nn.functional as F

from twelve_six.attention_perf import sdpa_expanded_reference
from twelve_six.model import CausalSelfAttention, ModelSpec, TwelveSixDecoder
from twelve_six.training import Trainer, TrainerConfig


def _spec(*, kv_heads: int) -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=kv_heads,
        head_dim=8,
        d_ff=64,
        rope_rotary_dim=8,
        attention_dropout=0.0,
    )


def _expanded_reference(attn: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    q, k, v = attn._project_qkv(x, position_offset=0)
    attended = sdpa_expanded_reference(
        q,
        k,
        v,
        dropout_p=attn.dropout if attn.training else 0.0,
        is_causal=True,
    )
    batch, _, seq_len, _ = q.shape
    attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, attn.q_dim)
    return attn.out_proj(attended)


def test_mha_production_path_remains_direct_sdpa_and_exact() -> None:
    torch.manual_seed(101)
    attn = CausalSelfAttention(_spec(kv_heads=4)).eval()
    x = torch.randn(2, 13, 32)
    q, k, v = attn._project_qkv(x, position_offset=0)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    expected = expected.transpose(1, 2).contiguous().view(2, 13, attn.q_dim)
    expected = attn.out_proj(expected)

    with patch(
        "twelve_six.model.sdpa_native_gqa",
        side_effect=AssertionError("MHA must not enter the native-GQA helper"),
    ):
        actual = attn._attend(q, k, v, is_causal=True)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_gqa_production_forward_matches_expanded_reference() -> None:
    torch.manual_seed(202)
    attn = CausalSelfAttention(_spec(kv_heads=2)).eval()
    x = torch.randn(2, 17, 32)
    expected = _expanded_reference(attn, x)
    actual = attn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_gqa_production_gradients_match_expanded_reference() -> None:
    torch.manual_seed(303)
    native = CausalSelfAttention(_spec(kv_heads=2)).train()
    reference = CausalSelfAttention(_spec(kv_heads=2)).train()
    reference.load_state_dict(native.state_dict())

    base_x = torch.randn(2, 19, 32)
    native_x = base_x.clone().requires_grad_()
    reference_x = base_x.clone().requires_grad_()

    native_output = native(native_x)
    reference_output = _expanded_reference(reference, reference_x)
    native_output.square().mean().backward()
    reference_output.square().mean().backward()

    torch.testing.assert_close(native_output, reference_output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(native_x.grad, reference_x.grad, rtol=1e-6, atol=1e-7)

    native_grads = dict(native.named_parameters())
    reference_grads = dict(reference.named_parameters())
    assert native_grads.keys() == reference_grads.keys()
    for name in native_grads:
        native_grad = native_grads[name].grad
        reference_grad = reference_grads[name].grad
        assert native_grad is not None, name
        assert reference_grad is not None, name
        torch.testing.assert_close(native_grad, reference_grad, rtol=1e-6, atol=1e-7)


def test_gqa_cached_prefill_and_decode_match_full_prefix() -> None:
    torch.manual_seed(404)
    model = TwelveSixDecoder(_spec(kv_heads=2)).eval()
    prompt = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    full_prompt = model(prompt).logits
    cached_prompt, cache = model.prefill_kv_cache(prompt)
    torch.testing.assert_close(cached_prompt.logits, full_prompt, rtol=1e-6, atol=1e-7)
    assert all(layer.key.shape[1] == 2 for layer in cache.layers)
    assert all(layer.value.shape[1] == 2 for layer in cache.layers)

    sequence = prompt
    for token_id in (15, 16, 17):
        token = torch.tensor([[token_id]], dtype=torch.long)
        cached, cache = model.decode_one_with_kv_cache(token, cache)
        sequence = torch.cat((sequence, token), dim=1)
        full = model(sequence).logits[:, -1:, :]
        torch.testing.assert_close(cached.logits, full, rtol=1e-6, atol=1e-7)
        assert all(layer.key.shape[1] == 2 for layer in cache.layers)


def test_real_trainer_update_runs_through_native_gqa_without_identity_drift() -> None:
    torch.manual_seed(505)
    spec = _spec(kv_heads=2)
    model = TwelveSixDecoder(spec)
    identity_before = model.spec.identity_sha256()
    state_signature_before = {
        name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
    }
    weight_before = model.blocks[0].attn.k_proj.weight.detach().clone()

    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=3e-4,
            weight_decay=0.0,
            max_steps=1,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=505,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            ],
            dtype=torch.long,
        )
    }
    metrics = trainer.train_microbatch(batch)

    assert metrics.optimizer_stepped is True
    assert metrics.optimizer_step == 1
    assert math.isfinite(metrics.loss)
    assert metrics.grad_norm is not None and math.isfinite(metrics.grad_norm)
    assert not torch.equal(model.blocks[0].attn.k_proj.weight.detach(), weight_before)
    assert model.spec.identity_sha256() == identity_before
    assert {
        name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
    } == state_signature_before
