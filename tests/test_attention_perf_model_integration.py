from __future__ import annotations

import torch

from twelve_six.attention_perf import sdpa_native_gqa
from twelve_six.model import CausalSelfAttention, ModelSpec


def _gqa_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        rope_rotary_dim=8,
        attention_dropout=0.0,
    )


def _project_native(attn: CausalSelfAttention, q, k, v, *, is_causal: bool):
    attended = sdpa_native_gqa(
        q,
        k,
        v,
        dropout_p=attn.dropout if attn.training else 0.0,
        is_causal=is_causal,
    )
    batch = q.shape[0]
    seq_len = q.shape[2]
    attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, attn.q_dim)
    return attn.out_proj(attended)


def test_native_gqa_matches_incumbent_training_attention_path() -> None:
    torch.manual_seed(321)
    attn = CausalSelfAttention(_gqa_spec())
    attn.train()
    x = torch.randn(2, 17, 32)
    q, k, v = attn._project_qkv(x, position_offset=0)

    expected = attn._attend(q, k, v, is_causal=True)
    actual = _project_native(attn, q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_native_gqa_matches_incumbent_prefill_and_decode_cache_semantics() -> None:
    torch.manual_seed(654)
    attn = CausalSelfAttention(_gqa_spec())
    attn.eval()
    prompt = torch.randn(1, 11, 32)
    expected_prefill, cache = attn.prefill(prompt)
    q, k, v = attn._project_qkv(prompt, position_offset=0)
    actual_prefill = _project_native(attn, q, k, v, is_causal=True)

    assert cache.key.shape == (1, 2, 11, 8)
    assert cache.value.shape == (1, 2, 11, 8)
    torch.testing.assert_close(actual_prefill, expected_prefill, rtol=1e-6, atol=1e-7)

    next_x = torch.randn(1, 1, 32)
    expected_decode, next_cache = attn.decode_one(next_x, cache)
    q, new_k, new_v = attn._project_qkv(next_x, position_offset=cache.sequence_length)
    key = torch.cat((cache.key, new_k), dim=2)
    value = torch.cat((cache.value, new_v), dim=2)
    actual_decode = _project_native(attn, q, key, value, is_causal=False)

    assert next_cache.sequence_length == 12
    torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-6, atol=1e-7)
