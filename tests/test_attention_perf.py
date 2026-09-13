from __future__ import annotations

import pytest
import torch

from twelve_six.attention_perf import (
    AttentionGeometry,
    expanded_kv_tensor_bytes,
    infer_geometry,
    kv_tensor_bytes,
    sdpa_expanded_reference,
    sdpa_native_gqa,
)


def _qkv(*, q_heads: int, kv_heads: int, seq: int = 32, dim: int = 16):
    torch.manual_seed(123)
    q = torch.randn(2, q_heads, seq, dim)
    k = torch.randn(2, kv_heads, seq, dim)
    v = torch.randn(2, kv_heads, seq, dim)
    return q, k, v


def test_mha_native_path_matches_current_sdpa_exactly() -> None:
    q, k, v = _qkv(q_heads=4, kv_heads=4)
    expected = sdpa_expanded_reference(q, k, v, is_causal=True)
    actual = sdpa_native_gqa(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_gqa_native_path_matches_expanded_forward_and_backward() -> None:
    base = _qkv(q_heads=4, kv_heads=2)
    results = []
    for fn in (sdpa_expanded_reference, sdpa_native_gqa):
        q, k, v = (tensor.clone().requires_grad_() for tensor in base)
        output = fn(q, k, v, is_causal=True)
        output.square().mean().backward()
        results.append((output.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()))

    for expected, actual in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_decode_one_uses_noncausal_rectangular_attention_semantics() -> None:
    q, _, _ = _qkv(q_heads=4, kv_heads=2, seq=1)
    torch.manual_seed(456)
    k = torch.randn(2, 2, 17, 16)
    v = torch.randn(2, 2, 17, 16)
    expected = sdpa_expanded_reference(q, k, v, is_causal=False)
    actual = sdpa_native_gqa(q, k, v, is_causal=False)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_transposed_projection_layout_is_accepted_without_forced_contiguous_copy() -> None:
    batch, seq, q_heads, kv_heads, dim = 2, 24, 4, 2, 16
    q_source = torch.randn(batch, seq, q_heads, dim)
    k_source = torch.randn(batch, seq, kv_heads, dim)
    v_source = torch.randn(batch, seq, kv_heads, dim)
    q = q_source.transpose(1, 2)
    k = k_source.transpose(1, 2)
    v = v_source.transpose(1, 2)
    assert not q.is_contiguous()
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1

    expected = sdpa_expanded_reference(q, k, v, is_causal=True)
    actual = sdpa_native_gqa(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_10m_100m_400m_materialized_kv_bytes_are_honest() -> None:
    geometries = {
        "10m": (8, 8, 1024, 40, 1_310_720, 1_310_720),
        "100m": (12, 12, 2048, 64, 6_291_456, 6_291_456),
        "400m": (16, 4, 4096, 64, 4_194_304, 16_777_216),
    }
    for stage, (q_heads, kv_heads, seq, dim, expected_native, expected_expanded) in geometries.items():
        native = kv_tensor_bytes(
            batch_size=1,
            kv_heads=kv_heads,
            sequence_length=seq,
            head_dim=dim,
            dtype=torch.bfloat16,
        )
        expanded = expanded_kv_tensor_bytes(
            batch_size=1,
            query_heads=q_heads,
            sequence_length=seq,
            head_dim=dim,
            dtype=torch.bfloat16,
        )
        assert native == expected_native, stage
        assert expanded == expected_expanded, stage

    assert geometries["400m"][5] == 4 * geometries["400m"][4]


def test_geometry_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="divisible"):
        AttentionGeometry(query_heads=6, kv_heads=4, head_dim=64)

    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 8)
    v = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError, match="head dimensions"):
        infer_geometry(q, k, v)


def test_native_gqa_helper_is_fullgraph_compile_compatible() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    q, k, v = _qkv(q_heads=4, kv_heads=2, seq=8, dim=8)

    def call(q_arg, k_arg, v_arg):
        return sdpa_native_gqa(q_arg, k_arg, v_arg, is_causal=True)

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    expected = call(q, k, v)
    actual = compiled(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
