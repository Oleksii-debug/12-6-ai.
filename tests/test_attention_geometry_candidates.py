from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from twelve_six.model import TwelveSixDecoder, load_stage_config

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "configs" / "stages"


def _load(name: str):
    return load_stage_config(STAGES / name)


def _full_cache_bytes(stage, *, bytes_per_element: int) -> int:
    spec = stage.model
    return (
        2
        * spec.n_layers
        * spec.n_kv_heads
        * spec.head_dim
        * spec.max_seq_len
        * bytes_per_element
    )


def test_s0_identity_and_attention_geometry_are_unchanged() -> None:
    s0 = _load("s0_10k.json")
    assert s0.expected_parameters == 10_140
    assert s0.model.n_heads == 2
    assert s0.model.n_kv_heads == 2
    assert s0.model.parameter_count() == 10_140
    assert s0.expected_model_identity_sha256 == s0.model.identity_sha256()


def test_s2_candidates_are_matched_and_cache_reductions_are_exact() -> None:
    mha = _load("s2_1m.json")
    gqa = _load("s2_1m_gqa2.candidate.json")
    mqa = _load("s2_1m_mqa1.candidate.json")

    assert mha.model.parameter_count() == 1_066_112
    assert gqa.model.parameter_count() == 1_066_624
    assert mqa.model.parameter_count() == 1_066_112
    relative_delta = (
        abs(gqa.model.parameter_count() - mha.model.parameter_count())
        / mha.model.parameter_count()
    )
    assert relative_delta < 0.0005

    mha_cache = _full_cache_bytes(mha, bytes_per_element=2)
    assert _full_cache_bytes(gqa, bytes_per_element=2) * 2 == mha_cache
    assert _full_cache_bytes(mqa, bytes_per_element=2) * 4 == mha_cache


def test_s3_candidates_are_matched_and_cover_gqa_to_mqa() -> None:
    mha = _load("s3_10m.json")
    gqa4 = _load("s3_10m_gqa4.candidate.json")
    gqa2 = _load("s3_10m_gqa2.candidate.json")
    mqa = _load("s3_10m_mqa1.candidate.json")

    assert mha.model.parameter_count() == 10_059_840
    assert gqa4.model.parameter_count() == 10_061_760
    assert gqa2.model.parameter_count() == 10_059_840
    assert mqa.model.parameter_count() == 10_061_760

    for stage in (gqa4, gqa2, mqa):
        relative_delta = (
            abs(stage.model.parameter_count() - mha.model.parameter_count())
            / mha.model.parameter_count()
        )
        assert relative_delta < 0.0002
        assert stage.model.n_heads % stage.model.n_kv_heads == 0

    mha_cache = _full_cache_bytes(mha, bytes_per_element=2)
    assert _full_cache_bytes(gqa4, bytes_per_element=2) * 2 == mha_cache
    assert _full_cache_bytes(gqa2, bytes_per_element=2) * 4 == mha_cache
    assert _full_cache_bytes(mqa, bytes_per_element=2) * 8 == mha_cache


def test_s4_gqa_candidate_is_near_matched_to_recorded_s4_mha_geometry() -> None:
    gqa = _load("s4_100m_gqa4.candidate.json")
    recorded_s4_mha_parameters = 100_384_512

    assert gqa.model.parameter_count() == 100_376_832
    relative_delta = (
        abs(gqa.model.parameter_count() - recorded_s4_mha_parameters)
        / recorded_s4_mha_parameters
    )
    assert relative_delta < 0.0001
    assert (gqa.model.n_heads, gqa.model.n_kv_heads) == (12, 4)

    mha_bf16_cache = 2 * 10 * 12 * 64 * 2048 * 2
    assert _full_cache_bytes(gqa, bytes_per_element=2) * 3 == mha_bf16_cache


def test_candidate_attention_parameter_savings_are_reallocated_to_mlp() -> None:
    s3_mha = _load("s3_10m.json").model.parameter_breakdown()
    s3_gqa4 = _load("s3_10m_gqa4.candidate.json").model.parameter_breakdown()
    s3_gqa2 = _load("s3_10m_gqa2.candidate.json").model.parameter_breakdown()

    assert s3_gqa4["attention_per_layer"] < s3_mha["attention_per_layer"]
    assert s3_gqa4["mlp_per_layer"] > s3_mha["mlp_per_layer"]
    assert s3_gqa2["attention_per_layer"] < s3_gqa4["attention_per_layer"]
    assert s3_gqa2["mlp_per_layer"] > s3_gqa4["mlp_per_layer"]


def test_native_gqa_routing_is_cuda_only_and_mha_never_requests_it() -> None:
    gqa_model = TwelveSixDecoder(_load("s2_1m_gqa2.candidate.json").model)
    mha_model = TwelveSixDecoder(_load("s2_1m.json").model)
    gqa_attention = gqa_model.blocks[0].attn
    mha_attention = mha_model.blocks[0].attn

    assert gqa_attention._native_gqa_enabled_for_device(torch.device("cuda")) is True
    assert gqa_attention._native_gqa_enabled_for_device(torch.device("cpu")) is False
    assert mha_attention._native_gqa_enabled_for_device(torch.device("cuda")) is False


def test_cpu_gqa_preserves_explicit_repeat_reference_path(monkeypatch) -> None:
    stage = _load("s2_1m_gqa2.candidate.json")
    model = TwelveSixDecoder(stage.model, stage.init).eval()
    attention = model.blocks[0].attn
    original_sdpa = F.scaled_dot_product_attention
    calls: list[tuple[int, int, int, dict[str, object]]] = []

    def spy_sdpa(q, k, v, **kwargs):
        calls.append((q.shape[1], k.shape[1], v.shape[1], dict(kwargs)))
        return original_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy_sdpa)
    x = torch.randn(1, 8, stage.model.d_model)
    with torch.no_grad():
        output = attention(x)

    assert output.shape == (1, 8, stage.model.d_model)
    assert len(calls) == 1
    q_heads, k_heads, v_heads, kwargs = calls[0]
    assert q_heads == stage.model.n_heads
    assert k_heads == stage.model.n_heads
    assert v_heads == stage.model.n_heads
    assert "enable_gqa" not in kwargs
