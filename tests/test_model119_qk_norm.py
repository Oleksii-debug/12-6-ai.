from __future__ import annotations

import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.qk_norm_research import (
    ResearchModelSpec,
    build_research_decoder,
    qk_rms_normalize,
)


def _s2() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=2048,
        max_seq_len=512,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        head_dim=32,
        d_ff=352,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=32,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def _small() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=64,
        d_model=48,
        n_layers=3,
        n_heads=4,
        n_kv_heads=4,
        head_dim=12,
        d_ff=128,
        rope_rotary_dim=12,
    )


def test_disabled_research_spec_preserves_canonical_identity() -> None:
    base = _s2()
    assert base.identity_sha256() == "2889fdea4d17b5f592686c1a1a2fcd7dd16a9a029219351e95973ccfdef60566"
    control = ResearchModelSpec.from_base(base, research_qk_norm=False)
    candidate = ResearchModelSpec.from_base(base, research_qk_norm=True)
    assert control.to_dict() == base.to_dict()
    assert control.identity_sha256() == base.identity_sha256()
    assert candidate.identity_sha256() != base.identity_sha256()
    assert candidate.to_dict()["research_qk_norm"] is True
    assert candidate.parameter_count() == base.parameter_count()


def test_qk_norm_is_parameter_free_and_control_forward_is_unchanged() -> None:
    base = _small()
    control = ResearchModelSpec.from_base(base, research_qk_norm=False)
    candidate = ResearchModelSpec.from_base(base, research_qk_norm=True)
    torch.manual_seed(123)
    incumbent = TwelveSixDecoder(base)
    torch.manual_seed(123)
    research_control = build_research_decoder(control)
    torch.manual_seed(123)
    research_candidate = build_research_decoder(candidate)
    ids = torch.arange(32, dtype=torch.long).view(1, 32)
    assert torch.equal(incumbent(ids).logits, research_control(ids).logits)
    assert sum(p.numel() for p in incumbent.parameters()) == sum(
        p.numel() for p in research_candidate.parameters()
    )
    assert list(incumbent.state_dict()) == list(research_candidate.state_dict())
    assert not torch.equal(incumbent(ids).logits, research_candidate(ids).logits)


def test_parameterless_qk_rms_normalizes_each_head() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 4, 11, 16)
    y = qk_rms_normalize(x, 1e-6)
    rms = y.float().square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-5, rtol=2e-5)
