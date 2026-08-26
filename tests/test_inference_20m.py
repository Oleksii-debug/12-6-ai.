from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from twelve_six.inference import GenerationConfig
from twelve_six.inference.twenty_m import (
    TWENTY_M_MAX_PARAMETERS,
    TWENTY_M_MIN_PARAMETERS,
    TwentyMInference,
    load_20m_model_spec,
    open_20m_inference,
    validate_20m_spec,
)
from twelve_six.model import ModelSpec


def _spec_20m(*, max_seq_len: int = 8) -> ModelSpec:
    # 20,012,928 trainable parameters with the canonical 256-byte vocabulary.
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=max_seq_len,
        d_model=384,
        n_layers=10,
        n_heads=6,
        n_kv_heads=3,
        head_dim=64,
        d_ff=1344,
        rope_rotary_dim=64,
    )


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=8,
        d_model=64,
        n_layers=2,
        n_heads=1,
        n_kv_heads=1,
        head_dim=64,
        d_ff=128,
        rope_rotary_dim=64,
    )


def test_20m_guard_accepts_mechanical_candidate_and_rejects_tiny() -> None:
    spec = _spec_20m()
    assert TWENTY_M_MIN_PARAMETERS <= spec.parameter_count() <= TWENTY_M_MAX_PARAMETERS
    assert validate_20m_spec(spec) is spec

    with pytest.raises(ValueError, match="outside the maintained ~20M inference band"):
        validate_20m_spec(_tiny_spec())


def test_model_spec_loader_accepts_raw_and_nested_json(tmp_path: Path) -> None:
    spec = _spec_20m()
    raw = tmp_path / "raw.json"
    nested = tmp_path / "nested.json"
    raw.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    nested.write_text(json.dumps({"model": spec.to_dict()}), encoding="utf-8")

    assert load_20m_model_spec(raw) == spec
    assert load_20m_model_spec(nested) == spec


def test_random_init_is_local_deterministic_and_raw_completion_only() -> None:
    spec = _spec_20m(max_seq_len=4)
    rng_before = torch.random.get_rng_state().clone()

    first = TwentyMInference.from_random_init(spec, seed=17)
    first_probe = next(first.backend.model.parameters()).detach().flatten()[:16].clone()
    result = first.generate("A", GenerationConfig(max_new_tokens=1))
    del first

    second = TwentyMInference.from_random_init(spec, seed=17)
    second_probe = next(second.backend.model.parameters()).detach().flatten()[:16].clone()

    assert torch.equal(first_probe, second_probe)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert result.prompt_token_ids == (65,)
    assert len(result.generated_token_ids) == 1
    assert result.stop_reason == "max_new_tokens"
    assert second.diagnostics()["learned_weights"] is False
    assert second.diagnostics()["source"] == "random_init"


def test_open_20m_inference_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        open_20m_inference()
    with pytest.raises(ValueError, match="exactly one"):
        open_20m_inference(checkpoint="x", model_spec=_spec_20m())
