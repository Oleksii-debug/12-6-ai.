from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from twelve_six import (
    ModelSpec,
    TwelveSixDecoder,
    count_trainable_parameters,
    load_stage_config,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("s0_10k.json", 10_140),
        ("s1_100k.json", 107_856),
        ("s2_1m.json", 1_066_112),
        ("s3_10m.json", 10_059_840),
    ],
)
def test_stage_parameter_formula_matches_frozen_evidence(filename: str, expected: int) -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / filename)
    assert stage.expected_parameters == expected
    assert stage.model.parameter_count() == expected


def test_model_spec_dict_round_trip_is_stable() -> None:
    spec = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json").model
    assert ModelSpec.from_dict(spec.to_dict()) == spec


def test_s0_actual_trainable_parameter_count_is_exact() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    model = TwelveSixDecoder(stage.model)
    assert count_trainable_parameters(model) == 10_140
    assert model.lm_head.weight is model.token_embedding.weight


def test_forward_shape_and_causal_prefix_invariance() -> None:
    torch.manual_seed(7)
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    model = TwelveSixDecoder(stage.model).eval()

    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    right = torch.tensor([[1, 2, 3, 4, 200, 201, 202, 203]], dtype=torch.long)
    left_logits = model(left).logits
    right_logits = model(right).logits

    assert left_logits.shape == (1, 8, 256)
    torch.testing.assert_close(left_logits[:, :4], right_logits[:, :4], rtol=0, atol=0)


def test_random_initialization_is_seed_reproducible_without_external_weights() -> None:
    spec = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json").model
    torch.manual_seed(1234)
    first = TwelveSixDecoder(spec)
    torch.manual_seed(1234)
    second = TwelveSixDecoder(spec)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    assert torch.count_nonzero(first.token_embedding.weight).item() > 0


def test_state_dict_round_trip_preserves_logits() -> None:
    spec = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json").model
    torch.manual_seed(99)
    source = TwelveSixDecoder(spec).eval()
    input_ids = torch.tensor([[2, 4, 6, 8]], dtype=torch.long)
    expected = source(input_ids).logits

    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)

    torch.manual_seed(100)
    restored = TwelveSixDecoder(spec).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.lm_head.weight is restored.token_embedding.weight


def test_generation_contract_is_greedy_and_context_bounded() -> None:
    spec = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json").model
    torch.manual_seed(1)
    model = TwelveSixDecoder(spec).eval()
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.long)

    first = model.generate(prompt, max_new_tokens=5)
    second = model.generate(prompt, max_new_tokens=5)
    assert first.shape == (1, 8)
    assert torch.equal(first, second)
    assert int(first.max()) < spec.vocab_size


def test_gqa_shape_contract_and_parameter_formula() -> None:
    spec = ModelSpec(
        vocab_size=128,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=80,
        tie_embeddings=True,
    )
    model = TwelveSixDecoder(spec).eval()
    tokens = torch.randint(0, spec.vocab_size, (2, 7))
    assert model(tokens).logits.shape == (2, 7, 128)
    assert count_trainable_parameters(model) == spec.parameter_count()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 21, "n_heads": 2},
        {"d_model": 15, "n_heads": 3},
        {"d_model": 32, "n_heads": 4, "n_kv_heads": 3},
    ],
)
def test_invalid_attention_shapes_fail_closed(kwargs: dict[str, int]) -> None:
    base = {
        "vocab_size": 64,
        "max_seq_len": 16,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "d_ff": 64,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        ModelSpec(**base)
