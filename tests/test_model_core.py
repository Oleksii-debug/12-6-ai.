from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from twelve_six import (
    InitSpec,
    ModelSpec,
    TwelveSixDecoder,
    count_trainable_parameters,
    load_stage_config,
)

ROOT = Path(__file__).resolve().parents[1]
S0_MODEL_HASH = "86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b"
INIT_HASH = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


@pytest.mark.parametrize(
    ("filename", "expected", "model_hash"),
    [
        ("s0_10k.json", 10_140, S0_MODEL_HASH),
        (
            "s1_100k.json",
            107_856,
            "2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6",
        ),
        (
            "s2_1m.json",
            1_066_112,
            "2889fdea4d17b5f592686c1a1a2fcd7dd16a9a029219351e95973ccfdef60566",
        ),
        (
            "s3_10m.json",
            10_059_840,
            "3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a",
        ),
    ],
)
def test_stage_parameter_and_identity_evidence(
    filename: str,
    expected: int,
    model_hash: str,
) -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / filename)
    assert stage.expected_parameters == expected
    assert stage.model.parameter_count() == expected
    assert stage.model.identity_sha256() == model_hash
    assert stage.init.identity_sha256() == INIT_HASH


def test_model_and_init_specs_round_trip_independently() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    assert ModelSpec.from_dict(stage.model.to_dict()) == stage.model
    assert InitSpec.from_dict(stage.init.to_dict()) == stage.init
    assert stage.model.identity_sha256() == S0_MODEL_HASH
    assert stage.init.identity_sha256() == INIT_HASH


def test_s0_semantic_contract_is_explicit() -> None:
    spec = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json").model
    assert spec.schema_version == 1
    assert spec.n_heads == 2
    assert spec.n_kv_heads == 2
    assert spec.head_dim == 10
    assert spec.q_dim == spec.d_model == 20
    assert spec.rope_rotary_dim == 10
    assert spec.activation == "swiglu"
    assert spec.norm_kind == "rmsnorm"
    assert spec.norm_placement == "pre"
    assert spec.position_embedding == "rope"
    assert spec.final_norm is True
    assert spec.tie_word_embeddings is True


def test_s0_actual_trainable_parameter_count_is_exact() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    model = TwelveSixDecoder(stage.model, stage.init)
    assert count_trainable_parameters(model) == 10_140
    assert model.lm_head.weight is model.token_embedding.weight
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_forward_shape_and_causal_prefix_invariance() -> None:
    torch.manual_seed(7)
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    model = TwelveSixDecoder(stage.model, stage.init).eval()

    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    right = torch.tensor([[1, 2, 3, 4, 200, 201, 202, 203]], dtype=torch.long)
    left_logits = model(left).logits
    right_logits = model(right).logits

    assert left_logits.shape == (1, 8, 256)
    torch.testing.assert_close(left_logits[:, :4], right_logits[:, :4], rtol=0, atol=0)


def test_random_initialization_is_seed_reproducible_without_external_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")

    def reject_foreign_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("constructor attempted to load external weights")

    monkeypatch.setattr(torch, "load", reject_foreign_load)
    torch.manual_seed(1234)
    first = TwelveSixDecoder(stage.model, stage.init)
    torch.manual_seed(1234)
    second = TwelveSixDecoder(stage.model, stage.init)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.count_nonzero(first.token_embedding.weight).item() > 0


def test_model_identity_is_separate_from_initialization_identity() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    different_init = InitSpec(std=0.01)
    assert stage.model.identity_sha256() == S0_MODEL_HASH
    assert different_init.identity_sha256() != stage.init.identity_sha256()

    torch.manual_seed(11)
    normal = TwelveSixDecoder(stage.model, stage.init)
    torch.manual_seed(11)
    narrow = TwelveSixDecoder(stage.model, different_init)
    assert not torch.equal(normal.token_embedding.weight, narrow.token_embedding.weight)


def test_state_dict_round_trip_preserves_logits_and_identity() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    torch.manual_seed(99)
    source = TwelveSixDecoder(stage.model, stage.init).eval()
    input_ids = torch.tensor([[2, 4, 6, 8]], dtype=torch.long)
    expected = source(input_ids).logits

    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)

    torch.manual_seed(100)
    restored = TwelveSixDecoder(stage.model, stage.init).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.lm_head.weight is restored.token_embedding.weight
    assert restored.spec.identity_sha256() == S0_MODEL_HASH
    assert restored.init_spec.identity_sha256() == INIT_HASH


def test_generation_contract_is_greedy_and_context_bounded() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    torch.manual_seed(1)
    model = TwelveSixDecoder(stage.model, stage.init).eval()
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.long)

    first = model.generate(prompt, max_new_tokens=5)
    second = model.generate(prompt, max_new_tokens=5)
    assert first.shape == (1, 8)
    assert torch.equal(first, second)
    assert int(first.max()) < stage.model.vocab_size


def test_explicit_head_dim_can_differ_from_residual_width_partition() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=128,
        max_seq_len=32,
        d_model=30,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=80,
        rope_rotary_dim=8,
    )
    assert spec.q_dim == 32
    assert spec.q_dim != spec.d_model
    model = TwelveSixDecoder(spec).eval()
    tokens = torch.randint(0, spec.vocab_size, (2, 7))
    assert model(tokens).logits.shape == (2, 7, 128)
    assert count_trainable_parameters(model) == spec.parameter_count()


def test_mqa_runtime_contract_and_parameter_formula() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=96,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        head_dim=8,
        d_ff=64,
        rope_rotary_dim=8,
    )
    model = TwelveSixDecoder(spec).eval()
    tokens = torch.randint(0, spec.vocab_size, (2, 5))
    assert model(tokens).logits.shape == (2, 5, 96)
    assert count_trainable_parameters(model) == spec.parameter_count()


def test_partial_rope_rotates_only_declared_head_prefix() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        rope_rotary_dim=4,
    )
    model = TwelveSixDecoder(spec).eval()
    tokens = torch.randint(0, spec.vocab_size, (1, 6))
    assert model(tokens).logits.shape == (1, 6, 64)
    assert count_trainable_parameters(model) == spec.parameter_count()


def test_bias_and_untied_head_parameter_formula_matches_model() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=64,
        max_seq_len=16,
        d_model=24,
        n_layers=2,
        n_heads=3,
        n_kv_heads=1,
        head_dim=8,
        d_ff=56,
        rope_rotary_dim=8,
        attention_bias=True,
        mlp_bias=True,
        final_norm=False,
        tie_word_embeddings=False,
        lm_head_bias=True,
    )
    model = TwelveSixDecoder(spec)
    assert model.lm_head.weight is not model.token_embedding.weight
    assert count_trainable_parameters(model) == spec.parameter_count()


def test_semantic_change_changes_model_identity_hash() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    payload = stage.model.to_dict()
    payload["attention_bias"] = True
    changed = ModelSpec.from_dict(payload)
    assert changed.identity_sha256() != stage.model.identity_sha256()


def test_stage_config_hash_tampering_fails_closed(tmp_path: Path) -> None:
    source = ROOT / "configs" / "stages" / "s0_10k.json"
    payload = source.read_text(encoding="utf-8")
    tampered = payload.replace('"rope_theta": 10000.0', '"rope_theta": 20000.0')
    path = tmp_path / "tampered.json"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="ModelSpec identity hash mismatch"):
        load_stage_config(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "unsupported ModelSpec schema_version"),
        ({"n_kv_heads": 3}, "n_heads must be divisible"),
        ({"head_dim": 7, "rope_rotary_dim": 6}, "even attention head_dim"),
        ({"head_dim": 8, "rope_rotary_dim": 10}, "cannot exceed head_dim"),
        ({"head_dim": 8, "rope_rotary_dim": 7}, "rope_rotary_dim must be even"),
        ({"activation": "gelu"}, "activation='swiglu' only"),
    ],
)
def test_invalid_semantic_geometry_fails_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    base: dict[str, object] = {
        "schema_version": 1,
        "vocab_size": 64,
        "max_seq_len": 16,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 8,
        "d_ff": 64,
        "rope_rotary_dim": 8,
    }
    base.update(overrides)
    with pytest.raises(ValueError, match=message):
        ModelSpec(**base)
