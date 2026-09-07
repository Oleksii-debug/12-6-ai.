from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.sampling import greedy_token
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import DecoderKVCache, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


class _StatelessBackend:
    """Deliberately hide begin_generation() to exercise the legacy full-prefix path."""

    eos_token_id: int | None = None

    def __init__(self, backend: S0TorchInferenceBackend) -> None:
        self._backend = backend
        self.max_context_tokens = backend.max_context_tokens

    def encode(self, text: str) -> list[int]:
        return self._backend.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._backend.decode(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return self._backend.next_token_logits(input_ids)


def _s0_model(seed: int = 20260825) -> TwelveSixDecoder:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    torch.manual_seed(seed)
    return TwelveSixDecoder(stage.model, stage.init)


def test_model_native_cache_matches_full_prefix_logits() -> None:
    model = _s0_model()
    model.eval()
    prompt = torch.tensor([[49, 50, 45, 54]], dtype=torch.long)

    full_prompt = model(prompt).logits
    cached_prompt, cache = model.prefill_kv_cache(prompt)
    torch.testing.assert_close(cached_prompt.logits, full_prompt, rtol=0.0, atol=0.0)
    assert cache.sequence_length == prompt.shape[1]
    assert cache.batch_size == 1
    assert cache.model_spec_sha256 == model.spec.identity_sha256()
    assert len(cache.layers) == model.spec.n_layers

    sequence = prompt
    for token_id in (65, 66, 67, 68):
        token = torch.tensor([[token_id]], dtype=torch.long)
        cached_output, cache = model.decode_one_with_kv_cache(token, cache)
        sequence = torch.cat((sequence, token), dim=1)
        full_output = model(sequence).logits[:, -1:, :]
        torch.testing.assert_close(cached_output.logits, full_output, rtol=1e-6, atol=1e-6)
        assert cache.sequence_length == sequence.shape[1]


def test_cache_preserves_unexpanded_gqa_geometry() -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    gqa_spec = replace(stage.model, n_kv_heads=1)
    torch.manual_seed(7)
    model = TwelveSixDecoder(gqa_spec, stage.init)
    model.eval()
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.long)

    _, cache = model.prefill_kv_cache(prompt)
    assert cache.layers
    for layer in cache.layers:
        assert layer.key.shape == (1, 1, 3, gqa_spec.head_dim)
        assert layer.value.shape == (1, 1, 3, gqa_spec.head_dim)

    next_token = torch.tensor([[13]], dtype=torch.long)
    cached, cache = model.decode_one_with_kv_cache(next_token, cache)
    full = model(torch.cat((prompt, next_token), dim=1)).logits[:, -1:, :]
    torch.testing.assert_close(cached.logits, full, rtol=1e-6, atol=1e-6)
    assert all(layer.key.shape[1] == 1 for layer in cache.layers)


def test_cache_fails_closed_on_training_identity_shape_and_context() -> None:
    model = _s0_model()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="model.eval"):
        model.prefill_kv_cache(prompt)

    model.eval()
    _, cache = model.prefill_kv_cache(prompt)
    wrong_identity = replace(cache, model_spec_sha256="0" * 64)
    with pytest.raises(ValueError, match="ModelSpec identity"):
        model.decode_one_with_kv_cache(torch.tensor([[3]]), wrong_identity)

    wrong_batch = replace(cache, batch_size=2)
    with pytest.raises(ValueError, match="batch size"):
        model.decode_one_with_kv_cache(torch.tensor([[3]]), wrong_batch)

    inconsistent = replace(cache, sequence_length=cache.sequence_length + 1)
    with pytest.raises(ValueError, match="layer sequence lengths"):
        model.decode_one_with_kv_cache(torch.tensor([[3]]), inconsistent)

    full_prompt = torch.ones((1, model.spec.max_seq_len), dtype=torch.long)
    _, full_cache = model.prefill_kv_cache(full_prompt)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.decode_one_with_kv_cache(torch.tensor([[1]]), full_cache)


def test_d07_generation_uses_cache_without_changing_greedy_or_sampling() -> None:
    model = _s0_model()
    tokenizer = ByteTokenizer()
    cached_backend = S0TorchInferenceBackend(model, tokenizer)
    stateless_backend = _StatelessBackend(cached_backend)

    greedy = GenerationConfig(max_new_tokens=8, sample=False, seed=17)
    assert generate(cached_backend, "12-6", greedy) == generate(stateless_backend, "12-6", greedy)

    sampled = GenerationConfig(
        max_new_tokens=8,
        sample=True,
        temperature=0.8,
        top_k=32,
        top_p=0.9,
        seed=17,
    )
    cached_sample = generate(cached_backend, "Base", sampled)
    assert cached_sample == generate(cached_backend, "Base", sampled)
    assert cached_sample == generate(stateless_backend, "Base", sampled)


def test_incremental_session_reduces_prefix_token_work_and_restores_mode() -> None:
    model = _s0_model()
    tokenizer = ByteTokenizer()
    backend = S0TorchInferenceBackend(model, tokenizer)
    prompt_ids = tokenizer.encode("cache")
    assert model.training is True

    session = backend.begin_generation(prompt_ids)
    try:
        assert model.training is False
        assert session.tokens_processed == len(prompt_ids)
        assert session.sequence_length == len(prompt_ids)
        for _ in range(3):
            token_id = greedy_token(session.next_token_logits())
            session.append(token_id)
        assert session.tokens_processed == len(prompt_ids) + 3
        assert session.sequence_length == len(prompt_ids) + 3

        stateless_prefix_token_work = sum(len(prompt_ids) + step for step in range(4))
        assert session.tokens_processed < stateless_prefix_token_work
    finally:
        session.close()

    assert model.training is True
    with pytest.raises(RuntimeError, match="closed"):
        session.next_token_logits()


def test_kv_cache_is_ephemeral_and_does_not_change_checkpoint_identity() -> None:
    model = _s0_model()
    model.eval()
    before_keys = tuple(model.state_dict())
    before_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_identity = model.spec.identity_sha256()

    _, cache = model.prefill_kv_cache(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert isinstance(cache, DecoderKVCache)

    assert tuple(model.state_dict()) == before_keys
    assert sum(parameter.numel() for parameter in model.parameters()) == before_parameter_count
    assert model.spec.identity_sha256() == model_identity
