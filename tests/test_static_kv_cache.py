from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.batching import BatchGenerationRequest, generate_batch_cached
from twelve_six.inference.sampling import greedy_token
from twelve_six.inference.static_kv import (
    allocate_static_kv_cache,
    decode_one_with_static_kv_cache,
    prefill_static_kv_cache,
)
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.integration.torch_batching import S0TorchBatchedInferenceBackend
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


def _tiny_gqa_model(*, max_seq_len: int = 16) -> TwelveSixDecoder:
    torch.manual_seed(226)
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=max_seq_len,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        rope_rotary_dim=8,
    )
    return TwelveSixDecoder(spec, InitSpec())


class _StatelessBackend:
    eos_token_id: int | None

    def __init__(self, backend: S0TorchInferenceBackend) -> None:
        self.backend = backend
        self.max_context_tokens = backend.max_context_tokens
        self.eos_token_id = backend.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.backend.decode(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return self.backend.next_token_logits(input_ids)


class _DynamicBackend(_StatelessBackend):
    def begin_generation(self, input_ids: Sequence[int]) -> object:
        return self.backend.begin_dynamic_generation(input_ids)


class _DynamicBatchedBackend:
    eos_token_id: int | None

    def __init__(self, backend: S0TorchBatchedInferenceBackend) -> None:
        self.backend = backend
        self.max_context_tokens = backend.max_context_tokens
        self.eos_token_id = backend.eos_token_id
        self.cache_row_filler_token_id = backend.cache_row_filler_token_id

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.backend.decode(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return self.backend.next_token_logits(input_ids)

    def next_token_logits_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> Sequence[Sequence[float]]:
        return self.backend.next_token_logits_batch(input_ids)

    def begin_generation_batch(self, input_ids: Sequence[Sequence[int]]) -> object:
        return self.backend.begin_dynamic_generation_batch(input_ids)


def test_static_prefill_and_decode_match_stateless_and_dynamic_gqa() -> None:
    model = _tiny_gqa_model()
    model.eval()
    prompt = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    next_token = torch.tensor([[14]], dtype=torch.long)

    static_cache = allocate_static_kv_cache(model, batch_size=1)
    storage = static_cache.storage_signature
    allocated = static_cache.allocated_bytes
    static_prompt = prefill_static_kv_cache(model, prompt, static_cache)
    dynamic_prompt, dynamic_cache = model.prefill_kv_cache(prompt)
    stateless_prompt = model(prompt)

    torch.testing.assert_close(static_prompt.logits, stateless_prompt.logits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(static_prompt.logits, dynamic_prompt.logits, rtol=1e-6, atol=1e-6)
    assert static_cache.valid_lengths == [4]
    assert static_cache.storage_signature == storage
    assert static_cache.allocated_bytes == allocated
    for layer in static_cache.layers:
        assert layer.key.shape == (1, model.spec.n_kv_heads, model.spec.max_seq_len, 8)
        assert layer.value.shape == layer.key.shape
        assert layer.key.shape[1] != model.spec.n_heads

    static_next = decode_one_with_static_kv_cache(model, next_token, static_cache)
    dynamic_next, dynamic_cache = model.decode_one_with_kv_cache(next_token, dynamic_cache)
    stateless_next = model(torch.cat((prompt, next_token), dim=1)).logits[:, -1:, :]

    torch.testing.assert_close(static_next.logits, stateless_next, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(static_next.logits, dynamic_next.logits, rtol=1e-6, atol=1e-6)
    assert static_cache.valid_lengths == [5]
    assert dynamic_cache.sequence_length == 5
    assert static_cache.storage_signature == storage
    assert static_cache.allocated_bytes == allocated


def test_static_decode_path_does_not_use_torch_cat_for_kv_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_gqa_model()
    model.eval()
    cache = allocate_static_kv_cache(model, batch_size=1)
    prefill_static_kv_cache(model, torch.tensor([[1, 2, 3]], dtype=torch.long), cache)
    storage = cache.storage_signature

    def reject_cat(*_: object, **__: object) -> object:
        raise AssertionError("torch.cat is forbidden on accepted static decode path")

    monkeypatch.setattr(torch, "cat", reject_cat)
    output = decode_one_with_static_kv_cache(
        model,
        torch.tensor([[4]], dtype=torch.long),
        cache,
    )

    assert output.logits.shape == (1, 1, model.spec.vocab_size)
    assert cache.valid_lengths == [4]
    assert cache.storage_signature == storage


def test_static_cache_reset_and_reuse_keep_backing_storage() -> None:
    model = _tiny_gqa_model()
    model.eval()
    cache = allocate_static_kv_cache(model, batch_size=2)
    storage = cache.storage_signature
    allocated = cache.allocated_bytes

    first = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    prefill_static_kv_cache(model, first, cache)
    assert cache.valid_lengths == [3, 3]
    assert cache.logical_bytes > 0

    cache.reset((0,))
    assert cache.valid_lengths == [0, 3]
    with pytest.raises(ValueError, match="heterogeneous"):
        _ = cache.sequence_length

    cache.reset()
    assert cache.valid_lengths == [0, 0]
    second = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    prefill_static_kv_cache(model, second, cache)
    assert cache.valid_lengths == [2, 2]
    assert cache.storage_signature == storage
    assert cache.allocated_bytes == allocated


def test_static_cache_context_boundary_fails_before_valid_length_mutation() -> None:
    model = _tiny_gqa_model(max_seq_len=6)
    model.eval()
    cache = allocate_static_kv_cache(model, batch_size=1, capacity=4)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    prefill_static_kv_cache(model, prompt, cache)
    storage = cache.storage_signature
    before_lengths = list(cache.valid_lengths)

    with pytest.raises(ValueError, match="fixed capacity"):
        decode_one_with_static_kv_cache(model, torch.tensor([[5]], dtype=torch.long), cache)

    assert cache.valid_lengths == before_lengths
    assert cache.storage_signature == storage


def test_static_generation_matches_dynamic_and_stateless_greedy_and_sampling() -> None:
    backend = S0TorchInferenceBackend(_tiny_gqa_model(), ByteTokenizer())
    stateless = _StatelessBackend(backend)
    dynamic = _DynamicBackend(backend)

    greedy = GenerationConfig(max_new_tokens=6, sample=False, seed=17)
    accepted_greedy = generate(backend, "cache", greedy)
    assert accepted_greedy == generate(dynamic, "cache", greedy)
    assert accepted_greedy == generate(stateless, "cache", greedy)

    sampled = GenerationConfig(
        max_new_tokens=6,
        sample=True,
        temperature=0.85,
        top_k=32,
        top_p=0.92,
        seed=226,
    )
    accepted_sample = generate(backend, "Base", sampled)
    assert accepted_sample == generate(dynamic, "Base", sampled)
    assert accepted_sample == generate(stateless, "Base", sampled)


def test_static_generation_preserves_stop_token_and_eos_semantics() -> None:
    backend = S0TorchInferenceBackend(_tiny_gqa_model(), ByteTokenizer())
    prompt_ids = backend.encode("stop")
    first_token = greedy_token(backend.next_token_logits(prompt_ids))

    stop_config = GenerationConfig(max_new_tokens=5, stop_token_ids=(first_token,))
    stop_result = generate(backend, "stop", stop_config)
    assert stop_result.generated_token_ids == (first_token,)
    assert stop_result.stop_reason == "stop_token"

    backend.eos_token_id = first_token
    eos_result = generate(backend, "stop", GenerationConfig(max_new_tokens=5))
    assert eos_result.generated_token_ids == (first_token,)
    assert eos_result.stop_reason == "eos"


def test_static_batched_generation_matches_dynamic_cache_and_keeps_storage_fixed() -> None:
    backend = S0TorchBatchedInferenceBackend(_tiny_gqa_model(), ByteTokenizer())
    rows = ((1, 2, 3), (4, 5, 6))

    with backend.begin_generation_batch(rows) as static_session:
        with backend.begin_dynamic_generation_batch(rows) as dynamic_session:
            torch.testing.assert_close(
                torch.tensor(static_session.next_token_logits_batch()),
                torch.tensor(dynamic_session.next_token_logits_batch()),
                rtol=1e-6,
                atol=1e-6,
            )
            storage = static_session.cache_storage_signature
            allocated = static_session.cache_bytes
            dynamic_before = dynamic_session.cache_bytes
            static_session.append_batch((7, 8))
            dynamic_session.append_batch((7, 8))
            torch.testing.assert_close(
                torch.tensor(static_session.next_token_logits_batch()),
                torch.tensor(dynamic_session.next_token_logits_batch()),
                rtol=1e-6,
                atol=1e-6,
            )
            assert static_session.cache_storage_signature == storage
            assert static_session.cache_bytes == allocated
            assert dynamic_session.cache_bytes > dynamic_before

    requests = (
        BatchGenerationRequest("aa", GenerationConfig(max_new_tokens=5)),
        BatchGenerationRequest("bb", GenerationConfig(max_new_tokens=2)),
        BatchGenerationRequest(
            "cc",
            GenerationConfig(
                max_new_tokens=4,
                sample=True,
                temperature=0.9,
                top_k=16,
                top_p=0.95,
                seed=41,
            ),
        ),
    )
    accepted = generate_batch_cached(backend, requests, max_batch_size=3)
    dynamic = generate_batch_cached(_DynamicBatchedBackend(backend), requests, max_batch_size=3)
    assert accepted.results == dynamic.results
    assert accepted.stats.model_batch_calls == dynamic.stats.model_batch_calls
    assert accepted.stats.logical_cached_input_positions == dynamic.stats.logical_cached_input_positions


def test_static_session_reuses_arena_and_enforces_model_max_context() -> None:
    backend = S0TorchInferenceBackend(_tiny_gqa_model(max_seq_len=8), ByteTokenizer())
    session = backend.begin_generation((1, 2, 3))
    try:
        storage = session.cache_storage_signature
        allocated = session.cache_bytes
        session.append(4)
        session.reset((5, 6))
        assert session.sequence_length == 2
        assert session.cache_storage_signature == storage
        assert session.cache_bytes == allocated

        while session.sequence_length < backend.max_context_tokens:
            session.append(7)
        with pytest.raises(ValueError, match="context limit"):
            session.append(7)
    finally:
        session.close()

    assert backend.active_generation_sessions == 0
