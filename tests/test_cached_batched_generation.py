from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from twelve_six.inference.batching import (
    BatchGenerationRequest,
    generate_batch_cached,
)
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.generation import generate
from twelve_six.integration.torch_batching import S0TorchBatchedInferenceBackend
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


def _tiny_model() -> TwelveSixDecoder:
    torch.manual_seed(20260826)
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=24,
        d_model=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        d_ff=32,
        rope_rotary_dim=8,
    )
    return TwelveSixDecoder(spec, InitSpec())


def test_real_torch_cached_batch_matches_independent_greedy_and_restores_mode() -> None:
    backend = S0TorchBatchedInferenceBackend(_tiny_model(), ByteTokenizer())
    backend.model.train()
    requests = (
        BatchGenerationRequest("a", GenerationConfig(max_new_tokens=5)),
        BatchGenerationRequest("b", GenerationConfig(max_new_tokens=2)),
        BatchGenerationRequest("abcd", GenerationConfig(max_new_tokens=4)),
        BatchGenerationRequest("xy", GenerationConfig(max_new_tokens=3)),
    )

    expected = tuple(generate(backend, request.prompt, request.config) for request in requests)
    assert backend.active_generation_sessions == 0
    assert backend.model.training is True

    actual = generate_batch_cached(backend, requests, max_batch_size=4)

    assert actual.results == expected
    assert backend.active_generation_sessions == 0
    assert backend.model.training is True
    assert actual.stats.prefill_batch_calls == 3
    assert actual.stats.max_batch_observed == 2
    assert actual.stats.retired_row_decode_positions > 0
    assert actual.stats.independent_cached_model_calls > actual.stats.model_batch_calls
    assert (
        actual.stats.independent_stateless_input_positions
        > actual.stats.scheduled_cached_input_positions
    )
    assert actual.stats.peak_cache_bytes > 0


def test_torch_batched_session_cache_bytes_and_ragged_prefill_fail_closed() -> None:
    backend = S0TorchBatchedInferenceBackend(_tiny_model(), ByteTokenizer())

    with pytest.raises(ValueError, match="exact-equal"):
        backend.begin_generation_batch(((1,), (2, 3)))

    with backend.begin_generation_batch(((1, 2), (3, 4))) as session:
        assert session.batch_size == 2
        assert session.sequence_length == 2
        assert session.cache_bytes == backend.estimate_cache_bytes(2, batch_size=2)
        session.append_batch((5, 6))
        assert session.sequence_length == 3
        assert session.cache_bytes == backend.estimate_cache_bytes(3, batch_size=2)

    assert backend.active_generation_sessions == 0


def test_torch_batched_session_setup_failure_releases_backend_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = S0TorchBatchedInferenceBackend(_tiny_model(), ByteTokenizer())
    backend.model.train()

    def fail_prefill(_: torch.Tensor) -> object:
        raise RuntimeError("injected prefill failure")

    monkeypatch.setattr(backend.model, "prefill_kv_cache", fail_prefill)

    with pytest.raises(RuntimeError, match="injected prefill failure"):
        backend.begin_generation_batch(((1, 2), (3, 4)))

    assert backend.active_generation_sessions == 0
    assert backend.model.training is True


class _FakeCachedSession:
    def __init__(
        self,
        backend: _FakeCachedBackend,
        input_ids: Sequence[Sequence[int]],
    ) -> None:
        lengths = {len(row) for row in input_ids}
        if not input_ids or len(lengths) != 1:
            raise ValueError("fake cache requires exact-equal rows")
        self.backend = backend
        self.rows = [list(row) for row in input_ids]
        self.closed = False
        backend.session_shapes.append(tuple(len(row) for row in input_ids))

    @property
    def batch_size(self) -> int:
        return len(self.rows)

    @property
    def sequence_length(self) -> int:
        return len(self.rows[0])

    @property
    def cache_bytes(self) -> int:
        return self.batch_size * self.sequence_length * 16

    def next_token_logits_batch(self) -> Sequence[Sequence[float]]:
        if self.closed:
            raise RuntimeError("session closed")
        return [self.backend.next_token_logits(row) for row in self.rows]

    def append_batch(self, token_ids: Sequence[int]) -> None:
        if self.closed:
            raise RuntimeError("session closed")
        if len(token_ids) != self.batch_size:
            raise ValueError("wrong append width")
        for row, token_id in zip(self.rows, token_ids, strict=True):
            row.append(token_id)

    def close(self) -> None:
        self.closed = True


class _FakeCachedBackend:
    eos_token_id: int | None = None
    max_context_tokens = 32
    cache_row_filler_token_id = 0

    def __init__(self) -> None:
        self.session_shapes: list[tuple[int, ...]] = []

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        shift = (sum(input_ids) + len(input_ids) * 17) % 11
        return [float(((token_id * 7 + shift) % 13) - 6) for token_id in range(128)]

    def next_token_logits_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> Sequence[Sequence[float]]:
        return [self.next_token_logits(row) for row in input_ids]

    def begin_generation_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> _FakeCachedSession:
        return _FakeCachedSession(self, input_ids)


class _StoppingCachedBackend(_FakeCachedBackend):
    eos_token_id = 67
    max_context_tokens = 12

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        first = input_ids[0]
        target = {ord("e"): 67, ord("t"): 66, ord("s"): 65}.get(first, 64)
        logits = [-100.0] * 128
        logits[target] = 100.0
        return logits


class _ContextLimitedCachedBackend(_FakeCachedBackend):
    max_context_tokens = 4


def test_cached_sampling_rng_is_request_local_after_neighbor_retires() -> None:
    sequential_backend = _FakeCachedBackend()
    batched_backend = _FakeCachedBackend()
    requests = (
        BatchGenerationRequest(
            "alpha",
            GenerationConfig(
                max_new_tokens=6,
                sample=True,
                seed=1,
                temperature=0.8,
                top_k=17,
                top_p=0.9,
            ),
        ),
        BatchGenerationRequest(
            "bravo",
            GenerationConfig(
                max_new_tokens=2,
                sample=True,
                seed=999,
                temperature=1.2,
                top_k=9,
                top_p=0.8,
            ),
        ),
        BatchGenerationRequest(
            "delta",
            GenerationConfig(
                max_new_tokens=5,
                sample=True,
                seed=1,
                temperature=0.8,
                top_k=17,
                top_p=0.9,
            ),
        ),
    )

    expected = tuple(
        generate(sequential_backend, request.prompt, request.config) for request in requests
    )
    actual = generate_batch_cached(batched_backend, requests, max_batch_size=3)

    assert actual.results == expected
    assert actual.stats.prefill_batch_calls == 1
    assert actual.stats.retired_row_decode_positions > 0
    assert actual.stats.scheduled_decode_positions > actual.stats.logical_decode_positions


def test_cached_stop_semantics_and_completed_rows_are_independent() -> None:
    sequential_backend = _StoppingCachedBackend()
    batched_backend = _StoppingCachedBackend()
    requests = (
        BatchGenerationRequest("e", GenerationConfig(max_new_tokens=5)),
        BatchGenerationRequest(
            "t",
            GenerationConfig(max_new_tokens=5, stop_token_ids=(66,)),
        ),
        BatchGenerationRequest(
            "s",
            GenerationConfig(max_new_tokens=5, stop_strings=("A",)),
        ),
        BatchGenerationRequest("x", GenerationConfig(max_new_tokens=4)),
    )

    expected = tuple(
        generate(sequential_backend, request.prompt, request.config) for request in requests
    )
    actual = generate_batch_cached(batched_backend, requests, max_batch_size=4)

    assert actual.results == expected
    assert tuple(result.stop_reason for result in actual.results) == (
        "eos",
        "stop_token",
        "stop_string",
        "max_new_tokens",
    )
    assert actual.results[2].text == ""
    assert actual.stats.retired_row_decode_positions > 0


def test_cached_context_limit_parity_when_entire_bucket_retires_together() -> None:
    sequential_backend = _ContextLimitedCachedBackend()
    batched_backend = _ContextLimitedCachedBackend()
    requests = (
        BatchGenerationRequest("abc", GenerationConfig(max_new_tokens=5)),
        BatchGenerationRequest("def", GenerationConfig(max_new_tokens=5)),
    )

    expected = tuple(
        generate(sequential_backend, request.prompt, request.config) for request in requests
    )
    actual = generate_batch_cached(batched_backend, requests, max_batch_size=2)

    assert actual.results == expected
    assert tuple(result.stop_reason for result in actual.results) == (
        "context_limit",
        "context_limit",
    )
    assert all(len(result.generated_token_ids) == 1 for result in actual.results)
    assert actual.stats.prefill_batch_calls == 1
    assert actual.stats.decode_batch_calls == 0
    assert actual.stats.logical_decode_positions == 0


def test_cached_scheduler_uses_exact_prompt_length_buckets_only() -> None:
    backend = _FakeCachedBackend()
    requests = (
        BatchGenerationRequest("a", GenerationConfig(max_new_tokens=1)),
        BatchGenerationRequest("b", GenerationConfig(max_new_tokens=1)),
        BatchGenerationRequest("cc", GenerationConfig(max_new_tokens=1)),
        BatchGenerationRequest("ddd", GenerationConfig(max_new_tokens=1)),
    )

    output = generate_batch_cached(backend, requests, max_batch_size=8)

    assert backend.session_shapes == [(1, 1), (2,), (3,)]
    assert output.stats.prefill_batch_calls == 3
    assert output.stats.decode_batch_calls == 0
    assert output.stats.retired_row_decode_positions == 0


@pytest.mark.parametrize("max_batch_size", [0, True])
def test_cached_invalid_batch_policy_fails_closed(max_batch_size: int) -> None:
    with pytest.raises(ValueError):
        generate_batch_cached(
            _FakeCachedBackend(),
            [BatchGenerationRequest("x")],
            max_batch_size=max_batch_size,
        )
