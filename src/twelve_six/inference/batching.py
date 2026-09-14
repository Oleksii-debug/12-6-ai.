from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token


@runtime_checkable
class BatchedInferenceBackend(InferenceBackend, Protocol):
    """Inference backend that can evaluate multiple independent prefixes at once."""

    def next_token_logits_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> Sequence[Sequence[float]]: ...


class CachedBatchGenerationSession(Protocol):
    """Fixed-row, equal-length KV-cache batch owned by one backend session."""

    @property
    def batch_size(self) -> int: ...

    @property
    def sequence_length(self) -> int: ...

    @property
    def cache_bytes(self) -> int: ...

    def next_token_logits_batch(self) -> Sequence[Sequence[float]]: ...

    def append_batch(self, token_ids: Sequence[int]) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class CachedBatchedInferenceBackend(BatchedInferenceBackend, Protocol):
    """Batched backend that can open one fixed-row model-native KV-cache session."""

    cache_row_filler_token_id: int

    def begin_generation_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> CachedBatchGenerationSession: ...


@dataclass(frozen=True, slots=True)
class BatchGenerationRequest:
    prompt: str
    config: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass(frozen=True, slots=True)
class BatchGenerationStats:
    model_batch_calls: int
    sequences_evaluated: int
    logical_input_positions: int
    right_padding_positions_scheduled: int
    max_batch_observed: int


@dataclass(frozen=True, slots=True)
class CachedBatchGenerationStats:
    model_batch_calls: int
    prefill_batch_calls: int
    decode_batch_calls: int
    sequences_prefilled: int
    logical_prefill_positions: int
    logical_decode_positions: int
    scheduled_decode_positions: int
    retired_row_decode_positions: int
    logical_cached_input_positions: int
    scheduled_cached_input_positions: int
    independent_cached_model_calls: int
    independent_stateless_input_positions: int
    max_batch_observed: int
    peak_cache_bytes: int


@dataclass(frozen=True, slots=True)
class BatchGenerationOutput:
    results: tuple[GenerationResult, ...]
    stats: BatchGenerationStats


@dataclass(frozen=True, slots=True)
class CachedBatchGenerationOutput:
    results: tuple[GenerationResult, ...]
    stats: CachedBatchGenerationStats


@dataclass(slots=True)
class _RequestState:
    index: int
    prompt_token_ids: tuple[int, ...]
    config: GenerationConfig
    generated: list[int]
    rng: random.Random
    stop_reason: StopReason = "max_new_tokens"
    matched_stop: str | None = None
    done: bool = False

    @property
    def prefix(self) -> tuple[int, ...]:
        return (*self.prompt_token_ids, *self.generated)


def _validated_backend_context(backend: InferenceBackend) -> int:
    value = backend.max_context_tokens
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("backend max_context_tokens must be a positive integer")
    eos_token_id = backend.eos_token_id
    if eos_token_id is not None and (
        not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool) or eos_token_id < 0
    ):
        raise ValueError("backend eos_token_id must be a non-negative integer or None")
    return value


def _validated_prompt_token_ids(
    backend: InferenceBackend,
    prompt: str,
    *,
    max_context_tokens: int,
) -> tuple[int, ...]:
    token_ids = backend.encode(prompt)
    if not token_ids:
        raise ValueError("prompt encoded to zero tokens; backend must provide a non-empty context")
    if len(token_ids) > max_context_tokens:
        raise ValueError(
            f"prompt has {len(token_ids)} tokens but backend max_context_tokens="
            f"{max_context_tokens}"
        )
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("backend encode() must return non-negative integer token IDs")
    return tuple(token_ids)


def _validated_batch_policy(
    *,
    max_batch_size: int,
    max_padding_tokens: int | None,
) -> None:
    if (
        not isinstance(max_batch_size, int)
        or isinstance(max_batch_size, bool)
        or max_batch_size <= 0
    ):
        raise ValueError("max_batch_size must be a positive integer")
    if max_padding_tokens is not None and (
        not isinstance(max_padding_tokens, int)
        or isinstance(max_padding_tokens, bool)
        or max_padding_tokens < 0
    ):
        raise ValueError("max_padding_tokens must be a non-negative integer or None")


def _request_states(
    backend: InferenceBackend,
    requests: Sequence[BatchGenerationRequest],
    *,
    max_context_tokens: int,
) -> list[_RequestState]:
    states: list[_RequestState] = []
    for index, request in enumerate(requests):
        if not isinstance(request, BatchGenerationRequest):
            raise TypeError("requests must contain BatchGenerationRequest values")
        prompt_token_ids = _validated_prompt_token_ids(
            backend,
            request.prompt,
            max_context_tokens=max_context_tokens,
        )
        states.append(
            _RequestState(
                index=index,
                prompt_token_ids=prompt_token_ids,
                config=request.config,
                generated=[],
                rng=random.Random(request.config.seed),
            )
        )
    return states


def _microbatches(
    states: Sequence[_RequestState],
    *,
    max_batch_size: int,
    max_padding_tokens: int | None,
) -> Iterator[list[_RequestState]]:
    ordered = sorted(states, key=lambda state: (len(state.prefix), state.index))
    current: list[_RequestState] = []
    current_lengths: list[int] = []

    for state in ordered:
        length = len(state.prefix)
        trial_lengths = [*current_lengths, length]
        trial_max = max(trial_lengths)
        padding = trial_max * len(trial_lengths) - sum(trial_lengths)
        exceeds_size = len(trial_lengths) > max_batch_size
        exceeds_padding = max_padding_tokens is not None and padding > max_padding_tokens
        if current and (exceeds_size or exceeds_padding):
            yield current
            current = [state]
            current_lengths = [length]
        else:
            current.append(state)
            current_lengths.append(length)

    if current:
        yield current


def _exact_prompt_length_microbatches(
    states: Sequence[_RequestState],
    *,
    max_batch_size: int,
) -> Iterator[list[_RequestState]]:
    """Yield fixed cache batches; heterogeneous lengths never share one cache."""
    ordered = sorted(states, key=lambda state: (len(state.prompt_token_ids), state.index))
    current: list[_RequestState] = []
    current_length: int | None = None

    for state in ordered:
        length = len(state.prompt_token_ids)
        if current and (length != current_length or len(current) >= max_batch_size):
            yield current
            current = []
        if not current:
            current_length = length
        current.append(state)

    if current:
        yield current


def _select_token(
    logits: Sequence[float],
    state: _RequestState,
) -> int:
    config = state.config
    if config.sample:
        return sample_token(
            logits,
            rng=state.rng,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
    return greedy_token(logits)


def _apply_stop_conditions(
    backend: InferenceBackend,
    state: _RequestState,
    token_id: int,
) -> None:
    config = state.config
    if backend.eos_token_id is not None and token_id == backend.eos_token_id:
        state.stop_reason = "eos"
        state.done = True
        return
    if token_id in config.stop_token_ids:
        state.stop_reason = "stop_token"
        state.done = True
        return
    if config.stop_strings:
        current_text = backend.decode(state.generated)
        matched_stop = next(
            (stop for stop in config.stop_strings if current_text.endswith(stop)),
            None,
        )
        if matched_stop is not None:
            state.stop_reason = "stop_string"
            state.matched_stop = matched_stop
            state.done = True


def _mark_non_model_completion(state: _RequestState, *, max_context_tokens: int) -> None:
    if state.done:
        return
    if len(state.generated) >= state.config.max_new_tokens:
        state.stop_reason = "max_new_tokens"
        state.done = True
        return
    if len(state.prefix) >= max_context_tokens:
        state.stop_reason = "context_limit"
        state.done = True


def _ordered_results(
    backend: InferenceBackend,
    states: Sequence[_RequestState],
) -> tuple[GenerationResult, ...]:
    ordered_results: list[GenerationResult | None] = [None] * len(states)
    for state in states:
        text = backend.decode(state.generated)
        if state.matched_stop is not None and state.config.strip_stop_strings:
            text = text[: -len(state.matched_stop)]
        ordered_results[state.index] = GenerationResult(
            prompt_token_ids=state.prompt_token_ids,
            generated_token_ids=tuple(state.generated),
            text=text,
            stop_reason=state.stop_reason,
        )

    if any(result is None for result in ordered_results):
        raise RuntimeError("internal batch generation result accounting failed")
    return tuple(result for result in ordered_results if result is not None)


def _validated_cache_filler_token_id(backend: CachedBatchedInferenceBackend) -> int:
    token_id = backend.cache_row_filler_token_id
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError("cache_row_filler_token_id must be a non-negative integer")
    return token_id


def _begin_cached_batch_session(
    backend: CachedBatchedInferenceBackend,
    rows: Sequence[Sequence[int]],
) -> CachedBatchGenerationSession:
    if not rows:
        raise ValueError("cached generation batch must be non-empty")
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError("KV-cache batching requires exact-equal prompt lengths per batch")
    factory = getattr(backend, "begin_generation_batch", None)
    if not callable(factory):
        raise TypeError("cached batching backend must expose begin_generation_batch()")
    session = factory(rows)
    for method_name in ("next_token_logits_batch", "append_batch", "close"):
        if not callable(getattr(session, method_name, None)):
            raise TypeError(f"cached batch session must expose {method_name}()")
    typed = cast(CachedBatchGenerationSession, session)
    if typed.batch_size != len(rows):
        typed.close()
        raise ValueError("cached batch session batch_size does not match request batch")
    expected_length = len(rows[0])
    if typed.sequence_length != expected_length:
        typed.close()
        raise ValueError("cached batch session sequence_length does not match prompt length")
    return typed


def _session_cache_bytes(session: CachedBatchGenerationSession) -> int:
    value = session.cache_bytes
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("cached batch session cache_bytes must be a non-negative integer")
    return value


def _cached_batch_logits(
    session: CachedBatchGenerationSession,
    *,
    expected_batch_size: int,
) -> Sequence[Sequence[float]]:
    logits = session.next_token_logits_batch()
    if len(logits) != expected_batch_size:
        raise ValueError("cached batch session returned a different logits batch size")
    return logits


def generate_batch(
    backend: BatchedInferenceBackend,
    requests: Sequence[BatchGenerationRequest],
    *,
    max_batch_size: int = 32,
    max_padding_tokens: int | None = None,
) -> BatchGenerationOutput:
    """Generate independent raw-Base completions while coalescing full-prefix forwards.

    Each request owns its own RNG and stopping state. Scheduling may change model-call
    grouping but must not change request-local generation semantics.
    """
    max_context_tokens = _validated_backend_context(backend)
    _validated_batch_policy(
        max_batch_size=max_batch_size,
        max_padding_tokens=max_padding_tokens,
    )
    states = _request_states(
        backend,
        requests,
        max_context_tokens=max_context_tokens,
    )

    model_batch_calls = 0
    sequences_evaluated = 0
    logical_input_positions = 0
    padding_positions = 0
    max_batch_observed = 0

    while True:
        active: list[_RequestState] = []
        for state in states:
            _mark_non_model_completion(state, max_context_tokens=max_context_tokens)
            if not state.done:
                active.append(state)

        if not active:
            break

        for microbatch in _microbatches(
            active,
            max_batch_size=max_batch_size,
            max_padding_tokens=max_padding_tokens,
        ):
            prefixes = [state.prefix for state in microbatch]
            batch_logits = backend.next_token_logits_batch(prefixes)
            if len(batch_logits) != len(microbatch):
                raise ValueError(
                    "backend next_token_logits_batch() returned a different batch size"
                )

            lengths = [len(prefix) for prefix in prefixes]
            max_length = max(lengths)
            model_batch_calls += 1
            sequences_evaluated += len(prefixes)
            logical_input_positions += sum(lengths)
            padding_positions += max_length * len(lengths) - sum(lengths)
            max_batch_observed = max(max_batch_observed, len(prefixes))

            for state, logits in zip(microbatch, batch_logits, strict=True):
                token_id = _select_token(logits, state)
                state.generated.append(token_id)
                _apply_stop_conditions(backend, state, token_id)

    return BatchGenerationOutput(
        results=_ordered_results(backend, states),
        stats=BatchGenerationStats(
            model_batch_calls=model_batch_calls,
            sequences_evaluated=sequences_evaluated,
            logical_input_positions=logical_input_positions,
            right_padding_positions_scheduled=padding_positions,
            max_batch_observed=max_batch_observed,
        ),
    )


def generate_batch_cached(
    backend: CachedBatchedInferenceBackend,
    requests: Sequence[BatchGenerationRequest],
    *,
    max_batch_size: int = 32,
) -> CachedBatchGenerationOutput:
    """Generate through fixed-row model-native KV-cache batches.

    Only exact-equal prompt lengths share a cache. Completed rows are never sampled again
    and never consume request RNG. They remain physically resident until their bucket
    drains, receiving an internal filler token only when another row requires a decode.
    No ragged cache compaction, semantic padding, or paged attention is attempted.
    """
    max_context_tokens = _validated_backend_context(backend)
    _validated_batch_policy(max_batch_size=max_batch_size, max_padding_tokens=0)
    filler_token_id = _validated_cache_filler_token_id(backend)
    states = _request_states(
        backend,
        requests,
        max_context_tokens=max_context_tokens,
    )

    model_batch_calls = 0
    prefill_batch_calls = 0
    decode_batch_calls = 0
    sequences_prefilled = 0
    logical_prefill_positions = 0
    logical_decode_positions = 0
    scheduled_decode_positions = 0
    retired_row_decode_positions = 0
    max_batch_observed = 0
    peak_cache_bytes = 0

    for microbatch in _exact_prompt_length_microbatches(
        states,
        max_batch_size=max_batch_size,
    ):
        rows = [state.prompt_token_ids for state in microbatch]
        session = _begin_cached_batch_session(backend, rows)
        batch_size = len(microbatch)
        model_batch_calls += 1
        prefill_batch_calls += 1
        sequences_prefilled += batch_size
        logical_prefill_positions += sum(len(row) for row in rows)
        max_batch_observed = max(max_batch_observed, batch_size)
        peak_cache_bytes = max(peak_cache_bytes, _session_cache_bytes(session))

        try:
            while True:
                active_rows: list[int] = []
                for row_index, state in enumerate(microbatch):
                    _mark_non_model_completion(
                        state,
                        max_context_tokens=max_context_tokens,
                    )
                    if not state.done:
                        active_rows.append(row_index)

                if not active_rows:
                    break

                batch_logits = _cached_batch_logits(
                    session,
                    expected_batch_size=batch_size,
                )
                selected_tokens: dict[int, int] = {}
                for row_index in active_rows:
                    state = microbatch[row_index]
                    token_id = _select_token(batch_logits[row_index], state)
                    selected_tokens[row_index] = token_id
                    state.generated.append(token_id)
                    _apply_stop_conditions(backend, state, token_id)

                future_rows: set[int] = set()
                for row_index in active_rows:
                    state = microbatch[row_index]
                    _mark_non_model_completion(
                        state,
                        max_context_tokens=max_context_tokens,
                    )
                    if not state.done:
                        future_rows.add(row_index)

                if not future_rows:
                    break

                append_ids: list[int] = []
                for row_index in range(batch_size):
                    if row_index in future_rows:
                        append_ids.append(selected_tokens[row_index])
                    else:
                        append_ids.append(filler_token_id)

                session.append_batch(append_ids)
                model_batch_calls += 1
                decode_batch_calls += 1
                logical_decode_positions += len(future_rows)
                scheduled_decode_positions += batch_size
                retired_row_decode_positions += batch_size - len(future_rows)
                peak_cache_bytes = max(peak_cache_bytes, _session_cache_bytes(session))
        finally:
            session.close()

    logical_cached_input_positions = logical_prefill_positions + logical_decode_positions
    scheduled_cached_input_positions = logical_prefill_positions + scheduled_decode_positions
    independent_cached_model_calls = len(states) + logical_decode_positions
    independent_stateless_input_positions = sum(
        sum(
            len(state.prompt_token_ids) + step_index
            for step_index in range(len(state.generated))
        )
        for state in states
    )

    return CachedBatchGenerationOutput(
        results=_ordered_results(backend, states),
        stats=CachedBatchGenerationStats(
            model_batch_calls=model_batch_calls,
            prefill_batch_calls=prefill_batch_calls,
            decode_batch_calls=decode_batch_calls,
            sequences_prefilled=sequences_prefilled,
            logical_prefill_positions=logical_prefill_positions,
            logical_decode_positions=logical_decode_positions,
            scheduled_decode_positions=scheduled_decode_positions,
            retired_row_decode_positions=retired_row_decode_positions,
            logical_cached_input_positions=logical_cached_input_positions,
            scheduled_cached_input_positions=scheduled_cached_input_positions,
            independent_cached_model_calls=independent_cached_model_calls,
            independent_stateless_input_positions=independent_stateless_input_positions,
            max_batch_observed=max_batch_observed,
            peak_cache_bytes=peak_cache_bytes,
        ),
    )
