from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token


@runtime_checkable
class BatchedInferenceBackend(InferenceBackend, Protocol):
    """Inference backend that can evaluate multiple independent prefixes at once."""

    def next_token_logits_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> Sequence[Sequence[float]]: ...


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
class BatchGenerationOutput:
    results: tuple[GenerationResult, ...]
    stats: BatchGenerationStats


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


def _validated_backend_context(backend: BatchedInferenceBackend) -> int:
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
    backend: BatchedInferenceBackend,
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
    backend: BatchedInferenceBackend,
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


def generate_batch(
    backend: BatchedInferenceBackend,
    requests: Sequence[BatchGenerationRequest],
    *,
    max_batch_size: int = 32,
    max_padding_tokens: int | None = None,
) -> BatchGenerationOutput:
    """Generate independent raw-Base completions while coalescing model forwards.

    Each request owns its own RNG and stopping state. Scheduling may change model-call
    grouping but must not change request-local generation semantics.
    """
    max_context_tokens = _validated_backend_context(backend)
    _validated_batch_policy(
        max_batch_size=max_batch_size,
        max_padding_tokens=max_padding_tokens,
    )

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

    model_batch_calls = 0
    sequences_evaluated = 0
    logical_input_positions = 0
    padding_positions = 0
    max_batch_observed = 0

    while True:
        active: list[_RequestState] = []
        for state in states:
            if state.done:
                continue
            if len(state.generated) >= state.config.max_new_tokens:
                state.stop_reason = "max_new_tokens"
                state.done = True
                continue
            if len(state.prefix) >= max_context_tokens:
                state.stop_reason = "context_limit"
                state.done = True
                continue
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

    return BatchGenerationOutput(
        results=tuple(result for result in ordered_results if result is not None),
        stats=BatchGenerationStats(
            model_batch_calls=model_batch_calls,
            sequences_evaluated=sequences_evaluated,
            logical_input_positions=logical_input_positions,
            right_padding_positions_scheduled=padding_positions,
            max_batch_observed=max_batch_observed,
        ),
    )
