from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal, Protocol, cast

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token

CacheMode = Literal["auto", "stateless", "static"]


class _GenerationSession(Protocol):
    def next_token_logits(self) -> Sequence[float]: ...

    def append(self, token_id: int) -> None: ...

    def close(self) -> None: ...


def _begin_incremental_session(
    backend: InferenceBackend,
    prompt_token_ids: Sequence[int],
    *,
    required: bool,
) -> _GenerationSession | None:
    factory = getattr(backend, "begin_generation", None)
    if factory is None:
        if required:
            raise ValueError("backend does not support static-KV generation")
        return None
    if not callable(factory):
        raise TypeError("backend begin_generation must be callable")
    session = factory(prompt_token_ids)
    for method_name in ("next_token_logits", "append", "close"):
        if not callable(getattr(session, method_name, None)):
            try:
                session.close()
            finally:
                raise TypeError(
                    f"incremental generation session must expose {method_name}()"
                )
    return cast(_GenerationSession, session)


def _validated_token_ids(token_ids: Sequence[int], *, field: str) -> tuple[int, ...]:
    values: list[int] = []
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(f"{field} must contain integers")
        if token_id < 0:
            raise ValueError(f"{field} must contain non-negative token IDs")
        values.append(token_id)
    return tuple(values)


def _validate_backend_contract(backend: InferenceBackend) -> int | None:
    max_context_tokens = backend.max_context_tokens
    if not isinstance(max_context_tokens, int) or isinstance(max_context_tokens, bool):
        raise TypeError("backend max_context_tokens must be an integer")
    if max_context_tokens < 1:
        raise ValueError("backend max_context_tokens must be a positive integer")

    eos_token_id = backend.eos_token_id
    if eos_token_id is not None:
        if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool):
            raise TypeError("backend eos_token_id must be an integer or None")
        if eos_token_id < 0:
            raise ValueError("backend eos_token_id must be non-negative when set")
    return eos_token_id


def _validate_cache_mode(cache_mode: CacheMode) -> CacheMode:
    if cache_mode not in {"auto", "stateless", "static"}:
        raise ValueError("cache_mode must be one of: auto, stateless, static")
    return cache_mode


def _validate_runtime_vocab_contract(
    logits: Sequence[float],
    *,
    eos_token_id: int | None,
    stop_token_ids: tuple[int, ...],
) -> None:
    vocab_size = len(logits)
    if vocab_size < 1:
        return
    if eos_token_id is not None and eos_token_id >= vocab_size:
        raise ValueError(
            f"backend eos_token_id {eos_token_id} is outside logits vocabulary "
            f"[0, {vocab_size})"
        )
    invalid_stop_ids = [token_id for token_id in stop_token_ids if token_id >= vocab_size]
    if invalid_stop_ids:
        raise ValueError(
            "stop_token_ids contain IDs outside logits vocabulary: "
            + ", ".join(str(token_id) for token_id in invalid_stop_ids)
        )


def generate_token_ids(
    backend: InferenceBackend,
    prompt_token_ids: Sequence[int],
    config: GenerationConfig | None = None,
    *,
    cache_mode: CacheMode = "auto",
) -> GenerationResult:
    """Generate from caller-supplied token IDs through the canonical generation loop.

    ``cache_mode='stateless'`` recomputes the full prefix for each decode step.
    ``cache_mode='static'`` requires the backend's accepted incremental/static-KV
    session. ``cache_mode='auto'`` preserves the historical behavior: use that
    session when present, otherwise fall back to stateless generation.
    """

    config = config or GenerationConfig()
    eos_token_id = _validate_backend_contract(backend)
    cache_mode = _validate_cache_mode(cache_mode)

    prompt_ids = _validated_token_ids(prompt_token_ids, field="prompt token IDs")
    if not prompt_ids:
        raise ValueError("prompt token IDs must be non-empty")
    if len(prompt_ids) > backend.max_context_tokens:
        raise ValueError(
            f"prompt has {len(prompt_ids)} tokens but backend max_context_tokens="
            f"{backend.max_context_tokens}"
        )

    generated: list[int] = []
    rng = random.Random(config.seed)
    stop_reason: StopReason = "max_new_tokens"
    matched_stop: str | None = None
    session: _GenerationSession | None = None
    if cache_mode != "stateless":
        session = _begin_incremental_session(
            backend,
            prompt_ids,
            required=cache_mode == "static",
        )

    try:
        for step_index in range(config.max_new_tokens):
            input_length = len(prompt_ids) + len(generated)
            if input_length >= backend.max_context_tokens:
                stop_reason = "context_limit"
                break

            if session is None:
                logits = backend.next_token_logits((*prompt_ids, *generated))
            else:
                logits = session.next_token_logits()
            _validate_runtime_vocab_contract(
                logits,
                eos_token_id=eos_token_id,
                stop_token_ids=config.stop_token_ids,
            )
            if config.sample:
                token_id = sample_token(
                    logits,
                    rng=rng,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                )
            else:
                token_id = greedy_token(logits)
            generated.append(token_id)

            if eos_token_id is not None and token_id == eos_token_id:
                stop_reason = "eos"
                break
            if token_id in config.stop_token_ids:
                stop_reason = "stop_token"
                break

            if config.stop_strings:
                current_text = backend.decode(generated)
                matched_stop = next(
                    (stop for stop in config.stop_strings if current_text.endswith(stop)),
                    None,
                )
                if matched_stop is not None:
                    stop_reason = "stop_string"
                    break

            needs_another_step = step_index + 1 < config.max_new_tokens
            has_context_room = len(prompt_ids) + len(generated) < backend.max_context_tokens
            if session is not None and needs_another_step and has_context_room:
                session.append(token_id)
    finally:
        if session is not None:
            session.close()

    text = backend.decode(generated)
    if matched_stop is not None and config.strip_stop_strings:
        text = text[: -len(matched_stop)]

    return GenerationResult(
        prompt_token_ids=prompt_ids,
        generated_token_ids=tuple(generated),
        text=text,
        stop_reason=stop_reason,
    )


def generate(
    backend: InferenceBackend,
    prompt: str,
    config: GenerationConfig | None = None,
    *,
    cache_mode: CacheMode = "auto",
) -> GenerationResult:
    """Generate from text encoded only by the tokenizer bound to ``backend``."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    prompt_token_ids = _validated_token_ids(backend.encode(prompt), field="prompt token IDs")
    if not prompt_token_ids:
        raise ValueError("prompt encoded to zero tokens; backend must provide a non-empty context")
    return generate_token_ids(
        backend,
        prompt_token_ids,
        config,
        cache_mode=cache_mode,
    )
