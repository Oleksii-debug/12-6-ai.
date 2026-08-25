from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol, cast

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token


class _GenerationSession(Protocol):
    def next_token_logits(self) -> Sequence[float]: ...

    def append(self, token_id: int) -> None: ...

    def close(self) -> None: ...


def _begin_incremental_session(
    backend: InferenceBackend,
    prompt_token_ids: Sequence[int],
) -> _GenerationSession | None:
    factory = getattr(backend, "begin_generation", None)
    if factory is None:
        return None
    if not callable(factory):
        raise TypeError("backend begin_generation must be callable")
    session = factory(prompt_token_ids)
    for method_name in ("next_token_logits", "append", "close"):
        if not callable(getattr(session, method_name, None)):
            raise TypeError(f"incremental generation session must expose {method_name}()")
    return cast(_GenerationSession, session)


def generate(
    backend: InferenceBackend,
    prompt: str,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    config = config or GenerationConfig()
    if not isinstance(backend.max_context_tokens, int) or backend.max_context_tokens < 1:
        raise ValueError("backend max_context_tokens must be a positive integer")

    prompt_token_ids = backend.encode(prompt)
    if not prompt_token_ids:
        raise ValueError("prompt encoded to zero tokens; backend must provide a non-empty context")
    if len(prompt_token_ids) > backend.max_context_tokens:
        raise ValueError(
            f"prompt has {len(prompt_token_ids)} tokens but backend max_context_tokens="
            f"{backend.max_context_tokens}"
        )

    generated: list[int] = []
    rng = random.Random(config.seed)
    stop_reason: StopReason = "max_new_tokens"
    matched_stop: str | None = None
    session = _begin_incremental_session(backend, prompt_token_ids)

    try:
        for step_index in range(config.max_new_tokens):
            input_length = len(prompt_token_ids) + len(generated)
            if input_length >= backend.max_context_tokens:
                stop_reason = "context_limit"
                break

            if session is None:
                logits = backend.next_token_logits((*prompt_token_ids, *generated))
            else:
                logits = session.next_token_logits()
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

            if backend.eos_token_id is not None and token_id == backend.eos_token_id:
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
            has_context_room = len(prompt_token_ids) + len(generated) < backend.max_context_tokens
            if session is not None and needs_another_step and has_context_room:
                session.append(token_id)
    finally:
        if session is not None:
            session.close()

    text = backend.decode(generated)
    if matched_stop is not None and config.strip_stop_strings:
        text = text[: -len(matched_stop)]

    return GenerationResult(
        prompt_token_ids=tuple(prompt_token_ids),
        generated_token_ids=tuple(generated),
        text=text,
        stop_reason=stop_reason,
    )
