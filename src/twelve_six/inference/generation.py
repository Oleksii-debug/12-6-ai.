from __future__ import annotations

import random
from collections.abc import Sequence

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token


def _require_token_ids(token_ids: Sequence[object], *, field: str) -> None:
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError(f"{field} must contain non-negative integer token IDs")


def generate(
    backend: InferenceBackend,
    prompt: str,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    config = config or GenerationConfig()
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if (
        not isinstance(backend.max_context_tokens, int)
        or isinstance(backend.max_context_tokens, bool)
        or backend.max_context_tokens < 1
    ):
        raise ValueError("backend max_context_tokens must be a positive integer")
    if backend.eos_token_id is not None and (
        not isinstance(backend.eos_token_id, int)
        or isinstance(backend.eos_token_id, bool)
        or backend.eos_token_id < 0
    ):
        raise ValueError("backend eos_token_id must be a non-negative integer or None")

    prompt_token_ids = backend.encode(prompt)
    if not isinstance(prompt_token_ids, list):
        raise TypeError("backend encode() must return list[int]")
    _require_token_ids(prompt_token_ids, field="encoded prompt")
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

    for _ in range(config.max_new_tokens):
        input_ids = (*prompt_token_ids, *generated)
        if len(input_ids) >= backend.max_context_tokens:
            stop_reason = "context_limit"
            break

        logits = backend.next_token_logits(input_ids)
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
            if not isinstance(current_text, str):
                raise TypeError("backend decode() must return str")
            matched_stop = next(
                (stop for stop in config.stop_strings if current_text.endswith(stop)),
                None,
            )
            if matched_stop is not None:
                stop_reason = "stop_string"
                break

    text = backend.decode(generated)
    if not isinstance(text, str):
        raise TypeError("backend decode() must return str")
    if matched_stop is not None and config.strip_stop_strings:
        text = text[: -len(matched_stop)]

    return GenerationResult(
        prompt_token_ids=tuple(prompt_token_ids),
        generated_token_ids=tuple(generated),
        text=text,
        stop_reason=stop_reason,
    )
