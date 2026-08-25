from __future__ import annotations

import random
from collections.abc import Sequence

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token


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


def generate(
    backend: InferenceBackend,
    prompt: str,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    config = config or GenerationConfig()
    eos_token_id = _validate_backend_contract(backend)

    prompt_token_ids = _validated_token_ids(backend.encode(prompt), field="prompt token IDs")
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

    text = backend.decode(generated)
    if matched_stop is not None and config.strip_stop_strings:
        text = text[: -len(matched_stop)]

    return GenerationResult(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=tuple(generated),
        text=text,
        stop_reason=stop_reason,
    )
