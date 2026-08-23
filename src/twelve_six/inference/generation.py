from __future__ import annotations

import random

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token


def generate(
    backend: InferenceBackend,
    prompt: str,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    config = config or GenerationConfig()
    prompt_token_ids = backend.encode(prompt)
    if not prompt_token_ids:
        raise ValueError("prompt encoded to zero tokens; backend must provide a non-empty context")

    generated: list[int] = []
    rng = random.Random(config.seed)
    stop_reason: StopReason = "max_new_tokens"
    matched_stop: str | None = None

    for _ in range(config.max_new_tokens):
        logits = backend.next_token_logits((*prompt_token_ids, *generated))
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

    text = backend.decode(generated)
    if matched_stop is not None and config.strip_stop_strings:
        text = text[: -len(matched_stop)]

    return GenerationResult(
        prompt_token_ids=tuple(prompt_token_ids),
        generated_token_ids=tuple(generated),
        text=text,
        stop_reason=stop_reason,
    )
