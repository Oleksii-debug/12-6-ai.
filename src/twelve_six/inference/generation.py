from __future__ import annotations

import random

from .contracts import GenerationConfig, GenerationResult, InferenceBackend, StopReason
from .sampling import greedy_token, sample_token


def _first_text_stop(text: str, stop_strings: tuple[str, ...]) -> tuple[int, int] | None:
    """Return the earliest textual stop as ``(start, end)``.

    A backend token may decode to more than one character. Searching only with
    ``endswith`` can miss a stop sequence that appears inside that token followed
    by additional decoded characters. Earliest start wins; when starts tie, the
    stop that finishes first wins, then caller order is stable.
    """

    matches: list[tuple[int, int, int]] = []
    for order, stop in enumerate(stop_strings):
        start = text.find(stop)
        if start >= 0:
            matches.append((start, start + len(stop), order))
    if not matches:
        return None
    start, end, _ = min(matches)
    return start, end


def _decode_text(backend: InferenceBackend, token_ids: list[int]) -> str:
    text = backend.decode(token_ids)
    if not isinstance(text, str):
        raise TypeError("backend decode must return a string")
    return text


def generate(
    backend: InferenceBackend,
    prompt: str,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    config = config or GenerationConfig()
    if not isinstance(backend.max_context_tokens, int) or isinstance(
        backend.max_context_tokens, bool
    ):
        raise TypeError("backend max_context_tokens must be an integer")
    if backend.max_context_tokens < 1:
        raise ValueError("backend max_context_tokens must be a positive integer")

    prompt_token_ids = backend.encode(prompt)
    if not isinstance(prompt_token_ids, list):
        raise TypeError("backend encode must return a list of token IDs")
    if not prompt_token_ids:
        raise ValueError("prompt encoded to zero tokens; backend must provide a non-empty context")
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in prompt_token_ids
    ):
        raise TypeError("backend encode must return integer token IDs")
    if len(prompt_token_ids) > backend.max_context_tokens:
        raise ValueError(
            f"prompt has {len(prompt_token_ids)} tokens but backend max_context_tokens="
            f"{backend.max_context_tokens}"
        )

    generated: list[int] = []
    rng = random.Random(config.seed)
    stop_reason: StopReason = "max_new_tokens"
    matched_text: str | None = None
    matched_stop_range: tuple[int, int] | None = None

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
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("sampler produced an invalid token ID")
        generated.append(token_id)

        if backend.eos_token_id is not None and token_id == backend.eos_token_id:
            stop_reason = "eos"
            break
        if token_id in config.stop_token_ids:
            stop_reason = "stop_token"
            break

        if config.stop_strings:
            current_text = _decode_text(backend, generated)
            matched_stop_range = _first_text_stop(current_text, config.stop_strings)
            if matched_stop_range is not None:
                matched_text = current_text
                stop_reason = "stop_string"
                break

    if matched_text is not None and matched_stop_range is not None:
        start, end = matched_stop_range
        text = matched_text[:start] if config.strip_stop_strings else matched_text[:end]
    else:
        text = _decode_text(backend, generated)

    return GenerationResult(
        prompt_token_ids=tuple(prompt_token_ids),
        generated_token_ids=tuple(generated),
        text=text,
        stop_reason=stop_reason,
    )
