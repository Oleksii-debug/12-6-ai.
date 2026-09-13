from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _validated_logits(logits: Sequence[float]) -> list[float]:
    try:
        values = [float(value) for value in logits]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("logits must contain real numeric values") from exc
    if not values:
        raise ValueError("logits must not be empty")
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("logits must not contain NaN or +inf")
    if all(value == -math.inf for value in values):
        raise ValueError("at least one logit must be finite")
    return values


def _validated_sampling_parameters(
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
) -> tuple[float, int | None, float]:
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a real number")
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0:
        raise ValueError("temperature must be finite and > 0")
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer when set")
        if top_k <= 0:
            raise ValueError("top_k must be > 0 when set")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise TypeError("top_p must be a real number")
    top_p_value = float(top_p)
    if not math.isfinite(top_p_value) or not 0 < top_p_value <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    return temperature_value, top_k, top_p_value


def greedy_token(logits: Sequence[float]) -> int:
    values = _validated_logits(logits)
    return max(range(len(values)), key=values.__getitem__)


def sample_token(
    logits: Sequence[float],
    *,
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> int:
    values = _validated_logits(logits)
    temperature, top_k, top_p = _validated_sampling_parameters(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    scaled: list[float] = []
    for value in values:
        if value == -math.inf:
            scaled.append(value)
            continue
        try:
            scaled_value = value / temperature
        except OverflowError as exc:
            raise ValueError("temperature scaling overflowed logits") from exc
        if not math.isfinite(scaled_value):
            raise ValueError("temperature scaling produced non-finite logits")
        scaled.append(scaled_value)

    max_logit = max(scaled)
    candidates = [
        (index, 0.0 if value == -math.inf else math.exp(value - max_logit))
        for index, value in enumerate(scaled)
    ]
    candidates = [candidate for candidate in candidates if candidate[1] > 0]
    if not candidates:
        raise ValueError("sampling produced no positive-probability candidates")
    candidates.sort(key=lambda item: (-item[1], item[0]))

    if top_k is not None:
        candidates = candidates[: min(top_k, len(candidates))]

    if top_p < 1.0:
        total = sum(weight for _, weight in candidates)
        cumulative = 0.0
        nucleus: list[tuple[int, float]] = []
        for candidate in candidates:
            nucleus.append(candidate)
            cumulative += candidate[1] / total
            if cumulative >= top_p:
                break
        candidates = nucleus

    total = sum(weight for _, weight in candidates)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("sampling probability mass must be finite and positive")
    threshold = rng.random() * total
    cumulative = 0.0
    for token_id, weight in candidates:
        cumulative += weight
        if threshold < cumulative:
            return token_id
    return candidates[-1][0]
