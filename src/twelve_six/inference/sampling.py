from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _validated_logits(logits: Sequence[float]) -> list[float]:
    values: list[float] = []
    for value in logits:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("logits must contain only real numbers")
        number = float(value)
        if math.isnan(number) or number == math.inf:
            raise ValueError("logits must not contain NaN or +inf")
        values.append(number)
    if not values:
        raise ValueError("logits must not be empty")
    if all(value == -math.inf for value in values):
        raise ValueError("at least one logit must be finite")
    return values


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

    # Subtract before dividing by temperature. Scaling the raw logits first can
    # overflow to +inf for a finite but tiny temperature, creating inf-inf/NaN
    # weights even though the underlying softmax is perfectly well-defined.
    max_logit = max(values)
    candidates = [
        (
            index,
            0.0
            if value == -math.inf
            else math.exp((value - max_logit) / temperature_value),
        )
        for index, value in enumerate(values)
    ]
    candidates = [candidate for candidate in candidates if candidate[1] > 0]
    candidates.sort(key=lambda item: (-item[1], item[0]))

    if top_k is not None:
        candidates = candidates[: min(top_k, len(candidates))]

    if top_p_value < 1.0:
        total = sum(weight for _, weight in candidates)
        cumulative = 0.0
        nucleus: list[tuple[int, float]] = []
        for candidate in candidates:
            nucleus.append(candidate)
            cumulative += candidate[1] / total
            if cumulative >= top_p_value:
                break
        candidates = nucleus

    total = sum(weight for _, weight in candidates)
    threshold = rng.random() * total
    cumulative = 0.0
    for token_id, weight in candidates:
        cumulative += weight
        if threshold < cumulative:
            return token_id
    return candidates[-1][0]
