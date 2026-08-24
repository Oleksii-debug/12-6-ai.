from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _validated_logits(logits: Sequence[float]) -> list[float]:
    values = [float(value) for value in logits]
    if not values:
        raise ValueError("logits must not be empty")
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("logits must not contain NaN or +inf")
    if all(value == -math.inf for value in values):
        raise ValueError("at least one logit must be finite")
    return values


def _finite_number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be a finite number")
    return numeric


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
    temperature_value = _finite_number(temperature, field="temperature")
    if temperature_value <= 0:
        raise ValueError("temperature must be > 0")
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer when set")
    top_p_value = _finite_number(top_p, field="top_p")
    if not 0 < top_p_value <= 1:
        raise ValueError("top_p must be in (0, 1]")

    # Subtract before temperature scaling. Dividing large finite logits by a
    # tiny positive temperature first can overflow to +inf, after which the
    # conventional ``scaled - max(scaled)`` normalization produces NaN. The
    # delta is always <= 0, so this ordering is stable even at subnormal
    # temperatures and underflow merely gives a zero-probability candidate.
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
