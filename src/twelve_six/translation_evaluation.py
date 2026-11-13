"""Offline evaluation for an English–Ukrainian translation pilot."""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Literal
from unicodedata import normalize

Direction = Literal["en-uk", "uk-en"]
_DIRECTIONS: tuple[Direction, ...] = ("en-uk", "uk-en")
_CHAR_ORDER = 6
_BETA = 2


@dataclass(frozen=True, slots=True)
class TranslationExample:
    example_id: str
    direction: Direction
    source: str
    reference: str


def _normalized(text: str) -> str:
    return normalize("NFC", text).strip()


def _ngrams(text: str, order: int) -> Counter[str]:
    return Counter(text[start:start + order] for start in range(len(text) - order + 1))


def corpus_chrf(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Return effective-order corpus chrF2 (0–100) for single-reference segments.

    Count clipped character n-gram matches within each segment and then pool
    counts by order. An order without reference n-grams is skipped for that
    segment. Average corpus precision and recall over orders available on both
    sides and apply the F2 formula. Keep case and remove whitespace after NFC
    normalization. See https://aclanthology.org/W15-3049/.
    """
    matched = [0] * _CHAR_ORDER
    predicted = [0] * _CHAR_ORDER
    expected = [0] * _CHAR_ORDER
    for prediction, reference in zip(predictions, references, strict=True):
        hypothesis = "".join(normalize("NFC", prediction).split())
        target = "".join(normalize("NFC", reference).split())
        for index in range(_CHAR_ORDER):
            if index >= len(target):
                break
            order = index + 1
            hypothesis_grams = _ngrams(hypothesis, order)
            target_grams = _ngrams(target, order)
            predicted[index] += max(len(hypothesis) - index, 0)
            expected[index] += len(target) - index
            matched[index] += sum((hypothesis_grams & target_grams).values())
    active = [i for i in range(_CHAR_ORDER) if predicted[i] and expected[i]]
    if not active:
        return 0.0
    precision = mean(matched[i] / predicted[i] for i in active)
    recall = mean(matched[i] / expected[i] for i in active)
    denominator = _BETA**2 * precision + recall
    return 100 * (1 + _BETA**2) * precision * recall / denominator if denominator else 0.0


def evaluate_translation(
    examples: Iterable[TranslationExample],
    translate: Callable[[Direction, str], str],
) -> dict[str, object]:
    """Score a caller-supplied held-out set in both translation directions.

    The generator receives only the direction and source. The caller supplies
    the held-out split. Per-direction metrics include counts, rates in [0, 1]
    and chrF2 in [0, 100]. The macro average gives each direction equal weight.
    """
    rows = tuple(examples)
    seen: set[tuple[Direction, str]] = set()
    for row in rows:
        if row.direction not in _DIRECTIONS:
            raise ValueError(f"unsupported translation direction: {row.direction!r}")
        if any(not isinstance(text, str) or not text.strip()
               for text in (row.example_id, row.source, row.reference)):
            raise ValueError("translation example ID, source and reference must be non-empty text")
        key = (row.direction, row.example_id)
        if key in seen:
            raise ValueError(f"duplicate translation example: {key!r}")
        seen.add(key)
    if {direction for direction, _ in seen} != set(_DIRECTIONS):
        raise ValueError("evaluation requires examples in both translation directions")

    predictions = []
    for row in rows:
        prediction = translate(row.direction, row.source)
        if not isinstance(prediction, str):
            raise TypeError(f"translator returned non-text for example {row.example_id!r}")
        normalized = _normalized(prediction)
        predictions.append({
            "example_id": row.example_id,
            "direction": row.direction,
            "source": row.source,
            "reference": row.reference,
            "prediction": prediction,
            "exact_match": normalized == _normalized(row.reference),
            "source_copy": normalized == _normalized(row.source),
            "empty_output": not normalized,
        })

    by_direction = {}
    for direction in _DIRECTIONS:
        selected = [item for item in predictions if item["direction"] == direction]
        count = len(selected)
        metrics = {"count": count}
        for label in ("exact_match", "source_copy", "empty_output"):
            matches = sum(item[label] for item in selected)
            metrics[f"{label}_count"] = matches
            metrics[f"{label}_rate"] = matches / count
        metrics["chrf"] = corpus_chrf(
            (item["prediction"] for item in selected), (item["reference"] for item in selected),
        )
        by_direction[direction] = metrics
    macro_average = {
        name: mean(result[name] for result in by_direction.values())
        for name in ("exact_match_rate", "source_copy_rate", "empty_output_rate", "chrf")
    }
    return {
        "schema_version": "12-6.translation-pilot-evaluation.v1",
        "example_count": len(rows),
        "metric_config": {
            "unicode_normalization": "NFC",
            "exact_match_whitespace": "trim",
            "chrf_case_sensitive": True,
            "chrf_whitespace": False,
            "chrf_char_order": _CHAR_ORDER,
            "chrf_beta": _BETA,
            "chrf_effective_order": True,
        },
        "by_direction": by_direction,
        "macro_average": macro_average,
        "predictions": predictions,
    }
