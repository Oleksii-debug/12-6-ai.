from __future__ import annotations

from twelve_six.data.document_quality import assess_document as assess_incumbent
from twelve_six.data.document_quality_v2 import (
    NATURAL_DIVERSITY_WINDOW_TOKENS,
    assess_document,
    diversity_window_evidence,
)


def _alpha_word(index: int) -> str:
    first = chr(ord("a") + (index // 26) % 26)
    second = chr(ord("a") + index % 26)
    return f"lexeme{first}{second}"


def _cyclic_text(unique_words: int, total_tokens: int) -> str:
    vocabulary = [_alpha_word(index) for index in range(unique_words)]
    return " ".join(vocabulary[index % unique_words] for index in range(total_tokens))


def test_long_rich_document_is_not_rejected_only_because_global_ttr_shrinks() -> None:
    text = _cyclic_text(unique_words=80, total_tokens=2048)

    incumbent = assess_incumbent("long-rich", text, "en")
    assert incumbent.reasons == ("low_token_diversity",)
    assert incumbent.features.distinct_token_ratio < 0.20

    evidence = diversity_window_evidence(text, "en")
    assert evidence.used_windowed_decision is True
    assert evidence.window_tokens == NATURAL_DIVERSITY_WINDOW_TOKENS
    assert evidence.evaluated_windows == 8
    assert evidence.low_diversity_windows == 0
    assert evidence.median_distinct_token_ratio > 0.20

    repaired = assess_document("long-rich", text, "en")
    assert repaired.accepted is True
    assert repaired.reasons == ()
    assert "global_ttr_below_windowed_diversity_floor" in repaired.warnings


def test_systematically_repetitive_long_document_still_fails() -> None:
    text = _cyclic_text(unique_words=16, total_tokens=2048)

    evidence = diversity_window_evidence(text, "en")
    assert evidence.used_windowed_decision is True
    assert evidence.low_diversity_fraction == 1.0

    incumbent = assess_incumbent("long-repetitive", text, "en")
    repaired = assess_document("long-repetitive", text, "en")
    assert "low_token_diversity" in incumbent.reasons
    assert repaired.accepted is False
    assert "low_token_diversity" in repaired.reasons


def test_short_natural_language_semantics_are_unchanged() -> None:
    text = _cyclic_text(unique_words=70, total_tokens=120)
    assert assess_document("short", text, "en") == assess_incumbent("short", text, "en")
    assert diversity_window_evidence(text, "en").used_windowed_decision is False


def test_code_mode_semantics_are_unchanged() -> None:
    text = "def add(left, right):\n    return left + right\n"
    assert assess_document("code", text, "code") == assess_incumbent("code", text, "code")
