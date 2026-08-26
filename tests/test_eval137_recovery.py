from __future__ import annotations

import pytest

from twelve_six.data import source_intake
from twelve_six.data.multilingual_pretraining import MultilingualDataError, detect_language
from twelve_six.training.eval137_recovery import (
    DOMINANT_SCRIPT_MIN_RATIO,
    dominance_aware_detect_language,
    eval137_language_gate,
)


def _ukrainian_with_roman_identifiers() -> str:
    body = (
        "Це український нормативний приклад для перевірки мови та даних моделі. "
        "Українська мова залишається основною у всьому довгому документі. "
    ) * 80
    identifiers = " ".join(["VIII IX VI IV XII"] * 12)
    return body + identifiers


def test_recovery_accepts_dominant_ukrainian_with_many_roman_identifiers() -> None:
    text = _ukrainian_with_roman_identifiers()
    with pytest.raises(MultilingualDataError, match="conflicts with detected 'mixed'"):
        detect_language(text, language_hint="uk")

    evidence = dominance_aware_detect_language(text, language_hint="uk")
    assert evidence.label == "uk"
    assert evidence.reason == "uk-dominant-script-with-minority-script"
    assert evidence.script.cyrillic_letters / evidence.script.alphabetic_letters >= (
        DOMINANT_SCRIPT_MIN_RATIO
    )


def test_recovery_does_not_accept_balanced_mixed_script_text() -> None:
    ukrainian = "Українська мова і дані моделі залишаються у цьому прикладі. " * 20
    english = "The English language and model data remain in this example. " * 20
    with pytest.raises(MultilingualDataError, match="conflicts with detected 'mixed'"):
        dominance_aware_detect_language(ukrainian + english, language_hint="uk")


def test_recovery_gate_is_scoped_and_restores_incumbent_function() -> None:
    original = source_intake.detect_language
    with eval137_language_gate():
        assert source_intake.detect_language is dominance_aware_detect_language
    assert source_intake.detect_language is original
