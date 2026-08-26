"""Recovery-only language admission seam for EVAL-137.

The incumbent multilingual LID intentionally fails closed when both Latin and
Cyrillic scripts exceed an absolute count.  That is useful for short records but
misclassifies long, otherwise single-language documents containing many Roman
numerals, citations, or identifiers.  EVAL-137 needs a source-agnostic repair:
accept the declared reviewed language only when its script still contributes at
least 90% of alphabetic letters and the incumbent lexical/script evidence for
that language is present.  Balanced mixed-script text remains rejected.

This module does not change corpus bytes, family partitions, optimizer settings,
or evaluation records.  The override is installed only around the EVAL-137 CLI
process and is restored afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, cast

from twelve_six.data import source_intake
from twelve_six.data.multilingual_pretraining import (
    LanguageEvidence,
    MultilingualDataError,
    detect_language as incumbent_detect_language,
)

DOMINANT_SCRIPT_MIN_RATIO = 0.90


def dominance_aware_detect_language(
    text: str,
    *,
    modality: Literal["natural", "code"] = "natural",
    language_hint: str | None = None,
) -> LanguageEvidence:
    """Preserve incumbent LID except for overwhelmingly dominant expected script."""
    try:
        return incumbent_detect_language(
            text,
            modality=modality,
            language_hint=language_hint,
        )
    except MultilingualDataError:
        if modality != "natural" or language_hint not in {"uk", "en"}:
            raise

        evidence = incumbent_detect_language(
            text,
            modality=modality,
            language_hint=None,
        )
        if evidence.label != "mixed":
            raise

        alpha = max(evidence.script.alphabetic_letters, 1)
        if language_hint == "uk":
            dominant_ratio = evidence.script.cyrillic_letters / alpha
            has_language_evidence = (
                evidence.script.ukrainian_specific_letters > 0
                or evidence.ukrainian_lexical_hits >= 2
            )
        else:
            dominant_ratio = evidence.script.latin_letters / alpha
            has_language_evidence = evidence.english_lexical_hits >= 1

        if dominant_ratio < DOMINANT_SCRIPT_MIN_RATIO or not has_language_evidence:
            raise

        label = cast(Literal["uk", "en"], language_hint)
        return LanguageEvidence(
            label=label,
            confidence=min(1.0, 0.60 + 0.40 * dominant_ratio),
            script=evidence.script,
            ukrainian_lexical_hits=evidence.ukrainian_lexical_hits,
            english_lexical_hits=evidence.english_lexical_hits,
            reason=f"{label}-dominant-script-with-minority-script",
        )


@contextmanager
def eval137_language_gate() -> Iterator[None]:
    """Install the generic dominance rule only for one EVAL-137 execution."""
    original = source_intake.detect_language
    source_intake.detect_language = dominance_aware_detect_language
    try:
        yield
    finally:
        source_intake.detect_language = original
