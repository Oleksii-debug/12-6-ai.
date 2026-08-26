"""Granularity-stable lexical-diversity adapter for D03 document quality.

DATA-296 showed that the incumbent whole-document type/token-ratio (TTR)
threshold can reject long, otherwise clean natural-language sources solely
because TTR decreases with document length. This module preserves every
incumbent quality gate, but evaluates the ``low_token_diversity`` reason for
natural language on deterministic fixed-token windows when the document is
large enough.

It deliberately does not change rights, privacy, language admission, dedup,
decontamination, or code-mode quality semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

from twelve_six.data.document_quality import (
    Mode,
    QualityDecision,
    QualityPolicy,
    assess_document as assess_document_incumbent,
    default_quality_policy,
)

NATURAL_DIVERSITY_WINDOW_TOKENS = 256
_NATURAL_WORD_RE = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True)
class DiversityWindowEvidence:
    """Deterministic evidence for the natural-language diversity decision."""

    token_count: int
    window_tokens: int
    evaluated_windows: int
    low_diversity_windows: int
    low_diversity_fraction: float
    minimum_distinct_token_ratio: float
    median_distinct_token_ratio: float
    decision_distinct_token_ratio: float
    used_windowed_decision: bool


def _natural_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _NATURAL_WORD_RE.finditer(text)]


def _window_slices(
    tokens: list[str], window_tokens: int, min_tail_tokens: int
) -> Iterable[list[str]]:
    for start in range(0, len(tokens), window_tokens):
        window = tokens[start : start + window_tokens]
        if len(window) == window_tokens or len(window) >= min_tail_tokens:
            yield window


def _distinct_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def diversity_window_evidence(
    text: str,
    mode: Mode,
    *,
    policy: QualityPolicy | None = None,
    window_tokens: int = NATURAL_DIVERSITY_WINDOW_TOKENS,
) -> DiversityWindowEvidence:
    """Compute fixed-window diversity evidence without consulting model outcomes."""
    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive")
    policy = policy or default_quality_policy()
    thresholds = policy.thresholds_for(mode)

    if mode == "code":
        return DiversityWindowEvidence(
            token_count=0,
            window_tokens=window_tokens,
            evaluated_windows=0,
            low_diversity_windows=0,
            low_diversity_fraction=0.0,
            minimum_distinct_token_ratio=0.0,
            median_distinct_token_ratio=0.0,
            decision_distinct_token_ratio=0.0,
            used_windowed_decision=False,
        )

    tokens = _natural_tokens(text)
    global_ratio = _distinct_ratio(tokens)
    use_windows = len(tokens) >= 2 * window_tokens
    if not use_windows:
        return DiversityWindowEvidence(
            token_count=len(tokens),
            window_tokens=window_tokens,
            evaluated_windows=0,
            low_diversity_windows=0,
            low_diversity_fraction=0.0,
            minimum_distinct_token_ratio=global_ratio,
            median_distinct_token_ratio=global_ratio,
            decision_distinct_token_ratio=global_ratio,
            used_windowed_decision=False,
        )

    windows = list(
        _window_slices(tokens, window_tokens, thresholds.diversity_min_tokens)
    )
    ratios = [_distinct_ratio(window) for window in windows]
    low = sum(ratio < thresholds.min_distinct_token_ratio for ratio in ratios)
    low_fraction = low / len(ratios) if ratios else 1.0
    decision_ratio = _median(ratios)
    return DiversityWindowEvidence(
        token_count=len(tokens),
        window_tokens=window_tokens,
        evaluated_windows=len(ratios),
        low_diversity_windows=low,
        low_diversity_fraction=round(low_fraction, 6),
        minimum_distinct_token_ratio=round(min(ratios, default=0.0), 6),
        median_distinct_token_ratio=round(decision_ratio, 6),
        decision_distinct_token_ratio=round(decision_ratio, 6),
        used_windowed_decision=True,
    )


def assess_document(
    record_id: str,
    text: str,
    mode: Mode,
    *,
    policy: QualityPolicy | None = None,
) -> QualityDecision:
    """Preserve incumbent gates and repair only length-sensitive TTR rejection."""
    policy = policy or default_quality_policy()
    incumbent = assess_document_incumbent(record_id, text, mode, policy=policy)
    if mode == "code" or "low_token_diversity" not in incumbent.reasons:
        return incumbent

    thresholds = policy.thresholds_for(mode)
    evidence = diversity_window_evidence(text, mode, policy=policy)
    if not evidence.used_windowed_decision:
        return incumbent

    # Majority-window rule: one locally repetitive section must not delete a
    # complete long document, while systematically repetitive documents still
    # fail the same low_token_diversity reason.
    if evidence.low_diversity_fraction >= 0.5:
        return incumbent

    reasons = tuple(
        reason for reason in incumbent.reasons if reason != "low_token_diversity"
    )
    warnings = tuple(
        sorted(
            set(
                (*incumbent.warnings, "global_ttr_below_windowed_diversity_floor")
            )
        )
    )
    score = max(0, 100 - 25 * len(reasons) - 5 * len(warnings))

    diversity_margin = (
        evidence.decision_distinct_token_ratio - thresholds.min_distinct_token_ratio
    ) / max(thresholds.min_distinct_token_ratio, 1e-9)
    edge_margin = round(max(0.0, diversity_margin), 6)

    return replace(
        incumbent,
        accepted=not reasons,
        score=score,
        reasons=reasons,
        warnings=warnings,
        edge_margin=edge_margin,
    )
