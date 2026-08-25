"""DATA-108 real-corpus calibration extension for the incumbent DATA-32 scorer.

This module intentionally imports and reuses DATA-32 feature extraction and decisions.
It only supplies candidate thresholds and evidence aggregation for real source families.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .document_quality import (
    ModeThresholds,
    QualityPolicy,
    assess_document,
    default_quality_policy,
)

REAL_CALIBRATION_SCHEMA = "12-6.document-quality-real-calibration.v1"


def _natural(**overrides: Any) -> ModeThresholds:
    values: dict[str, Any] = {
        "min_chars": 60,
        "max_chars": 250_000,
        "max_symbol_ratio": 0.40,
        "max_repeated_line_ratio": 0.60,
        "max_url_char_ratio": 0.25,
        "max_template_line_ratio": 0.50,
        "max_boilerplate_line_ratio": 0.50,
        "min_distinct_token_ratio": 0.20,
        "max_dominant_token_ratio": 0.22,
        "diversity_min_tokens": 30,
        "max_other_script_letter_ratio": 0.20,
    }
    values.update(overrides)
    return ModeThresholds(**values)


def _code(**overrides: Any) -> ModeThresholds:
    values: dict[str, Any] = {
        "min_chars": 30,
        "max_chars": 400_000,
        "max_symbol_ratio": 0.78,
        "max_repeated_line_ratio": 0.75,
        "max_url_char_ratio": 0.45,
        "max_template_line_ratio": 0.70,
        "max_boilerplate_line_ratio": 0.70,
        "min_distinct_token_ratio": 0.10,
        "max_dominant_token_ratio": 0.38,
        "diversity_min_tokens": 20,
        "max_other_script_letter_ratio": 1.0,
        "min_code_structure_score": 2,
    }
    values.update(overrides)
    return ModeThresholds(**values)


def candidate_quality_policies() -> tuple[QualityPolicy, ...]:
    """Small predeclared threshold set; selection is calibration-only."""
    incumbent = default_quality_policy()
    balanced = QualityPolicy(
        policy_id="data108-real-balanced-v2",
        uk=_natural(
            min_chars=24,
            max_symbol_ratio=0.60,
            max_repeated_line_ratio=0.55,
            max_template_line_ratio=0.40,
            max_boilerplate_line_ratio=0.40,
            min_distinct_token_ratio=0.16,
            max_dominant_token_ratio=0.28,
            max_other_script_letter_ratio=0.60,
        ),
        en=_natural(
            min_chars=24,
            max_symbol_ratio=0.60,
            max_repeated_line_ratio=0.55,
            max_template_line_ratio=0.40,
            max_boilerplate_line_ratio=0.40,
            min_distinct_token_ratio=0.16,
            max_dominant_token_ratio=0.28,
            max_other_script_letter_ratio=0.60,
        ),
        code=_code(
            min_chars=16,
            max_symbol_ratio=0.90,
            max_repeated_line_ratio=0.65,
            max_template_line_ratio=0.60,
            max_boilerplate_line_ratio=0.60,
            min_distinct_token_ratio=0.08,
            max_dominant_token_ratio=0.45,
        ),
        score_weights_version="interpretable-penalty-v1+data108-thresholds",
    )
    preserve = QualityPolicy(
        policy_id="data108-real-preserve-v2",
        uk=_natural(
            min_chars=24,
            max_symbol_ratio=0.65,
            max_repeated_line_ratio=0.65,
            max_url_char_ratio=0.30,
            min_distinct_token_ratio=0.16,
            max_dominant_token_ratio=0.28,
            max_other_script_letter_ratio=0.75,
        ),
        en=_natural(
            min_chars=24,
            max_symbol_ratio=0.65,
            max_repeated_line_ratio=0.65,
            max_url_char_ratio=0.30,
            min_distinct_token_ratio=0.16,
            max_dominant_token_ratio=0.28,
            max_other_script_letter_ratio=0.75,
        ),
        code=_code(
            min_chars=16,
            max_symbol_ratio=0.92,
            max_repeated_line_ratio=0.80,
            max_url_char_ratio=0.50,
            min_distinct_token_ratio=0.08,
            max_dominant_token_ratio=0.48,
        ),
        score_weights_version="interpretable-penalty-v1+data108-thresholds",
    )
    strict = QualityPolicy(
        policy_id="data108-real-strict-v2",
        uk=_natural(
            min_chars=32,
            max_symbol_ratio=0.50,
            max_repeated_line_ratio=0.45,
            max_template_line_ratio=0.30,
            max_boilerplate_line_ratio=0.30,
            min_distinct_token_ratio=0.20,
            max_dominant_token_ratio=0.22,
            max_other_script_letter_ratio=0.40,
        ),
        en=_natural(
            min_chars=32,
            max_symbol_ratio=0.50,
            max_repeated_line_ratio=0.45,
            max_template_line_ratio=0.30,
            max_boilerplate_line_ratio=0.30,
            min_distinct_token_ratio=0.20,
            max_dominant_token_ratio=0.22,
            max_other_script_letter_ratio=0.40,
        ),
        code=_code(
            min_chars=24,
            max_symbol_ratio=0.82,
            max_repeated_line_ratio=0.55,
            max_template_line_ratio=0.50,
            max_boilerplate_line_ratio=0.50,
        ),
        score_weights_version="interpretable-penalty-v1+data108-thresholds",
    )
    return incumbent, balanced, preserve, strict


def evaluate_labeled_rows(
    rows: Sequence[Mapping[str, Any]], policy: QualityPolicy
) -> dict[str, Any]:
    """Evaluate labels with false accept/reject rates by source family and modality."""
    totals: Counter[str] = Counter()
    family: dict[str, Counter[str]] = defaultdict(Counter)
    mode: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, Any]] = []

    for row in rows:
        label = row.get("label")
        if label not in {"ACCEPT", "REJECT"}:
            raise ValueError("label must be ACCEPT or REJECT")
        source_family = str(row.get("source_family", ""))
        if not source_family:
            raise ValueError("source_family is required")
        text = row.get("text")
        sample_mode = row.get("mode")
        if not isinstance(text, str) or sample_mode not in {"uk", "en", "code"}:
            raise ValueError("resolved row requires text and mode uk|en|code")
        decision = assess_document(str(row["id"]), text, sample_mode, policy=policy)
        predicted = "ACCEPT" if decision.accepted else "REJECT"
        is_fa = label == "REJECT" and predicted == "ACCEPT"
        is_fr = label == "ACCEPT" and predicted == "REJECT"
        for bucket in (totals, family[source_family], mode[str(sample_mode)]):
            bucket["samples"] += 1
            bucket["accept_labels"] += label == "ACCEPT"
            bucket["reject_labels"] += label == "REJECT"
            bucket["false_accepts"] += is_fa
            bucket["false_rejects"] += is_fr
        if is_fa or is_fr:
            errors.append(
                {
                    "id": row["id"],
                    "source_family": source_family,
                    "mode": sample_mode,
                    "label": label,
                    "predicted": predicted,
                    "reasons": list(decision.reasons),
                }
            )

    def render(counter: Counter[str]) -> dict[str, Any]:
        accept = counter["accept_labels"]
        reject = counter["reject_labels"]
        return {
            "samples": counter["samples"],
            "false_accepts": counter["false_accepts"],
            "false_rejects": counter["false_rejects"],
            "false_accept_rate_on_rejects": round(counter["false_accepts"] / reject, 6)
            if reject
            else 0.0,
            "false_reject_rate_on_accepts": round(counter["false_rejects"] / accept, 6)
            if accept
            else 0.0,
        }

    family_rendered = {key: render(value) for key, value in sorted(family.items())}
    max_family_errors = max(
        (value["false_accepts"] + value["false_rejects"] for value in family_rendered.values()),
        default=0,
    )
    return {
        "schema_version": REAL_CALIBRATION_SCHEMA,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.manifest()["policy_sha256"],
        "overall": render(totals),
        "by_source_family": family_rendered,
        "by_mode": {key: render(value) for key, value in sorted(mode.items())},
        "max_source_family_errors": max_family_errors,
        "errors": errors,
    }


def select_policy_on_calibration(rows: Sequence[Mapping[str, Any]]) -> tuple[QualityPolicy, list[dict[str, Any]]]:
    """Select only from the fixed candidates using calibration labels, never holdout."""
    results: list[dict[str, Any]] = []
    candidates = candidate_quality_policies()
    for rank, policy in enumerate(candidates):
        report = evaluate_labeled_rows(rows, policy)
        overall = report["overall"]
        objective = (
            report["max_source_family_errors"],
            overall["false_accepts"] + overall["false_rejects"],
            overall["false_rejects"],
            overall["false_accepts"],
            rank,
        )
        results.append({**report, "selection_objective": list(objective)})
    winner_index = min(range(len(results)), key=lambda index: tuple(results[index]["selection_objective"]))
    return candidates[winner_index], results
