"""Deterministic small-repeat decision rules for 12-6 research comparisons.

This module is deliberately statistical-policy code, not a hypothesis-testing package.
With two or three repeats it reports descriptive uncertainty and practical materiality;
it does not manufacture asymptotic p-values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import product
import math
import random
from statistics import mean, median, variance
from typing import Iterable, Sequence


class Decision(StrEnum):
    CLEAR_WIN = "CLEAR_WIN"
    PRACTICAL_TIE = "PRACTICAL_TIE"
    UNSTABLE = "UNSTABLE"
    INSUFFICIENT_REPEATS = "INSUFFICIENT_REPEATS"


class MetricDirection(StrEnum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class MetricPurpose(StrEnum):
    SELECTION_VALIDATION = "selection_validation"
    TRAINING = "training"
    FINAL_TEST = "final_test"
    DIAGNOSTIC_ONLY = "diagnostic_only"


class SelectionMetricError(ValueError):
    """Raised when a non-selection metric is offered to the winner selector."""


@dataclass(frozen=True)
class Pair:
    run_id: str
    baseline: float | None = None
    candidate: float | None = None
    oriented_delta: float | None = None

    def delta(self, direction: MetricDirection) -> float:
        raw_values = self.baseline is not None or self.candidate is not None
        if self.oriented_delta is not None and raw_values:
            raise ValueError("pair must provide raw metrics or oriented_delta, not both")
        if self.oriented_delta is not None:
            value = float(self.oriented_delta)
        else:
            if self.baseline is None or self.candidate is None:
                raise ValueError("pair requires baseline and candidate metrics")
            baseline = float(self.baseline)
            candidate = float(self.candidate)
            value = (
                baseline - candidate
                if direction == MetricDirection.LOWER_IS_BETTER
                else candidate - baseline
            )
        if not math.isfinite(value):
            raise ValueError(f"non-finite delta for run {self.run_id!r}")
        return value


@dataclass(frozen=True)
class DecisionConfig:
    materiality: float
    metric_name: str
    metric_purpose: MetricPurpose = MetricPurpose.SELECTION_VALIDATION
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER
    min_repeats: int = 3
    confidence: float = 0.90
    min_win_fraction: float = 0.75
    min_signal_to_noise: float = 0.80
    numeric_tie_epsilon: float = 1e-12
    bootstrap_samples: int = 4096
    bootstrap_seed: int = 140
    max_exploratory_repeats: int = 7

    def __post_init__(self) -> None:
        if self.metric_purpose != MetricPurpose.SELECTION_VALIDATION:
            raise SelectionMetricError(
                "research winner selection accepts selection_validation metrics only; "
                f"got {self.metric_purpose.value}"
            )
        if not math.isfinite(self.materiality) or self.materiality <= 0:
            raise ValueError("materiality must be finite and > 0")
        if self.min_repeats < 3:
            raise ValueError("min_repeats must be >= 3")
        if not (0.5 < self.confidence < 1.0):
            raise ValueError("confidence must be between 0.5 and 1")
        if not (0.5 <= self.min_win_fraction <= 1.0):
            raise ValueError("min_win_fraction must be in [0.5, 1]")
        if self.min_signal_to_noise < 0:
            raise ValueError("min_signal_to_noise must be >= 0")
        if self.max_exploratory_repeats < self.min_repeats:
            raise ValueError("max_exploratory_repeats must be >= min_repeats")


@dataclass(frozen=True)
class UncertaintyInterval:
    lower: float
    upper: float
    confidence: float
    method: str
    interpretation: str


@dataclass(frozen=True)
class RepeatRecommendation:
    current_repeats: int
    target_repeats: int
    additional_repeats: int
    capped: bool
    rationale: str


@dataclass(frozen=True)
class DecisionResult:
    schema: str
    metric_name: str
    metric_purpose: str
    direction: str
    candidate: str
    baseline: str
    repeat_ids: tuple[str, ...]
    repeats: int
    oriented_delta_definition: str
    mean_delta: float | None
    median_delta: float | None
    sample_variance: float | None
    sample_sd: float | None
    uncertainty_interval: UncertaintyInterval | None
    candidate_wins: int
    baseline_wins: int
    numeric_ties: int
    effect_size_vs_run_noise: float | None
    effect_size_status: str
    materiality_threshold: float
    decision: Decision
    winner: str | None
    reason_codes: tuple[str, ...]
    evidence_strength: str
    repeat_recommendation: RepeatRecommendation
    inferential_claim: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take percentile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    fraction = position - lo
    return sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction


def _bootstrap_interval(
    deltas: Sequence[float], *, confidence: float, samples: int, seed: int
) -> UncertaintyInterval | None:
    n = len(deltas)
    if n < 2:
        return None

    boot_means: list[float] = []
    if n <= 5:
        # Enumerate the complete empirical bootstrap distribution. At n=3 this is
        # only 27 resamples, so there is no Monte Carlo randomness to hide behind.
        for indices in product(range(n), repeat=n):
            boot_means.append(sum(deltas[index] for index in indices) / n)
        method = "exact_paired_bootstrap_percentile"
    else:
        rng = random.Random(seed)
        for _ in range(samples):
            boot_means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
        method = f"seeded_paired_bootstrap_percentile_{samples}"

    boot_means.sort()
    alpha = 1.0 - confidence
    return UncertaintyInterval(
        lower=_percentile(boot_means, alpha / 2.0),
        upper=_percentile(boot_means, 1.0 - alpha / 2.0),
        confidence=confidence,
        method=method,
        interpretation=(
            "descriptive empirical bootstrap interval; with small n it is not an "
            "asymptotic significance statement"
        ),
    )


def _effect_size(mean_delta: float, sample_sd: float | None) -> tuple[float | None, str]:
    if sample_sd is None:
        return None, "undefined_fewer_than_two_repeats"
    if sample_sd == 0.0:
        if mean_delta == 0.0:
            return 0.0, "zero_effect_zero_noise"
        return None, "nonzero_effect_zero_observed_noise"
    return mean_delta / sample_sd, "paired_mean_over_run_to_run_sd"


def _recommend_repeats(
    *,
    n: int,
    decision: Decision,
    sample_sd: float | None,
    config: DecisionConfig,
) -> RepeatRecommendation:
    if n < config.min_repeats:
        target = config.min_repeats
        return RepeatRecommendation(
            current_repeats=n,
            target_repeats=target,
            additional_repeats=target - n,
            capped=False,
            rationale="reach the minimum three paired repeats before promoting a research winner",
        )

    if decision in (Decision.CLEAR_WIN, Decision.PRACTICAL_TIE):
        return RepeatRecommendation(
            current_repeats=n,
            target_repeats=n,
            additional_repeats=0,
            capped=False,
            rationale=(
                "observed repeats already resolve the practical decision under the "
                "preregistered rule"
            ),
        )

    if sample_sd is None or sample_sd == 0.0:
        target = min(config.max_exploratory_repeats, n + 1)
        return RepeatRecommendation(
            current_repeats=n,
            target_repeats=target,
            additional_repeats=target - n,
            capped=False,
            rationale="add one paired repeat because the current result is unresolved",
        )

    # Planning heuristic only: target enough paired repeats for the approximate
    # 90% half-width to be no larger than the materiality floor. This does not
    # retroactively turn the small-n analysis into a z-test.
    z90 = 1.6448536269514722
    planned = math.ceil((z90 * sample_sd / config.materiality) ** 2)
    target = max(n + 1, config.min_repeats, planned)
    capped = target > config.max_exploratory_repeats
    target = min(target, config.max_exploratory_repeats)
    return RepeatRecommendation(
        current_repeats=n,
        target_repeats=target,
        additional_repeats=max(0, target - n),
        capped=capped,
        rationale=(
            "variance-proportional repeat planning; if the cap is reached while still unstable, "
            "redesign or enlarge the experiment instead of automatically buying more seeds"
        ),
    )


def analyze_paired_runs(
    pairs: Iterable[Pair],
    *,
    candidate: str,
    baseline: str,
    config: DecisionConfig,
) -> DecisionResult:
    pair_list = list(pairs)
    run_ids = [pair.run_id for pair in pair_list]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("paired repeat IDs must be unique")

    deltas = [pair.delta(config.direction) for pair in pair_list]
    n = len(deltas)
    avg = mean(deltas) if deltas else None
    med = median(deltas) if deltas else None
    var = variance(deltas) if n >= 2 else None
    sd = math.sqrt(var) if var is not None else None
    interval = _bootstrap_interval(
        deltas,
        confidence=config.confidence,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )

    wins = sum(delta > config.numeric_tie_epsilon for delta in deltas)
    losses = sum(delta < -config.numeric_tie_epsilon for delta in deltas)
    ties = n - wins - losses
    effect, effect_status = _effect_size(avg or 0.0, sd)

    decision: Decision
    winner: str | None = None
    reasons: list[str] = []

    if n < config.min_repeats:
        decision = Decision.INSUFFICIENT_REPEATS
        reasons.append("fewer_than_three_paired_repeats")
    else:
        assert avg is not None and med is not None and interval is not None
        required_wins = math.ceil(config.min_win_fraction * n)
        effect_ok = (
            effect is not None and abs(effect) >= config.min_signal_to_noise
        ) or effect_status == "nonzero_effect_zero_observed_noise"

        candidate_clear = (
            avg >= config.materiality
            and med >= config.materiality
            and interval.lower > 0.0
            and wins >= required_wins
            and effect_ok
        )
        baseline_clear = (
            avg <= -config.materiality
            and med <= -config.materiality
            and interval.upper < 0.0
            and losses >= required_wins
            and effect_ok
        )
        practical_tie = (
            abs(avg) < config.materiality
            and abs(med) < config.materiality
            and interval.lower > -config.materiality
            and interval.upper < config.materiality
        )

        if candidate_clear:
            decision = Decision.CLEAR_WIN
            winner = candidate
            reasons.extend(("candidate_material_effect", "directionally_consistent"))
        elif baseline_clear:
            decision = Decision.CLEAR_WIN
            winner = baseline
            reasons.extend(("baseline_material_effect", "directionally_consistent"))
        elif practical_tie:
            decision = Decision.PRACTICAL_TIE
            reasons.append("entire_descriptive_interval_inside_materiality_band")
        else:
            decision = Decision.UNSTABLE
            if interval.lower <= 0.0 <= interval.upper:
                reasons.append("uncertainty_spans_zero")
            if abs(avg) >= config.materiality:
                reasons.append("point_estimate_material_but_not_repeatable_enough")
            else:
                reasons.append("practical_equivalence_not_resolved")

    strength = (
        "none"
        if n == 0
        else "single_repeat_descriptive"
        if n == 1
        else "two_repeat_descriptive"
        if n == 2
        else "small_n_repeatability_not_significance"
        if n <= 5
        else "repeated_empirical"
    )
    repeat_recommendation = _recommend_repeats(
        n=n, decision=decision, sample_sd=sd, config=config
    )

    delta_definition = (
        "positive means candidate is better; baseline-candidate for lower-is-better metrics"
        if config.direction == MetricDirection.LOWER_IS_BETTER
        else "positive means candidate is better; candidate-baseline for higher-is-better metrics"
    )

    return DecisionResult(
        schema="12-6.research-decision.v1",
        metric_name=config.metric_name,
        metric_purpose=config.metric_purpose.value,
        direction=config.direction.value,
        candidate=candidate,
        baseline=baseline,
        repeat_ids=tuple(run_ids),
        repeats=n,
        oriented_delta_definition=delta_definition,
        mean_delta=avg,
        median_delta=med,
        sample_variance=var,
        sample_sd=sd,
        uncertainty_interval=interval,
        candidate_wins=wins,
        baseline_wins=losses,
        numeric_ties=ties,
        effect_size_vs_run_noise=effect,
        effect_size_status=effect_status,
        materiality_threshold=config.materiality,
        decision=decision,
        winner=winner,
        reason_codes=tuple(reasons),
        evidence_strength=strength,
        repeat_recommendation=repeat_recommendation,
        inferential_claim=(
            "practical paired-repeat decision only; no p-value or asymptotic significance claim"
        ),
    )
