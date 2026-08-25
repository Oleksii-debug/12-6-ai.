import math

import pytest

from twelve_six.research_decision import (
    Decision,
    DecisionConfig,
    MetricDirection,
    MetricPurpose,
    Pair,
    SelectionMetricError,
    analyze_paired_runs,
)


def cfg(materiality=0.01):
    return DecisionConfig(
        materiality=materiality,
        metric_name="held_out_validation_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )


def test_head_count_tiny_three_seed_delta_is_practical_tie():
    # MODEL-13 supporting paired h2-h4 deltas were h2-h4 =
    # [-1.1e-5, -4.5e-5, -1.67e-4]. Lower is better, so orient them positive.
    result = analyze_paired_runs(
        [
            Pair("seed-101", oriented_delta=1.1e-5),
            Pair("seed-202", oriented_delta=4.5e-5),
            Pair("seed-303", oriented_delta=1.67e-4),
        ],
        candidate="mha_h2_hd24",
        baseline="mha_h4_hd12_control",
        config=cfg(materiality=0.005),
    )
    assert result.candidate_wins == 3
    assert result.decision == Decision.PRACTICAL_TIE
    assert result.winner is None
    assert result.repeat_recommendation.additional_repeats == 0
    assert result.uncertainty_interval is not None
    assert result.uncertainty_interval.method == "exact_paired_bootstrap_percentile"


def test_single_seed_cannot_promote_even_large_difference():
    result = analyze_paired_runs(
        [Pair("seed-1337", baseline=4.18, candidate=3.70)],
        candidate="candidate",
        baseline="control",
        config=cfg(),
    )
    assert result.decision == Decision.INSUFFICIENT_REPEATS
    assert result.repeat_recommendation.target_repeats == 3
    assert result.repeat_recommendation.additional_repeats == 2


def test_two_seeds_cannot_promote():
    result = analyze_paired_runs(
        [Pair("s1", oriented_delta=0.2), Pair("s2", oriented_delta=0.25)],
        candidate="candidate",
        baseline="control",
        config=cfg(),
    )
    assert result.decision == Decision.INSUFFICIENT_REPEATS
    assert result.repeat_recommendation.additional_repeats == 1


def test_material_consistent_three_seed_effect_can_clear_without_p_value():
    result = analyze_paired_runs(
        [
            Pair("s1", oriented_delta=0.030),
            Pair("s2", oriented_delta=0.026),
            Pair("s3", oriented_delta=0.034),
        ],
        candidate="candidate",
        baseline="control",
        config=cfg(materiality=0.01),
    )
    assert result.decision == Decision.CLEAR_WIN
    assert result.winner == "candidate"
    assert result.evidence_strength == "small_n_repeatability_not_significance"
    assert "no p-value" in result.inferential_claim


def test_sign_flips_with_material_point_estimate_are_unstable():
    result = analyze_paired_runs(
        [
            Pair("s1", oriented_delta=0.08),
            Pair("s2", oriented_delta=0.07),
            Pair("s3", oriented_delta=-0.04),
        ],
        candidate="candidate",
        baseline="control",
        config=cfg(materiality=0.01),
    )
    assert result.decision == Decision.UNSTABLE
    assert result.repeat_recommendation.additional_repeats >= 1
    assert result.repeat_recommendation.target_repeats <= 7


def test_strong_negative_effect_reports_baseline_as_winner():
    result = analyze_paired_runs(
        [
            Pair("s1", oriented_delta=-0.030),
            Pair("s2", oriented_delta=-0.026),
            Pair("s3", oriented_delta=-0.034),
        ],
        candidate="candidate",
        baseline="control",
        config=cfg(materiality=0.01),
    )
    assert result.decision == Decision.CLEAR_WIN
    assert result.winner == "control"


def test_final_test_metrics_are_ineligible_for_selection():
    with pytest.raises(SelectionMetricError):
        DecisionConfig(
            materiality=0.01,
            metric_name="final_test_loss",
            metric_purpose=MetricPurpose.FINAL_TEST,
        )


def test_diagnostic_metrics_are_also_ineligible_for_winner_selection():
    with pytest.raises(SelectionMetricError):
        DecisionConfig(
            materiality=0.01,
            metric_name="gradient_norm",
            metric_purpose=MetricPurpose.DIAGNOSTIC_ONLY,
        )


def test_pair_raw_metric_orientation_lower_is_better():
    pair = Pair("s", baseline=3.0, candidate=2.9)
    assert math.isclose(pair.delta(MetricDirection.LOWER_IS_BETTER), 0.1)


def test_duplicate_pair_ids_fail_closed():
    with pytest.raises(ValueError, match="unique"):
        analyze_paired_runs(
            [Pair("s1", oriented_delta=0.1), Pair("s1", oriented_delta=0.2)],
            candidate="candidate",
            baseline="control",
            config=cfg(),
        )
