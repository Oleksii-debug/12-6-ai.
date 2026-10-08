"""Plan 4 Section 2: acceptance on immutable LOCAL_FREE public fixtures."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tools.evaluation_harness import (
    EvaluationHarnessError,
    compare_candidates,
    evaluate_candidate,
    freeze_protocol,
    freeze_suite,
)


A = "a" * 64
B = "b" * 64
EVALUATOR = "e" * 64


def text_fixture():
    suite = freeze_suite({
        "schema_version": "12-6.plan4-evaluation-suite.v1",
        "suite_id": "text-smoke",
        "suite_version": "fixture-1",
        "visibility": "public_fixture",
        "kind": "text_exact_match",
        "cases": [
            {"id": "q1", "expected": "yes"},
            {"id": "q2", "expected": "no"},
            {"id": "q3", "expected": "maybe"},
            {"id": "q4", "expected": "ok"},
        ],
    })
    protocol = freeze_protocol(
        suite, metrics=("accuracy",), seed=104, evaluator_sha256=EVALUATOR
    )
    return suite, protocol


def numeric_fixture():
    suite = freeze_suite({
        "schema_version": "12-6.plan4-evaluation-suite.v1",
        "suite_id": "number-smoke",
        "suite_version": "fixture-1",
        "visibility": "public_fixture",
        "kind": "numeric_regression",
        "cases": [
            {"id": "q1", "expected": 1.0},
            {"id": "q2", "expected": 2.0},
            {"id": "q3", "expected": 3.0},
            {"id": "q4", "expected": 4.0},
        ],
    })
    protocol = freeze_protocol(
        suite,
        metrics=("root_mean_square_error", "mean_absolute_error"),
        seed=52,
        evaluator_sha256=EVALUATOR,
    )
    return suite, protocol


def test_deterministic_replay_machine_readable_restart_and_ci():
    suite, protocol = text_fixture()
    predictions = {"q1": "yes", "q2": "no", "q3": "bad", "q4": "ok"}
    report = evaluate_candidate(suite, protocol, predictions=predictions, model_sha256=A)
    assert report["status"] == "PASS"
    assert report["sample_count"] == 4
    assert report["metrics"][0]["name"] == "accuracy"
    assert report["metrics"][0]["value"] == 0.75
    low, high = report["metrics"][0]["confidence_interval_95"]
    assert 0 < low <= 0.75 <= high <= 1
    assert "maybe" not in json.dumps(report)
    assert report == evaluate_candidate(
        json.loads(json.dumps(suite)), json.loads(json.dumps(protocol)),
        predictions=dict(reversed(list(predictions.items()))), model_sha256=A
    )


def test_same_frozen_protocol_two_candidates_and_champion_ranking():
    suite, protocol = text_fixture()
    champ = evaluate_candidate(
        suite, protocol, predictions={"q1": "yes", "q2": "no", "q3": "maybe", "q4": "ok"},
        model_sha256=A, role="champion"
    )
    candidate = evaluate_candidate(
        suite, protocol, predictions={"q1": "yes", "q2": "wrong", "q3": "maybe", "q4": "bad"},
        model_sha256=B
    )
    comparison = compare_candidates(suite, protocol, (candidate, champ))
    assert comparison["status"] == "COMPARABLE"
    assert [x["model_sha256"] for x in comparison["rankings"]["accuracy"]] == [A, B]
    assert comparison == compare_candidates(suite, protocol, (champ, candidate))
    assert comparison["protocol_sha256"] == protocol["protocol_sha256"]


def test_numeric_metrics_ci_seed_replay_and_direction():
    suite, protocol = numeric_fixture()
    top = evaluate_candidate(
        suite, protocol, predictions={"q1": 1.0, "q2": 2.0, "q3": 3.0, "q4": 4.0},
        model_sha256=A
    )
    other = evaluate_candidate(
        suite, protocol, predictions={"q1": 2.0, "q2": 2.5, "q3": 2.0, "q4": 3.0},
        model_sha256=B
    )
    assert [v["name"] for v in other["metrics"]] == [
        "mean_absolute_error", "root_mean_square_error"
    ]
    assert all(v["value"] > 0 and v["higher_is_better"] is False
               for v in other["metrics"])
    assert other == evaluate_candidate(
        suite, protocol, predictions={"q1": 2.0, "q2": 2.5, "q3": 2.0, "q4": 3.0},
        model_sha256=B
    )
    comparison = compare_candidates(suite, protocol, (top, other))
    for metric in comparison["rankings"].values():
        assert [x["model_sha256"] for x in metric] == [A, B]


@pytest.mark.parametrize("change", [
    lambda v: v.update({"visibility": "reserved_final_test"}),
    lambda v: v.update({"visibility": "training"}),
    lambda v: v.update({"kind": "unknown"}),
    lambda v: v.update({"cases": []}),
    lambda v: v["cases"].append(deepcopy(v["cases"][0])),
    lambda v: v["cases"][0].update({"expected": True}),
])
def test_untrusted_or_invalid_suite_fails_closed(change):
    suite, _ = text_fixture()
    spec = deepcopy(suite["spec"])
    change(spec)
    with pytest.raises(EvaluationHarnessError):
        freeze_suite(spec)


@pytest.mark.parametrize("metrics", [
    (), ("accuracy", "accuracy"), ("unsupported",),
    ("accuracy", "root_mean_square_error"),
])
def test_metric_unsupported_or_missing_fails_closed(metrics):
    suite, _ = text_fixture()
    with pytest.raises(EvaluationHarnessError, match="metric"):
        freeze_protocol(suite, metrics=metrics, seed=0, evaluator_sha256=EVALUATOR)


@pytest.mark.parametrize("predictions", [
    {"q1": "yes", "q2": "no", "q3": "maybe"},
    {"q1": "yes", "q2": "no", "q3": "maybe", "q4": "ok", "q5": "invented"},
    {"q1": "yes", "q2": "no", "q3": "maybe", "q4": None},
])
def test_failed_partial_and_invalid_predictions_are_errors_not_zero(predictions):
    suite, protocol = text_fixture()
    with pytest.raises(EvaluationHarnessError):
        evaluate_candidate(
            suite, protocol, predictions=predictions, model_sha256=A
        )


def test_protocol_and_suite_mutation_fails_closed():
    suite, protocol = text_fixture()
    drift = deepcopy(suite)
    drift["spec"]["cases"][0]["expected"] = "no"
    with pytest.raises(EvaluationHarnessError, match="digest drift"):
        evaluate_candidate(
            drift, protocol, predictions={"q1": "yes", "q2": "no", "q3": "maybe", "q4": "ok"},
            model_sha256=A
        )
    bad = deepcopy(protocol)
    bad["seed"] += 1
    with pytest.raises(EvaluationHarnessError, match="protocol"):
        evaluate_candidate(
            suite, bad, predictions={"q1": "yes", "q2": "no", "q3": "maybe", "q4": "ok"},
            model_sha256=A
        )


def test_comparison_rejects_mismatched_protocol_duplicate_model_and_tamper():
    suite, protocol = text_fixture()
    report = evaluate_candidate(
        suite, protocol,
        predictions={"q1": "yes", "q2": "no", "q3": "maybe", "q4": "ok"},
        model_sha256=A
    )
    with pytest.raises(EvaluationHarnessError, match="duplicate model"):
        compare_candidates(suite, protocol, (report, report))
    bad = deepcopy(report)
    bad["metrics"][0]["value"] = 0.0
    with pytest.raises(EvaluationHarnessError, match="report digest"):
        compare_candidates(suite, protocol, (report, bad))
    changed = freeze_protocol(
        suite, metrics=("accuracy",), seed=105, evaluator_sha256=EVALUATOR
    )
    other = evaluate_candidate(
        suite, changed, predictions={"q1": "yes", "q2": "no", "q3": "maybe", "q4": "ok"},
        model_sha256=B
    )
    with pytest.raises(EvaluationHarnessError, match="protocol mismatch"):
        compare_candidates(suite, protocol, (report, other))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "3"])
def test_numeric_nonfinite_or_nonnumeric_prediction_fails_closed(value):
    suite, protocol = numeric_fixture()
    with pytest.raises(EvaluationHarnessError, match="numeric prediction"):
        evaluate_candidate(
            suite, protocol, model_sha256=A,
            predictions={"q1": 1, "q2": value, "q3": 3, "q4": 4}
        )


def test_unknown_model_identity_and_confidence_refused():
    suite, protocol = text_fixture()
    with pytest.raises(EvaluationHarnessError, match="confidence level"):
        freeze_protocol(
            suite, metrics=("accuracy",), seed=0, evaluator_sha256=EVALUATOR,
            confidence_level=0.9
        )
    with pytest.raises(EvaluationHarnessError, match="model identity"):
        evaluate_candidate(
            suite, protocol, model_sha256="invalid",
            predictions={"q1": "yes", "q2": "no", "q3": "maybe", "q4": "ok"}
        )
