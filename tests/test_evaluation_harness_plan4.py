from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from tools.evaluation_harness import (
    EVALUATOR_SHA256,
    HarnessError,
    evaluate_suite,
    freeze_suite,
    validate_suite,
)


def protocol(*, metrics=None, seed=17):
    return freeze_suite(
        suite_id="public-fixture-v1",
        dataset_version="v2026-fixture",
        task_type="exact_match",
        tasks=[
            {"id": "a", "answer": "sealed-answer-A"},
            {"id": "b", "answer": "sealed-answer-B"},
        ],
        seed=seed,
        metrics=metrics,
    )


def models():
    return [
        {
            "candidate_id": "champion", "role": "champion", "model_sha256": "1" * 64,
            "predictions": {"a": "sealed-answer-A", "b": "sealed-answer-B"},
        },
        {
            "candidate_id": "trial", "role": "candidate", "model_sha256": "2" * 64,
            "predictions": {"a": "sealed-answer-A", "b": "wrong"},
        },
    ]


def test_frozen_suite_and_cross_candidate_machine_report():
    p = protocol()
    report = evaluate_suite(p, candidates=models())
    assert validate_suite(p) == p
    assert report["schema_version"] == "12-6.plan4-evaluation-report.v1"
    assert report["status"] == "COMPLETE"
    assert report["protocol_sha256"] == p["protocol_sha256"]
    assert report["evaluator_sha256"] == EVALUATOR_SHA256
    assert report["seed"] == 17 and report["task_count"] == 2
    assert report["candidates"][0]["metrics"]["accuracy"]["value"] == 1.0
    assert report["candidates"][1]["metrics"]["accuracy"]["value"] == 0.5
    assert report["candidates"][0]["protocol_sha256"] == report["candidates"][1]["protocol_sha256"]
    assert report["comparisons"][0]["status"] == "COMPARABLE"
    assert report["comparisons"][0]["delta"] == -0.5
    assert report["comparisons"][0]["higher_is_better"] is True
    assert not report["reserved_final_test_access_authorized"]
    assert not report["training_authorized"]
    assert not report["compute_authorized"]
    dumped = json.dumps(report, sort_keys=True)
    assert "sealed-answer-A" not in dumped and "sealed-answer-B" not in dumped
    payload = {key: val for key, val in report.items() if key != "report_sha256"}
    expected = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                    allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    assert report["report_sha256"] == expected


def test_confidence_uncertainty_are_bounded_and_deterministic():
    result = evaluate_suite(protocol(), candidates=models())
    metric = result["candidates"][1]["metrics"]
    low = metric["wilson95_lower"]["value"]
    high = metric["wilson95_upper"]["value"]
    assert 0 <= low < .5 < high <= 1
    assert metric["stderr"]["value"] > 0
    assert evaluate_suite(protocol(), candidates=models()) == result


def test_seed_dataset_task_and_metric_changes_rebind_protocol():
    first = protocol()
    assert first["protocol_sha256"] != protocol(seed=18)["protocol_sha256"]
    different_version = freeze_suite(
        suite_id="public-fixture-v1", dataset_version="new-version",
        task_type="exact_match", tasks=first["tasks"], seed=17,
    )
    assert first["protocol_sha256"] != different_version["protocol_sha256"]
    changed_task = copy.deepcopy(first)
    changed_task["tasks"][1]["answer"] = "different"
    with pytest.raises(HarnessError, match="identity"):
        validate_suite(changed_task)
    assert first["protocol_sha256"] != protocol(metrics=["accuracy"])["protocol_sha256"]


def test_serialization_restart_and_repeated_evaluation_identical():
    p = json.loads(json.dumps(protocol()))
    candidates = json.loads(json.dumps(models()))
    first = evaluate_suite(p, candidates=candidates)
    restarted = evaluate_suite(json.loads(json.dumps(p)), candidates=json.loads(json.dumps(candidates)))
    assert first == restarted


@pytest.mark.parametrize("alter", [
    lambda p: p.update({"protocol_sha256": "0" * 64}),
    lambda p: p.update({"seed": 99}),
    lambda p: p.update({"unknown_field": True}),
    lambda p: p["tasks"].reverse(),
    lambda p: p.update({"schema_version": "older"}),
])
def test_protocol_tamper_fails_closed(alter):
    p = protocol()
    alter(p)
    with pytest.raises(HarnessError):
        evaluate_suite(p, candidates=models())


@pytest.mark.parametrize("broken", [
    [{"id": "a", "answer": "yes"}, {"id": "a", "answer": "no"}],
    [{"id": "a", "answer": True}],
    [{"id": "a", "answer": "ok", "ignored": "bad"}],
    [],
])
def test_invalid_task_shape_rejected(broken):
    with pytest.raises(HarnessError):
        freeze_suite(suite_id="s", dataset_version="v", task_type="exact_match",
                     tasks=broken, seed=5)


def test_bool_alias_seed_rejected():
    with pytest.raises(HarnessError, match="seed"):
        protocol(seed=True)


def test_missing_predictions_mark_error_not_zero_or_pass():
    runs = models()
    runs[1]["predictions"] = {"a": "sealed-answer-A"}
    report = evaluate_suite(protocol(), candidates=runs)
    assert report["status"] == "INCOMPLETE"
    assert report["candidates"][1]["status"] == "ERROR"
    assert report["candidates"][1]["metrics"]["accuracy"] == {
        "status": "NOT_COMPUTED", "value": None
    }
    assert report["comparisons"][0]["status"] == "UNAVAILABLE"
    assert report["comparisons"][0]["delta"] is None


def test_unsupported_requested_metric_is_not_zero_or_pass():
    report = evaluate_suite(protocol(metrics=["accuracy", "unknown_quality"]),
                            candidates=models())
    assert report["status"] == "INCOMPLETE"
    for candidate in report["candidates"]:
        assert candidate["status"] == "UNSUPPORTED"
        assert candidate["metrics"]["unknown_quality"] == {
            "status": "UNSUPPORTED", "value": None
        }
    assert report["comparisons"][0]["status"] == "UNAVAILABLE"


def test_failure_then_recovery_to_identical_frozen_protocol():
    p = protocol()
    bad = models()
    bad[0]["predictions"] = {"a": "sealed-answer-A"}
    assert evaluate_suite(p, candidates=bad)["status"] == "INCOMPLETE"
    recovered = evaluate_suite(p, candidates=models())
    assert recovered["status"] == "COMPLETE"
    assert recovered == evaluate_suite(json.loads(json.dumps(p)), candidates=models())


def test_numeric_suite_bootstrap_uncertainty_and_comparison():
    p = freeze_suite(
        suite_id="public-regression-v1", dataset_version="r1", task_type="numeric",
        tasks=[{"id": "x", "answer": 2}, {"id": "y", "answer": 4},
               {"id": "z", "answer": 6}], seed=9,
    )
    report = evaluate_suite(p, candidates=[
        {"candidate_id": "best", "role": "champion", "model_sha256": "a" * 64,
         "predictions": {"x": 2, "y": 4, "z": 6}},
        {"candidate_id": "alt", "role": "candidate", "model_sha256": "b" * 64,
         "predictions": {"x": 3, "y": 5, "z": 7}},
    ])
    assert report["status"] == "COMPLETE"
    assert report["comparisons"][0]["metric"] == "mae"
    assert report["comparisons"][0]["delta"] == 1.0
    assert report["comparisons"][0]["higher_is_better"] is False
    metrics = report["candidates"][1]["metrics"]
    assert metrics["mae"]["value"] == 1
    assert metrics["bootstrap95_lower"]["value"] == 1
    assert metrics["bootstrap95_upper"]["value"] == 1
    assert metrics["stderr"]["value"] == 0
    assert evaluate_suite(p, candidates=[
        {"candidate_id": "best", "role": "champion", "model_sha256": "a" * 64,
         "predictions": {"x": 2, "y": 4, "z": 6}},
        {"candidate_id": "alt", "role": "candidate", "model_sha256": "b" * 64,
         "predictions": {"x": 3, "y": 5, "z": 7}},
    ]) == report


def test_numeric_nan_candidate_does_not_hide_failure():
    p = freeze_suite(
        suite_id="reg", dataset_version="v1", task_type="numeric",
        tasks=[{"id": "x", "answer": 2.0}], seed=11,
    )
    entries = [
        {"candidate_id": "baseline", "role": "champion", "model_sha256": "1" * 64,
         "predictions": {"x": 2.0}},
        {"candidate_id": "nan", "role": "candidate", "model_sha256": "2" * 64,
         "predictions": {"x": math.nan}},
    ]
    report = evaluate_suite(p, candidates=entries)
    assert report["status"] == "INCOMPLETE"
    assert report["candidates"][1]["metrics"]["mae"]["value"] is None
    assert report["comparisons"][0]["delta"] is None


@pytest.mark.parametrize("mutate", [
    lambda runs: runs.append(copy.deepcopy(runs[1])),
    lambda runs: runs[0].update({"role": "candidate"}),
    lambda runs: runs[1].update({"role": "champion"}),
    lambda runs: runs[1].update({"model_sha256": "not-a-sha"}),
    lambda runs: runs[1].update({"untrusted_extra": "yes"}),
])
def test_ambiguous_candidate_or_champion_fails_closed(mutate):
    runs = models()
    mutate(runs)
    with pytest.raises(HarnessError):
        evaluate_suite(protocol(), candidates=runs)


def test_frozen_metric_identity_protects_incompatible_candidate_comparison():
    p = protocol()
    tampered = copy.deepcopy(p)
    tampered["metrics"] = ["accuracy"]
    with pytest.raises(HarnessError):
        evaluate_suite(tampered, candidates=models())
