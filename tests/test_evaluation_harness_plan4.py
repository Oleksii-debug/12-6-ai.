from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools.evaluation_harness import (
    EvaluationError,
    FrozenProtocol,
    FrozenSuite,
    compare,
    evaluate,
    publish_report,
    read_report,
)


def fixtures():
    suite = FrozenSuite("accuracy-fixture", "v1", (
        ("a", "hi", "yes"), ("b", "bye", "no"), ("c", "other", "maybe")
    ))
    protocol = FrozenProtocol(suite.identity, (5, 17), "exact_match", "fixture-runner-v1")
    return suite, protocol


def test_deterministic_replay_frozen_identity_confidence_and_comparison():
    suite, protocol = fixtures()
    def first(prompt, seed):
        return {"hi": "yes", "bye": "no", "other": "bad"}[prompt]
    def second(prompt, seed):
        return "bad"
    left = evaluate(suite, protocol, candidate_sha256="a"*64, predict=first)
    right = evaluate(suite, protocol, candidate_sha256="b"*64, predict=second)
    assert left == evaluate(suite, protocol, candidate_sha256="a"*64, predict=first)
    assert left["observation_count"] == 6 and left["score"] == pytest.approx(2/3)
    assert left["uncertainty"]["lower"] < left["score"] < left["uncertainty"]["upper"]
    assert compare(left, right)["preferred"] == "first"
    assert "yes" not in json.dumps(left)
    assert "hi" not in json.dumps(left)
    assert left["report_sha256"] != right["report_sha256"]


def test_numeric_metric_and_insufficient_uncertainty_not_zero():
    suite = FrozenSuite("numeric", "v1", (("a", "p", "5"), ("b", "q", "3")))
    protocol = FrozenProtocol(suite.identity, (1,), "mean_absolute_error", "mae-v1")
    report = evaluate(suite, protocol, candidate_sha256="c"*64,
                      predict=lambda prompt, seed: {"p": "3", "q": "2"}[prompt])
    assert report["score"] == 1.5
    assert report["direction"] == "lower"
    assert report["uncertainty"]["method"] == "normal_approx_descriptive_95"
    one = FrozenSuite("single", "v1", (("a", "p", "5"),))
    only = FrozenProtocol(one.identity, (1,), "mean_absolute_error", "mae-v1")
    insufficient = evaluate(one, only, candidate_sha256="c"*64,
                            predict=lambda prompt, seed: "5")
    assert insufficient["uncertainty"]["lower"] is None
    assert insufficient["uncertainty"]["method"] == "insufficient_samples"


@pytest.mark.parametrize("metric", ["unknown", "", "confidence_as_zero"])
def test_unsupported_metric_fails_closed(metric):
    suite, protocol = fixtures()
    with pytest.raises(EvaluationError, match="unsupported metric"):
        replace(protocol, metric=metric)


def test_reserved_suites_and_unfrozen_duplicate_cases_rejected():
    with pytest.raises(EvaluationError, match="EvaluationVault"):
        FrozenSuite("reserved", "v1", (("x", "p", "label"),), visibility="reserved")
    with pytest.raises(EvaluationError, match="duplicate"):
        FrozenSuite("fixture", "v1", (("x", "p", "a"), ("x", "q", "b")))
    with pytest.raises(EvaluationError, match="immutable"):
        FrozenSuite("fixture", "v1", [["x", "p", "a"]])  # type: ignore[arg-type]


def test_seed_validation_and_protocol_drift():
    suite, protocol = fixtures()
    with pytest.raises(EvaluationError, match="seeds"):
        replace(protocol, seeds=(1, 1))
    with pytest.raises(EvaluationError, match="seeds"):
        replace(protocol, seeds=(True,))  # type: ignore[arg-type]
    drift = FrozenSuite(suite.suite_id, "v2", suite.cases)
    with pytest.raises(EvaluationError, match="identity mismatch"):
        evaluate(drift, protocol, candidate_sha256="a"*64, predict=lambda p, s: "yes")


def test_nondeterministic_candidate_and_crash_are_not_masked_as_zero():
    suite, protocol = fixtures()
    counter = [0]
    def drifting(prompt, seed):
        counter[0] += 1
        return str(counter[0])
    with pytest.raises(EvaluationError, match="nondeterministic"):
        evaluate(suite, protocol, candidate_sha256="a"*64, predict=drifting)
    def broken(prompt, seed):
        raise RuntimeError("internal secret label")
    with pytest.raises(EvaluationError, match="candidate execution failed") as err:
        evaluate(suite, protocol, candidate_sha256="a"*64, predict=broken)
    assert "secret" not in str(err.value)


def test_invalid_numeric_and_non_string_candidate_rejected():
    suite = FrozenSuite("numeric", "v1", (("a", "p", "1"),))
    protocol = FrozenProtocol(suite.identity, (0,), "mean_absolute_error", "v1")
    with pytest.raises(EvaluationError, match="non-finite"):
        evaluate(suite, protocol, candidate_sha256="a"*64, predict=lambda p, s: "nan")
    with pytest.raises(EvaluationError, match="unsupported candidate output"):
        evaluate(suite, protocol, candidate_sha256="a"*64, predict=lambda p, s: None)


def test_comparison_rejects_different_suite_seeds_metric_and_same_candidate():
    suite, protocol = fixtures()
    one = evaluate(suite, protocol, candidate_sha256="a"*64, predict=lambda p, s: "yes")
    with pytest.raises(EvaluationError, match="different candidate"):
        compare(one, one)
    altered = FrozenProtocol(suite.identity, (99,), protocol.metric, protocol.evaluator_version)
    two = evaluate(suite, altered, candidate_sha256="b"*64, predict=lambda p, s: "yes")
    with pytest.raises(EvaluationError, match="incomparable"):
        compare(one, two)


def test_machine_readable_report_restart_integrity_and_create_only(tmp_path):
    suite, protocol = fixtures()
    report = evaluate(suite, protocol, candidate_sha256="a"*64,
                      predict=lambda prompt, seed: "yes")
    path = tmp_path / "report.json"
    publish_report(path, report)
    publish_report(path, report)
    assert read_report(path) == report
    mutated = {**report, "score": 0.0}
    with pytest.raises(EvaluationError, match="digest"):
        publish_report(tmp_path / "invalid.json", mutated)
    path.write_bytes(path.read_bytes().replace(b'"score":', b'"scorz":'))
    with pytest.raises(EvaluationError, match="corrupted"):
        read_report(path)
    with pytest.raises(EvaluationError, match="immutable"):
        publish_report(path, report)


def test_replay_adversarial_report_duplication_and_incompatible_candidate_hash(tmp_path):
    suite, protocol = fixtures()
    with pytest.raises(EvaluationError, match="identity mismatch"):
        evaluate(suite, protocol, candidate_sha256="INVALID", predict=lambda p, s: "yes")
    path = tmp_path / "duplicates.json"
    path.write_text('{"status":"PASS","status":"PASS"}')
    with pytest.raises(EvaluationError, match="corrupted"):
        read_report(path)
