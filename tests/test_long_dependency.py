from __future__ import annotations

import copy

from twelve_six.long_dependency import (
    DEFAULT_DISTANCES,
    SHORT_CONTROL_DISTANCE,
    materialize_suite,
    score_suite,
    suite_identity_sha256,
    validate_report,
)


class CopyAwareBackend:
    eos_token_id = None

    def __init__(self, max_context_tokens: int):
        self.max_context_tokens = max_context_tokens
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("ascii"))

    def decode(self, token_ids):
        return bytes(token_ids).decode("ascii")

    def next_token_logits(self, input_ids):
        self.calls += 1
        logits = [0.0] * 256
        source = int(input_ids[0])
        logits[source] = 5.0
        return logits


class LastTokenBackend(CopyAwareBackend):
    def next_token_logits(self, input_ids):
        self.calls += 1
        logits = [0.0] * 256
        logits[int(input_ids[-1])] = 5.0
        return logits


def test_suite_identity_and_exact_distances_are_deterministic():
    backend = CopyAwareBackend(512)
    left = materialize_suite(backend, cases_per_family_distance=4)
    right = materialize_suite(backend, cases_per_family_distance=4)
    assert left.suite_identity_sha256 == right.suite_identity_sha256
    assert left.materialized_identity_sha256 == right.materialized_identity_sha256
    assert left.suite_identity_sha256 == suite_identity_sha256(cases_per_family_distance=4)
    assert left.metadata["training_allowed"] is False
    assert left.metadata["instruction_following"] is False
    assert {case.dependency_distance for case in left.cases} == {
        SHORT_CONTROL_DISTANCE,
        *DEFAULT_DISTANCES,
    }
    for case in left.cases:
        assert len(case.prefix_ids) == case.dependency_distance
        assert case.target_index - case.source_index == case.dependency_distance
        assert case.truncated_prefix_ids[0] != case.prefix_ids[0] or len(
            case.truncated_prefix_ids
        ) < len(case.prefix_ids)


def test_scoring_never_extrapolates_and_controls_destroy_copy_signal():
    backend = CopyAwareBackend(128)
    suite = materialize_suite(backend, cases_per_family_distance=4)
    report = score_suite(backend, suite, model_label="copy-aware-128")
    validate_report(report)
    assert report["evaluation"]["extrapolation_attempted"] is False
    assert report["evaluation"]["unsupported_case_counts_by_distance"] == {
        "256": 12,
        "512": 12,
    }
    rows = {
        row["dependency_distance"]: row
        for row in report["by_distance"]
        if row["role"] == "long_dependency"
    }
    assert set(rows) == {32, 64, 128}
    for row in rows.values():
        assert row["conditions"]["full"]["pairwise_accuracy"] == 1.0
        assert row["conditions"]["shuffled"]["pairwise_accuracy"] == 0.0
        assert row["full_vs_shuffled"]["pair_margin_gain_nats"] > 0.0


def test_short_control_prevents_false_long_context_claim():
    backend = LastTokenBackend(512)
    suite = materialize_suite(backend, cases_per_family_distance=4)
    report = score_suite(backend, suite, model_label="local-only")
    assert report["interpretation"]["status"] == "probe_format_not_resolved_at_short_control"
    assert report["interpretation"]["usable_long_dependency_claim"] is False


def test_scoring_does_not_mutate_backend_configuration():
    backend = CopyAwareBackend(64)
    before = copy.deepcopy(backend.__dict__)
    suite = materialize_suite(backend, cases_per_family_distance=2)
    stable_before_score = {
        key: value for key, value in backend.__dict__.items() if key != "calls"
    }
    score_suite(backend, suite, model_label="copy-aware-64")
    stable_after_score = {
        key: value for key, value in backend.__dict__.items() if key != "calls"
    }
    assert stable_before_score == stable_after_score
    assert before["max_context_tokens"] == backend.max_context_tokens
