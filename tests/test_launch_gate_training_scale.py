from __future__ import annotations

import pytest

from twelve_six import launch_gate


def _capability_request(*, require_external: bool = False) -> dict[str, object]:
    return {
        "training_intent": "capability",
        "parameter_count": 100,
        "training_scale": {
            "minimum_unique_train_tokens_per_parameter": 20,
            "minimum_optimized_tokens_per_parameter": 25,
            "require_external_training_sources": require_external,
        },
    }


def test_training_intent_is_mandatory() -> None:
    with pytest.raises(
        launch_gate.LaunchGateError,
        match="training_intent must be explicitly mechanics or capability",
    ):
        launch_gate._verify_training_scale({}, {}, {})


def test_mechanics_intent_preserves_small_scale_probes() -> None:
    result = launch_gate._verify_training_scale(
        {"training_intent": "mechanics"},
        {},
        {"optimizer_steps": 1},
    )
    assert result == {
        "intent": "mechanics",
        "capability_scale_checks": False,
    }


def test_capability_intent_requires_scale_policy() -> None:
    with pytest.raises(
        launch_gate.LaunchGateError,
        match="capability training requires training_scale policy",
    ):
        launch_gate._verify_training_scale(
            {"training_intent": "capability", "parameter_count": 100},
            {"train_byte_tokens": 10_000},
            {"target_optimized_tokens": 10_000},
        )


def test_capability_gate_accepts_sufficient_data_and_compute_budget() -> None:
    result = launch_gate._verify_training_scale(
        _capability_request(require_external=True),
        {
            "train_byte_tokens": 2_500,
            "external_training_eligible_sources": 1,
        },
        {"target_optimized_tokens": 2_500},
    )
    assert result["intent"] == "capability"
    assert result["required_unique_train_tokens"] == 2_000
    assert result["required_optimized_tokens"] == 2_500
    assert result["available_unique_train_tokens"] == 2_500
    assert result["target_optimized_tokens"] == 2_500
    assert result["external_training_eligible_sources"] == 1


def test_capability_gate_rejects_insufficient_unique_corpus() -> None:
    with pytest.raises(
        launch_gate.LaunchGateError,
        match="capability corpus below unique token floor: 1999 < 2000",
    ):
        launch_gate._verify_training_scale(
            _capability_request(),
            {"train_byte_tokens": 1_999},
            {"target_optimized_tokens": 2_500},
        )


def test_capability_gate_rejects_insufficient_optimized_token_budget() -> None:
    with pytest.raises(
        launch_gate.LaunchGateError,
        match="capability optimized-token budget below floor: 2499 < 2500",
    ):
        launch_gate._verify_training_scale(
            _capability_request(),
            {"train_byte_tokens": 2_500},
            {"target_optimized_tokens": 2_499},
        )


def test_capability_gate_can_require_external_training_sources() -> None:
    with pytest.raises(
        launch_gate.LaunchGateError,
        match="at least one eligible external training source",
    ):
        launch_gate._verify_training_scale(
            _capability_request(require_external=True),
            {
                "train_byte_tokens": 2_500,
                "external_training_eligible_sources": 0,
            },
            {"target_optimized_tokens": 2_500},
        )
