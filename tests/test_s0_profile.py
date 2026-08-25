from __future__ import annotations

import copy
from pathlib import Path

import pytest

from twelve_six.checkpoint import detect_git_sha
from twelve_six.s0_profile import (
    AUTHORITY,
    SCHEMA_VERSION,
    run_s0_cpu_profile,
    validate_s0_cpu_profile,
)


@pytest.fixture(scope="module")
def profile_report() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    source_sha = detect_git_sha(root)
    assert source_sha is not None
    return run_s0_cpu_profile(
        root,
        source_sha=source_sha,
        seed=1337,
        training_steps=2,
        repetitions=1,
    )


def test_real_s0_cpu_profile_is_exact_bound_and_fail_closed(
    profile_report: dict[str, object],
) -> None:
    root = Path(__file__).resolve().parents[1]
    source_sha = detect_git_sha(root)
    assert source_sha is not None

    validate_s0_cpu_profile(profile_report, expected_source_sha=source_sha)
    assert profile_report["schema_version"] == SCHEMA_VERSION
    assert profile_report["authority"] == AUTHORITY

    identity = profile_report["identity"]
    assert isinstance(identity, dict)
    assert identity["parameter_count"] == 10_140

    phases = profile_report["phases"]
    assert isinstance(phases, dict)
    assert phases["checkpoint_save"]["work"]["checkpoint_bytes"] > 0
    assert phases["greedy_generation"]["work"]["generated_tokens"] > 0

    full_training = profile_report["full_training"]
    assert isinstance(full_training, dict)
    assert full_training["optimized_tokens"] > 0
    assert full_training["validation_optimized_tokens"] == 0
    assert full_training["tokens_per_second"] > 0

    truth = profile_report["truth_boundary"]
    assert isinstance(truth, dict)
    assert all(value is False for value in truth.values())


def test_profile_validator_rejects_overclaim_and_tamper(
    profile_report: dict[str, object],
) -> None:
    tampered = copy.deepcopy(profile_report)
    truth = tampered["truth_boundary"]
    assert isinstance(truth, dict)
    truth["capacity_or_sla_claim"] = True
    with pytest.raises(ValueError, match="truth boundary"):
        validate_s0_cpu_profile(tampered)


def test_profile_validator_rejects_wrong_exact_head(
    profile_report: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="expected exact head"):
        validate_s0_cpu_profile(profile_report, expected_source_sha="0" * 40)
