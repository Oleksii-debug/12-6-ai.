from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.scaling_budget import (
    FIXED_COMPUTE_BUDGET,
    FIXED_TOKEN_BUDGET,
    PARAMETER_COUNTS,
    _budget_plan,
    _git_head,
    _run_candidate,
    _token_quantum,
)


def test_fixed_token_budget_is_exact_and_rejects_incumbent_overshoot_request() -> None:
    quantum = _token_quantum(batch_size=4, sequence_length=64)
    assert quantum == 252
    assert FIXED_TOKEN_BUDGET == 65_772
    assert FIXED_TOKEN_BUDGET % quantum == 0
    plan = _budget_plan(
        mode="fixed_tokens",
        budget=FIXED_TOKEN_BUDGET,
        parameters=PARAMETER_COUNTS[0],
        token_quantum=quantum,
    )
    assert plan["optimizer_steps"] == 261
    assert plan["optimized_tokens"] == FIXED_TOKEN_BUDGET
    assert plan["compute_remainder"] == 0
    with pytest.raises(ValueError, match="not exactly representable"):
        _budget_plan(
            mode="fixed_tokens",
            budget=65_536,
            parameters=PARAMETER_COUNTS[0],
            token_quantum=quantum,
        )


def test_fixed_compute_floor_is_maximal_without_exceeding_budget() -> None:
    quantum = _token_quantum(batch_size=4, sequence_length=64)
    expected_tokens = (182_952, 65_268, 37_296, 16_632)
    for parameters, expected in zip(PARAMETER_COUNTS, expected_tokens, strict=True):
        plan = _budget_plan(
            mode="fixed_compute",
            budget=FIXED_COMPUTE_BUDGET,
            parameters=parameters,
            token_quantum=quantum,
        )
        assert plan["optimized_tokens"] == expected
        assert plan["compute_proxy"] <= FIXED_COMPUTE_BUDGET
        assert 0 <= plan["compute_remainder"] < plan["compute_per_update"]
        assert plan["compute_proxy"] + plan["compute_per_update"] > FIXED_COMPUTE_BUDGET


def test_budget_requires_resume_capable_two_update_minimum() -> None:
    quantum = _token_quantum(batch_size=4, sequence_length=64)
    with pytest.raises(ValueError, match="at least two updates"):
        _budget_plan(
            mode="fixed_tokens",
            budget=quantum,
            parameters=PARAMETER_COUNTS[0],
            token_quantum=quantum,
        )


def test_small_candidate_proves_exact_tokens_eval_isolation_and_resume(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_sha = _git_head(repo_root)
    quantum = _token_quantum(batch_size=4, sequence_length=64)
    report = _run_candidate(
        repo_root=repo_root,
        source_sha=source_sha,
        mode="fixed_tokens",
        budget=4 * quantum,
        candidate_index=0,
        checkpoint_root=tmp_path / "checkpoints",
        torch_threads=1,
    )
    assert report["accounting"]["optimized_tokens"] == 4 * quantum
    assert report["accounting"]["optimizer_steps"] == 4
    assert report["accounting"]["evaluation_tokens_counted_as_optimized"] == 0
    assert report["accounting"]["token_drift_detected"] is False
    assert report["checkpoint_resume"]["resume_exercised"] is True
    assert report["checkpoint_resume"]["resume_exact"] is True
    assert report["checkpoint_resume"]["resume_optimizer_step"] == 2
    assert report["generalization"]["validation_tokens"] > 0
    assert report["checkpoint_resume"]["checkpoint_bytes"] > 0
