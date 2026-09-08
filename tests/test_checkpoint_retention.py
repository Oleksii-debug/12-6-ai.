from __future__ import annotations

import pytest

from twelve_six.checkpoint.core import CheckpointCompatibilityError
from twelve_six.checkpoint.retention import plan_checkpoint_retention
from twelve_six.checkpoint.selection import CheckpointCandidate


def _candidate(
    marker: str,
    *,
    step: int,
    tokens_seen: int,
    completed: bool = False,
    metric: float | None = None,
    lineage_marker: str = "a",
) -> CheckpointCandidate:
    return CheckpointCandidate(
        checkpoint_id=marker * 64,
        run_manifest_hash=lineage_marker * 64,
        model_spec_hash="b" * 64,
        tokenizer_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        training_config_hash="e" * 64,
        step=step,
        tokens_seen=tokens_seen,
        completed=completed,
        metric_name=None if metric is None else "validation_loss",
        metric_value=metric,
    )


def test_retention_keeps_recent_final_and_best() -> None:
    candidates = [
        _candidate("1", step=1, tokens_seen=100, metric=0.9),
        _candidate("2", step=2, tokens_seen=200, metric=0.4),
        _candidate("3", step=3, tokens_seen=300, metric=0.6),
        _candidate("4", step=4, tokens_seen=400, completed=True, metric=0.7),
    ]

    plan = plan_checkpoint_retention(
        candidates,
        keep_recent=2,
        best_metric_name="validation_loss",
        best_mode="min",
    )

    assert plan.chronological == "4" * 64
    assert plan.final == "4" * 64
    assert plan.best == "2" * 64
    assert plan.keep == ("2" * 64, "3" * 64, "4" * 64)
    assert plan.delete == ("1" * 64,)


def test_retention_without_metric_keeps_rollback_window() -> None:
    candidates = [
        _candidate("1", step=1, tokens_seen=100),
        _candidate("2", step=2, tokens_seen=200),
        _candidate("3", step=3, tokens_seen=300),
    ]

    plan = plan_checkpoint_retention(candidates, keep_recent=2)

    assert plan.best is None
    assert plan.final is None
    assert plan.keep == ("2" * 64, "3" * 64)
    assert plan.delete == ("1" * 64,)


@pytest.mark.parametrize("keep_recent", [0, -1, True, 1.5])
def test_retention_rejects_invalid_keep_recent(keep_recent: object) -> None:
    with pytest.raises(CheckpointCompatibilityError, match="keep_recent"):
        plan_checkpoint_retention(
            [_candidate("1", step=1, tokens_seen=100)],
            keep_recent=keep_recent,  # type: ignore[arg-type]
        )


def test_retention_rejects_partial_best_authority() -> None:
    with pytest.raises(CheckpointCompatibilityError, match="best_metric_name and best_mode"):
        plan_checkpoint_retention(
            [_candidate("1", step=1, tokens_seen=100)],
            keep_recent=1,
            best_metric_name="validation_loss",
        )


def test_retention_rejects_stale_completed_checkpoint_before_deletion_plan() -> None:
    candidates = [
        _candidate("1", step=1, tokens_seen=100, completed=True),
        _candidate("2", step=2, tokens_seen=200),
    ]

    with pytest.raises(CheckpointCompatibilityError, match="not at the latest"):
        plan_checkpoint_retention(candidates, keep_recent=1)


def test_retention_rejects_cross_lineage_candidates_before_deletion_plan() -> None:
    candidates = [
        _candidate("1", step=1, tokens_seen=100, lineage_marker="a"),
        _candidate("2", step=2, tokens_seen=200, lineage_marker="f"),
    ]

    with pytest.raises(CheckpointCompatibilityError, match="different lineages"):
        plan_checkpoint_retention(candidates, keep_recent=1)


def test_retention_rejects_ambiguous_best_before_deletion_plan() -> None:
    candidates = [
        _candidate("1", step=1, tokens_seen=100, metric=0.5),
        _candidate("2", step=2, tokens_seen=200, metric=0.5),
    ]

    with pytest.raises(CheckpointCompatibilityError, match="ambiguous best"):
        plan_checkpoint_retention(
            candidates,
            keep_recent=1,
            best_metric_name="validation_loss",
            best_mode="min",
        )
