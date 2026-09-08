from __future__ import annotations

import pytest

from twelve_six.checkpoint.core import CheckpointCompatibilityError
from twelve_six.checkpoint.selection import (
    CheckpointCandidate,
    select_best,
    select_chronological,
    select_final,
)


def _manifest(*, checkpoint_id: str, step: int, tokens_seen: int, run: str = "r") -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "identity": {
            "run_manifest_hash": run,
            "model_spec_hash": "m",
            "tokenizer_hash": "t",
            "dataset_manifest_hash": "d",
            "training_config_hash": "c",
            "step": step,
            "tokens_seen": tokens_seen,
        },
    }


def _candidate(
    checkpoint_id: str,
    step: int,
    tokens_seen: int,
    *,
    completed: bool = False,
    metric_name: str | None = None,
    metric_value: float | None = None,
    run: str = "r",
) -> CheckpointCandidate:
    return CheckpointCandidate.from_manifest(
        _manifest(checkpoint_id=checkpoint_id, step=step, tokens_seen=tokens_seen, run=run),
        completed=completed,
        metric_name=metric_name,
        metric_value=metric_value,
    )


def test_chronological_uses_training_progress_not_publication_order() -> None:
    early = _candidate("early", 3, 300)
    late = _candidate("late", 4, 400)
    assert select_chronological([late, early]) is late


def test_chronological_rejects_equal_progress_distinct_artifacts() -> None:
    a = _candidate("a", 4, 400)
    b = _candidate("b", 4, 400)
    with pytest.raises(CheckpointCompatibilityError, match="ambiguous chronological"):
        select_chronological([a, b])


def test_selection_rejects_cross_lineage_candidates() -> None:
    a = _candidate("a", 4, 400, run="run-a")
    b = _candidate("b", 5, 500, run="run-b")
    with pytest.raises(CheckpointCompatibilityError, match="different lineages"):
        select_chronological([a, b])


def test_selection_rejects_duplicate_checkpoint_ids() -> None:
    a = _candidate("same", 4, 400)
    b = _candidate("same", 5, 500)
    with pytest.raises(CheckpointCompatibilityError, match="duplicate checkpoint_id"):
        select_chronological([a, b])


def test_final_requires_explicit_completion_at_latest_progress() -> None:
    final = _candidate("final", 5, 500, completed=True)
    newer = _candidate("newer", 6, 600)
    with pytest.raises(CheckpointCompatibilityError, match="not at the latest"):
        select_final([final, newer])


def test_final_returns_unique_completed_latest_checkpoint() -> None:
    earlier = _candidate("earlier", 4, 400)
    final = _candidate("final", 5, 500, completed=True)
    assert select_final([earlier, final]) is final


def test_best_min_and_max_are_explicit() -> None:
    a = _candidate("a", 4, 400, metric_name="validation_loss", metric_value=1.25)
    b = _candidate("b", 5, 500, metric_name="validation_loss", metric_value=1.10)
    assert select_best([a, b], metric_name="validation_loss", mode="min") is b
    assert select_best([a, b], metric_name="validation_loss", mode="max") is a


def test_best_rejects_missing_or_mixed_metric_authority() -> None:
    a = _candidate("a", 4, 400, metric_name="validation_loss", metric_value=1.25)
    b = _candidate("b", 5, 500)
    with pytest.raises(CheckpointCompatibilityError, match="every candidate"):
        select_best([a, b], metric_name="validation_loss", mode="min")


def test_best_rejects_equal_metric_tie() -> None:
    a = _candidate("a", 4, 400, metric_name="validation_loss", metric_value=1.0)
    b = _candidate("b", 5, 500, metric_name="validation_loss", metric_value=1.0)
    with pytest.raises(CheckpointCompatibilityError, match="ambiguous best"):
        select_best([a, b], metric_name="validation_loss", mode="min")


def test_candidate_rejects_partial_metric_and_non_finite_metric() -> None:
    with pytest.raises(CheckpointCompatibilityError, match="both be present"):
        _candidate("a", 1, 10, metric_name="validation_loss")
    with pytest.raises(CheckpointCompatibilityError, match="finite"):
        _candidate("a", 1, 10, metric_name="validation_loss", metric_value=float("nan"))
