"""Deterministic, fail-closed retention planning for immutable checkpoints.

This module never deletes checkpoint artifacts.  It produces a plan only after
selection authority has been validated, so a caller cannot prune through an
ambiguous lineage/progress set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import CheckpointCompatibilityError
from .selection import (
    CheckpointCandidate,
    _materialize,
    select_best,
    select_chronological,
    select_final,
)


@dataclass(frozen=True)
class CheckpointRetentionPlan:
    """Immutable checkpoint IDs to retain and those eligible for deletion."""

    keep: tuple[str, ...]
    delete: tuple[str, ...]
    chronological: str
    final: str | None
    best: str | None


def plan_checkpoint_retention(
    candidates: list[CheckpointCandidate] | tuple[CheckpointCandidate, ...],
    *,
    keep_recent: int,
    best_metric_name: str | None = None,
    best_mode: str | None = None,
) -> CheckpointRetentionPlan:
    """Plan retention without mutating storage.

    The latest checkpoint is always kept.  The newest ``keep_recent`` progress
    points are kept as rollback windows.  If any checkpoint claims completion,
    selection must resolve one valid terminal ``final`` checkpoint and it is
    kept.  When best-metric authority is supplied, the unique best checkpoint
    is also kept.  Every ambiguity inherited from checkpoint selection fails
    closed before a deletion candidate is emitted.
    """

    if not isinstance(keep_recent, int) or isinstance(keep_recent, bool) or keep_recent < 1:
        raise CheckpointCompatibilityError("keep_recent must be a positive integer")
    if (best_metric_name is None) != (best_mode is None):
        raise CheckpointCompatibilityError(
            "best_metric_name and best_mode must either both be present or both be absent"
        )

    items = _materialize(candidates)
    chronological = select_chronological(items)

    final: CheckpointCandidate | None = None
    if any(item.completed for item in items):
        final = select_final(items)

    best: CheckpointCandidate | None = None
    if best_metric_name is not None and best_mode is not None:
        best = select_best(items, metric_name=best_metric_name, mode=best_mode)

    newest_first = sorted(items, key=lambda item: item.progress_key, reverse=True)
    keep_ids = {item.checkpoint_id for item in newest_first[:keep_recent]}
    keep_ids.add(chronological.checkpoint_id)
    if final is not None:
        keep_ids.add(final.checkpoint_id)
    if best is not None:
        keep_ids.add(best.checkpoint_id)

    # Order both outputs chronologically so plans are stable and reviewable.
    oldest_first = sorted(items, key=lambda item: item.progress_key)
    keep = tuple(item.checkpoint_id for item in oldest_first if item.checkpoint_id in keep_ids)
    delete = tuple(item.checkpoint_id for item in oldest_first if item.checkpoint_id not in keep_ids)

    return CheckpointRetentionPlan(
        keep=keep,
        delete=delete,
        chronological=chronological.checkpoint_id,
        final=None if final is None else final.checkpoint_id,
        best=None if best is None else best.checkpoint_id,
    )
