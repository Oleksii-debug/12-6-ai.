"""Fail-closed bridge between DATA-294 exposure ledgers and Trainer microbatches."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from torch import Tensor

from twelve_six.data.unique_loss_ledger import (
    ExposureAccountingError,
    ExposureBudgetGuard,
)
from twelve_six.training.trainer import Batch, StepMetrics, Trainer


def count_batch_optimized_targets(batch: Batch, *, ignore_index: int = -100) -> int:
    """Count the exact targets Trainer will count before it executes the microbatch."""
    if "input_ids" not in batch:
        raise KeyError("batch must contain input_ids")
    if "labels" in batch and "target_ids" in batch:
        raise ValueError("batch must not contain both labels and target_ids")
    aligned = "target_ids" in batch
    targets: Tensor = batch.get(
        "target_ids",
        batch.get("labels", batch["input_ids"]),
    )
    loss_mask = batch.get("loss_mask")
    if aligned:
        valid = targets.ne(ignore_index)
        if loss_mask is not None:
            valid = valid & loss_mask.bool()
        count = int(valid.sum().item())
    else:
        if loss_mask is not None:
            raise ValueError("loss_mask is valid only with aligned target_ids")
        count = int(targets[:, 1:].ne(ignore_index).sum().item())
    if count <= 0:
        raise ExposureAccountingError("microbatch has no optimized causal targets")
    return count


def train_microbatch_with_exposure(
    trainer: Trainer,
    guard: ExposureBudgetGuard,
    batch: Batch,
    claims: Iterable[Mapping[str, Any]],
) -> StepMetrics:
    """Reserve exact unique positions, then execute one Trainer microbatch.

    Reservation happens before forward/backward. If training fails, the positions stay
    consumed: retrying them would be an untracked replay. Recovery must restore a
    checkpoint containing both Trainer state and the matching exposure-guard state.
    """
    target_count = count_batch_optimized_targets(batch)
    guard.authorize_batch(claims, actual_optimized_targets=target_count)
    metrics = trainer.train_microbatch(batch)
    if metrics.tokens != target_count:
        raise ExposureAccountingError(
            "Trainer optimized-target count drifted after exposure authorization"
        )
    if trainer.tokens_seen != guard.consumed_targets:
        raise ExposureAccountingError(
            "Trainer tokens_seen diverged from DATA-294 consumed unique targets"
        )
    return metrics


def assert_guarded_checkpoint_safe(trainer: Trainer, guard: ExposureBudgetGuard) -> None:
    """Require a checkpoint boundary with synchronized Trainer/exposure counters."""
    trainer.assert_checkpoint_safe()
    if trainer.tokens_seen != guard.consumed_targets:
        raise ExposureAccountingError(
            "checkpoint would separate Trainer state from exposure accounting state"
        )


def restore_guarded_exposure_state(
    trainer: Trainer,
    guard: ExposureBudgetGuard,
    state: Mapping[str, Any],
) -> None:
    """Restore exposure state only after the matching Trainer checkpoint was loaded."""
    guard.load_state_dict(state)
    if trainer.tokens_seen != guard.consumed_targets:
        raise ExposureAccountingError(
            "restored Trainer token counter does not match exposure state"
        )
