"""Fail-closed checkpoint selection semantics for immutable D05 checkpoints.

Selection is deliberately separate from checkpoint publication.  It consumes
verified manifest facts and external evaluation evidence, never rewrites a
checkpoint directory, and refuses ambiguous or cross-lineage comparisons.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .core import CheckpointCompatibilityError


@dataclass(frozen=True)
class CheckpointCandidate:
    """Selection facts for one already-verified immutable checkpoint."""

    checkpoint_id: str
    run_manifest_hash: str
    model_spec_hash: str
    tokenizer_hash: str
    dataset_manifest_hash: str
    training_config_hash: str
    step: int
    tokens_seen: int
    completed: bool = False
    metric_name: str | None = None
    metric_value: float | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        completed: bool = False,
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> CheckpointCandidate:
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise CheckpointCompatibilityError("checkpoint manifest identity must be a mapping")
        checkpoint_id = manifest.get("checkpoint_id")
        required = {
            "checkpoint_id": checkpoint_id,
            "run_manifest_hash": identity.get("run_manifest_hash"),
            "model_spec_hash": identity.get("model_spec_hash"),
            "tokenizer_hash": identity.get("tokenizer_hash"),
            "dataset_manifest_hash": identity.get("dataset_manifest_hash"),
            "training_config_hash": identity.get("training_config_hash"),
        }
        for field, value in required.items():
            if not isinstance(value, str) or not value:
                raise CheckpointCompatibilityError(f"{field} must be a non-empty string")
        step = identity.get("step")
        tokens_seen = identity.get("tokens_seen")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or not isinstance(tokens_seen, int)
            or isinstance(tokens_seen, bool)
            or tokens_seen < 0
        ):
            raise CheckpointCompatibilityError(
                "checkpoint progress must contain non-negative integer step/tokens_seen"
            )
        if not isinstance(completed, bool):
            raise CheckpointCompatibilityError("completed must be boolean")
        if (metric_name is None) != (metric_value is None):
            raise CheckpointCompatibilityError(
                "metric_name and metric_value must either both be present or both be absent"
            )
        if metric_name is not None:
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise CheckpointCompatibilityError("metric_name must be a non-empty string")
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                raise CheckpointCompatibilityError("metric_value must be numeric")
            if not isfinite(float(metric_value)):
                raise CheckpointCompatibilityError("metric_value must be finite")
        return cls(
            checkpoint_id=checkpoint_id,
            run_manifest_hash=required["run_manifest_hash"],
            model_spec_hash=required["model_spec_hash"],
            tokenizer_hash=required["tokenizer_hash"],
            dataset_manifest_hash=required["dataset_manifest_hash"],
            training_config_hash=required["training_config_hash"],
            step=step,
            tokens_seen=tokens_seen,
            completed=completed,
            metric_name=metric_name,
            metric_value=None if metric_value is None else float(metric_value),
        )

    @property
    def lineage_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.run_manifest_hash,
            self.model_spec_hash,
            self.tokenizer_hash,
            self.dataset_manifest_hash,
            self.training_config_hash,
        )

    @property
    def progress_key(self) -> tuple[int, int]:
        return (self.tokens_seen, self.step)


def _materialize(candidates: Iterable[CheckpointCandidate]) -> list[CheckpointCandidate]:
    items = list(candidates)
    if not items:
        raise CheckpointCompatibilityError("checkpoint selection requires at least one candidate")
    if any(not isinstance(item, CheckpointCandidate) for item in items):
        raise CheckpointCompatibilityError(
            "checkpoint selection accepts only verified CheckpointCandidate records"
        )
    lineage = items[0].lineage_key
    if any(item.lineage_key != lineage for item in items[1:]):
        raise CheckpointCompatibilityError("cannot select checkpoints across different lineages")
    ids = [item.checkpoint_id for item in items]
    if len(set(ids)) != len(ids):
        raise CheckpointCompatibilityError("duplicate checkpoint_id in selection set")
    return items


def _unique_extreme(
    items: list[CheckpointCandidate],
    *,
    key: Any,
    maximize: bool,
    label: str,
) -> CheckpointCandidate:
    values = [key(item) for item in items]
    extreme = max(values) if maximize else min(values)
    winners = [item for item in items if key(item) == extreme]
    if len(winners) != 1:
        ids = sorted(item.checkpoint_id for item in winners)
        raise CheckpointCompatibilityError(
            f"ambiguous {label} checkpoint selection at equal authority: {ids}"
        )
    return winners[0]


def select_chronological(candidates: Iterable[CheckpointCandidate]) -> CheckpointCandidate:
    """Return the unique checkpoint with greatest tokens_seen then step.

    Equal progress with distinct checkpoint identities is corruption/authority
    ambiguity, not a timestamp tie-break.  Wall-clock creation time is excluded
    intentionally because retries can publish later artifacts for older progress.
    """

    items = _materialize(candidates)
    return _unique_extreme(
        items,
        key=lambda item: item.progress_key,
        maximize=True,
        label="chronological",
    )


def select_final(candidates: Iterable[CheckpointCandidate]) -> CheckpointCandidate:
    """Return the unique completed checkpoint at terminal progress.

    A completed checkpoint behind a newer non-completed checkpoint is rejected:
    callers must not silently call an older artifact "final" once later training
    progress exists in the same run lineage.
    """

    items = _materialize(candidates)
    latest = select_chronological(items)
    finals = [item for item in items if item.completed]
    if not finals:
        raise CheckpointCompatibilityError("no checkpoint is explicitly marked completed")
    final = _unique_extreme(
        finals,
        key=lambda item: item.progress_key,
        maximize=True,
        label="final",
    )
    if final.progress_key != latest.progress_key:
        raise CheckpointCompatibilityError(
            "completed checkpoint is not at the latest recorded training progress"
        )
    return final


def select_best(
    candidates: Iterable[CheckpointCandidate],
    *,
    metric_name: str,
    mode: str,
) -> CheckpointCandidate:
    """Return the unique best evaluation-bound checkpoint.

    ``mode`` is exactly ``min`` or ``max``.  Missing metrics, mixed metric names,
    non-finite values, and equal best values across different checkpoint IDs are
    rejected instead of being resolved by arbitrary filesystem/timestamp order.
    """

    items = _materialize(candidates)
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise CheckpointCompatibilityError("metric_name must be a non-empty string")
    if mode not in {"min", "max"}:
        raise CheckpointCompatibilityError("best-checkpoint mode must be exactly 'min' or 'max'")
    eligible = []
    for item in items:
        if item.metric_name != metric_name or item.metric_value is None:
            raise CheckpointCompatibilityError(
                "every candidate must carry the requested best-checkpoint metric"
            )
        if not isfinite(item.metric_value):
            raise CheckpointCompatibilityError("best-checkpoint metric must be finite")
        eligible.append(item)
    return _unique_extreme(
        eligible,
        key=lambda item: item.metric_value,
        maximize=mode == "max",
        label="best",
    )
