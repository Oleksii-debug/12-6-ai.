"""Fail-closed checkpoint selection semantics for immutable D05 checkpoints.

Selection is deliberately separate from checkpoint publication. It consumes
verified manifest facts and external evaluation evidence, never rewrites a
checkpoint directory, and refuses ambiguous or cross-lineage comparisons.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .core import CheckpointCompatibilityError

_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: Any, *, field: str) -> str:
    """Require a canonical lowercase SHA-256 identity at the selection boundary."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise CheckpointCompatibilityError(f"{field} must be exact lowercase 64-hex")
    return value


def _require_finite_metric_value(value: Any) -> float:
    """Normalize metric evidence without leaking numeric conversion failures."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointCompatibilityError("metric_value must be numeric")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise CheckpointCompatibilityError("metric_value must be finite") from None
    if not isfinite(normalized):
        raise CheckpointCompatibilityError("metric_value must be finite")
    return normalized


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
        candidate = cls(
            checkpoint_id=manifest.get("checkpoint_id"),
            run_manifest_hash=identity.get("run_manifest_hash"),
            model_spec_hash=identity.get("model_spec_hash"),
            tokenizer_hash=identity.get("tokenizer_hash"),
            dataset_manifest_hash=identity.get("dataset_manifest_hash"),
            training_config_hash=identity.get("training_config_hash"),
            step=identity.get("step"),
            tokens_seen=identity.get("tokens_seen"),
            completed=completed,
            metric_name=metric_name,
            metric_value=metric_value,
        )
        _validate_candidate(candidate)
        return cls(
            checkpoint_id=candidate.checkpoint_id,
            run_manifest_hash=candidate.run_manifest_hash,
            model_spec_hash=candidate.model_spec_hash,
            tokenizer_hash=candidate.tokenizer_hash,
            dataset_manifest_hash=candidate.dataset_manifest_hash,
            training_config_hash=candidate.training_config_hash,
            step=candidate.step,
            tokens_seen=candidate.tokens_seen,
            completed=candidate.completed,
            metric_name=candidate.metric_name,
            metric_value=(
                None
                if candidate.metric_value is None
                else _require_finite_metric_value(candidate.metric_value)
            ),
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


def _validate_candidate(item: CheckpointCandidate) -> None:
    """Revalidate candidates so direct dataclass construction cannot bypass authority checks."""

    _require_sha256(item.checkpoint_id, field="checkpoint_id")
    _require_sha256(item.run_manifest_hash, field="run_manifest_hash")
    _require_sha256(item.model_spec_hash, field="model_spec_hash")
    _require_sha256(item.tokenizer_hash, field="tokenizer_hash")
    _require_sha256(item.dataset_manifest_hash, field="dataset_manifest_hash")
    _require_sha256(item.training_config_hash, field="training_config_hash")
    if (
        not isinstance(item.step, int)
        or isinstance(item.step, bool)
        or item.step < 0
        or not isinstance(item.tokens_seen, int)
        or isinstance(item.tokens_seen, bool)
        or item.tokens_seen < 0
    ):
        raise CheckpointCompatibilityError(
            "checkpoint progress must contain non-negative integer step/tokens_seen"
        )
    if not isinstance(item.completed, bool):
        raise CheckpointCompatibilityError("completed must be boolean")
    if (item.metric_name is None) != (item.metric_value is None):
        raise CheckpointCompatibilityError(
            "metric_name and metric_value must either both be present or both be absent"
        )
    if item.metric_name is not None:
        if not isinstance(item.metric_name, str) or not item.metric_name.strip():
            raise CheckpointCompatibilityError("metric_name must be a non-empty string")
        _require_finite_metric_value(item.metric_value)


def _materialize(candidates: Iterable[CheckpointCandidate]) -> list[CheckpointCandidate]:
    items = list(candidates)
    if not items:
        raise CheckpointCompatibilityError("checkpoint selection requires at least one candidate")
    if any(not isinstance(item, CheckpointCandidate) for item in items):
        raise CheckpointCompatibilityError(
            "checkpoint selection accepts only verified CheckpointCandidate records"
        )
    for item in items:
        _validate_candidate(item)
    lineage = items[0].lineage_key
    if any(item.lineage_key != lineage for item in items[1:]):
        raise CheckpointCompatibilityError("cannot select checkpoints across different lineages")
    ids = [item.checkpoint_id for item in items]
    if len(set(ids)) != len(ids):
        raise CheckpointCompatibilityError("duplicate checkpoint_id in selection set")

    progress_owners: dict[tuple[int, int], str] = {}
    for item in items:
        prior_id = progress_owners.setdefault(item.progress_key, item.checkpoint_id)
        if prior_id != item.checkpoint_id:
            raise CheckpointCompatibilityError(
                "ambiguous checkpoint selection at equal training progress: "
                f"{sorted((prior_id, item.checkpoint_id))}"
            )

    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left.progress_key == right.progress_key:
                continue
            step_order = (left.step > right.step) - (left.step < right.step)
            token_order = (left.tokens_seen > right.tokens_seen) - (
                left.tokens_seen < right.tokens_seen
            )
            if step_order == 0 or token_order == 0 or step_order != token_order:
                raise CheckpointCompatibilityError(
                    "inconsistent checkpoint progress counters: step and tokens_seen must "
                    "advance together within one lineage"
                )
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
    """Return the unique checkpoint with greatest tokens_seen then step."""

    items = _materialize(candidates)
    return _unique_extreme(
        items,
        key=lambda item: item.progress_key,
        maximize=True,
        label="chronological",
    )


def select_final(candidates: Iterable[CheckpointCandidate]) -> CheckpointCandidate:
    """Return the unique completed checkpoint at terminal progress."""

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
    """Return the unique best evaluation-bound checkpoint."""

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
        _require_finite_metric_value(item.metric_value)
        eligible.append(item)
    return _unique_extreme(
        eligible,
        key=lambda item: item.metric_value,
        maximize=mode == "max",
        label="best",
    )
