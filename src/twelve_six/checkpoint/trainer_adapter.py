"""Trainer-owned checkpoint adapter.

D02 owns trainer semantics. D05 only converts a trainer's public state_dict()
contract into the data-only checkpoint format and gives the decoded state back
to trainer.load_state_dict(). This avoids duplicating optimizer/scheduler/scaler
ownership inside the checkpoint API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .core import CheckpointIdentity, LoadResult, load_checkpoint, save_checkpoint


def _trainer_state_as_mapping(state: Any) -> Mapping[str, Any]:
    if is_dataclass(state) and not isinstance(state, type):
        return asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError(
        "trainer.state_dict() must return a dataclass instance or mapping for data-only serialization"
    )


def save_trainer_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    trainer: Any,
    identity: CheckpointIdentity,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save model + trainer-owned optimizer/scheduler/scaler/counter state.

    The trainer must expose state_dict(). Its state is serialized as data only;
    no trainer class/module identity is imported into the checkpoint format.
    """

    if not hasattr(trainer, "state_dict"):
        raise TypeError("trainer must provide state_dict()")
    state = _trainer_state_as_mapping(trainer.state_dict())
    return save_checkpoint(
        directory,
        model=model,
        trainer_state=state,
        identity=identity,
        overwrite=overwrite,
    )


def load_trainer_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    trainer: Any,
    strict_model: bool = True,
    restore_rng: bool = True,
    expected_git_sha: str | None = None,
    expected_model_spec_hash: str | None = None,
    expected_tokenizer_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
) -> LoadResult:
    """Load a verified checkpoint then restore the trainer through its contract."""

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")
    result = load_checkpoint(
        directory,
        model=model,
        strict_model=strict_model,
        restore_rng=restore_rng,
        expected_git_sha=expected_git_sha,
        expected_model_spec_hash=expected_model_spec_hash,
        expected_tokenizer_hash=expected_tokenizer_hash,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
    )
    trainer.load_state_dict(result.trainer_state)
    return result
