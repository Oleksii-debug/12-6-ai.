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

from .core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    LoadResult,
    load_checkpoint,
    verify_checkpoint,
)


def _trainer_state_as_mapping(state: Any) -> Mapping[str, Any]:
    if is_dataclass(state) and not isinstance(state, type):
        return asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError(
        "trainer.state_dict() must return a dataclass instance or mapping for data-only serialization"
    )


def _assert_bound_metadata(
    manifest: Mapping[str, Any],
    *,
    expected_init_spec_hash: str | None,
    expected_split_identity: str | None,
    expected_packing_hash: str | None,
    expected_packing_version: str | None,
    expected_training_config_hash: str | None,
    expected_environment_lock_hash: str | None,
    expected_seed: int | None,
) -> None:
    """Check canonical run-binding fields after verification and before mutation."""

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint identity is missing")
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint training_config is missing")
    data = training_config.get("data")
    if not isinstance(data, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint bound data identity is missing")

    checks = {
        "init_spec_hash": (expected_init_spec_hash, training_config.get("init_spec_sha256")),
        "split_identity": (expected_split_identity, data.get("split_identity")),
        "packing_hash": (expected_packing_hash, data.get("packing_sha256")),
        "packing_version": (expected_packing_version, data.get("packing_version")),
        "training_config_hash": (
            expected_training_config_hash,
            identity.get("training_config_hash"),
        ),
        "environment_lock_hash": (
            expected_environment_lock_hash,
            identity.get("environment_lock_hash"),
        ),
        "seed": (expected_seed, identity.get("seed")),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected is not None and expected != actual
    }
    if mismatches:
        raise CheckpointCompatibilityError(f"checkpoint canonical binding mismatch: {mismatches}")


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
    expected_init_spec_hash: str | None = None,
    expected_tokenizer_hash: str | None = None,
    expected_tokenizer_vocab_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
    expected_split_identity: str | None = None,
    expected_packing_hash: str | None = None,
    expected_packing_version: str | None = None,
    expected_run_manifest_hash: str | None = None,
    expected_training_config_hash: str | None = None,
    expected_environment_lock_hash: str | None = None,
    expected_seed: int | None = None,
) -> LoadResult:
    """Verify the complete canonical identity, then restore a fresh D02 trainer.

    The preflight is intentionally redundant with :func:`load_checkpoint`: the
    first verification checks canonical nested run-binding fields before any target
    object can be mutated, while the second verification closes a possible file
    change between preflight and load. Trainer state is applied only after the
    model checkpoint has passed both checks.
    """

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")

    verified_manifest = verify_checkpoint(directory)
    _assert_bound_metadata(
        verified_manifest,
        expected_init_spec_hash=expected_init_spec_hash,
        expected_split_identity=expected_split_identity,
        expected_packing_hash=expected_packing_hash,
        expected_packing_version=expected_packing_version,
        expected_training_config_hash=expected_training_config_hash,
        expected_environment_lock_hash=expected_environment_lock_hash,
        expected_seed=expected_seed,
    )

    result = load_checkpoint(
        directory,
        model=model,
        strict_model=strict_model,
        restore_rng=restore_rng,
        expected_git_sha=expected_git_sha,
        expected_model_spec_hash=expected_model_spec_hash,
        expected_tokenizer_hash=expected_tokenizer_hash,
        expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
        expected_run_manifest_hash=expected_run_manifest_hash,
    )
    trainer.load_state_dict(result.trainer_state)
    return result
