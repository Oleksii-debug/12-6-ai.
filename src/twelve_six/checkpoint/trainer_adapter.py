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
    _decode_verified_state,
    save_checkpoint,
)
from .hardening import (
    load_verified_checkpoint,
    preflight_trainer_state,
    prepare_checkpoint_load,
)


def _trainer_state_as_mapping(state: Any) -> Mapping[str, Any]:
    if is_dataclass(state) and not isinstance(state, type):
        return asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError(
        "trainer.state_dict() must return a dataclass instance or mapping "
        "for data-only serialization"
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

    expectations = (
        expected_init_spec_hash,
        expected_split_identity,
        expected_packing_hash,
        expected_packing_version,
        expected_training_config_hash,
        expected_environment_lock_hash,
        expected_seed,
    )
    if all(value is None for value in expectations):
        return

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint identity is missing")
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint training_config is missing")

    need_data = any(
        value is not None
        for value in (expected_split_identity, expected_packing_hash, expected_packing_version)
    )
    data = training_config.get("data")
    if need_data and not isinstance(data, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint bound data identity is missing")
    if not isinstance(data, Mapping):
        data = {}

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


def _preflight_trainer_semantics(trainer: Any, trainer_state: Mapping[str, Any]) -> None:
    """Mirror D02 resume invariants before any live model state is changed."""

    preflight_trainer_state(trainer, trainer_state)
    target_config = getattr(trainer, "config", None)
    if target_config is None or not is_dataclass(target_config):
        raise CheckpointCompatibilityError("trainer must expose a dataclass config for safe resume")
    expected_config = asdict(target_config)
    if trainer_state.get("config") != expected_config:
        raise CheckpointCompatibilityError("trainer config mismatch; refusing unsafe resume")

    micro_step = trainer_state["micro_step"]
    optimizer_step = trainer_state["optimizer_step"]
    accumulation = expected_config.get("gradient_accumulation_steps")
    max_steps = expected_config.get("max_steps")
    if not isinstance(accumulation, int) or isinstance(accumulation, bool) or accumulation <= 0:
        raise CheckpointCompatibilityError("trainer gradient_accumulation_steps is invalid")
    if micro_step != optimizer_step * accumulation:
        raise CheckpointCompatibilityError(
            "checkpoint is not at a complete committed accumulation boundary: "
            f"micro_step={micro_step}, expected={optimizer_step * accumulation}"
        )
    if isinstance(max_steps, int) and not isinstance(max_steps, bool) and optimizer_step > max_steps:
        raise CheckpointCompatibilityError("checkpoint optimizer_step exceeds configured max_steps")

    checkpoint_scheduler = trainer_state.get("scheduler")
    live_scheduler = getattr(trainer, "scheduler", None)
    if (checkpoint_scheduler is None) != (live_scheduler is None):
        raise CheckpointCompatibilityError("scheduler state/config mismatch")


def save_trainer_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    trainer: Any,
    identity: CheckpointIdentity,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save model + trainer-owned optimizer/scheduler/scaler/counter state."""

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
    """Verify one exact byte snapshot, bind it, then restore fresh D02 targets.

    Canonical identity, model tensor metadata, trainer counters, and optimizer
    tensor metadata are checked against the same immutable byte snapshot before
    the first live model/trainer mutation.
    """

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")

    verified = prepare_checkpoint_load(directory)
    _assert_bound_metadata(
        verified.manifest,
        expected_init_spec_hash=expected_init_spec_hash,
        expected_split_identity=expected_split_identity,
        expected_packing_hash=expected_packing_hash,
        expected_packing_version=expected_packing_version,
        expected_training_config_hash=expected_training_config_hash,
        expected_environment_lock_hash=expected_environment_lock_hash,
        expected_seed=expected_seed,
    )

    _, combined_state = _decode_verified_state(verified)
    trainer_state = combined_state.get("trainer")
    if not isinstance(trainer_state, Mapping):
        raise CheckpointCompatibilityError("checkpoint trainer state must be a mapping")
    _preflight_trainer_semantics(trainer, trainer_state)

    result = load_verified_checkpoint(
        verified,
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
