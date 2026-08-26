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
    _preflight_optimizer_state,
    _preflight_stateful_target,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    save_checkpoint,
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


def _config_as_mapping(config: Any) -> Mapping[str, Any] | None:
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        value = config.model_dump(mode="python")
        return value if isinstance(value, Mapping) else None
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        return value if isinstance(value, Mapping) else None
    return None


def _require_counter(state: Mapping[str, Any], field: str) -> int:
    value = state.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CheckpointCompatibilityError(
            f"trainer {field} must be a non-negative integer"
        )
    return value


def _preflight_trainer_state(
    trainer: Any,
    state: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate trainer-owned resume payload before any live model mutation.

    The public Trainer loader owns optimizer/scheduler/scaler/counter semantics,
    but calling it only after model restoration is not transactional: a corrupt
    nested optimizer slot could otherwise reject after model weights changed.
    This preflight mirrors the loader's fail-closed invariants without mutating
    the live trainer.
    """

    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint trainer state must be a mapping")

    required = {
        "micro_step",
        "optimizer_step",
        "tokens_seen",
        "optimizer",
        "scheduler",
        "scaler",
        "config",
    }
    if set(state) != required:
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - required)
        raise CheckpointCompatibilityError(
            f"trainer state keys differ: missing={missing}, unexpected={unexpected}"
        )

    current_config = _config_as_mapping(getattr(trainer, "config", None))
    saved_config = state.get("config")
    if current_config is None:
        raise CheckpointCompatibilityError("trainer config is not serializable for preflight")
    if not isinstance(saved_config, Mapping) or dict(saved_config) != dict(current_config):
        raise CheckpointCompatibilityError("trainer config mismatch; refusing unsafe resume")

    micro_step = _require_counter(state, "micro_step")
    optimizer_step = _require_counter(state, "optimizer_step")
    tokens_seen = _require_counter(state, "tokens_seen")

    accumulation = current_config.get("gradient_accumulation_steps")
    if not isinstance(accumulation, int) or isinstance(accumulation, bool) or accumulation <= 0:
        raise CheckpointCompatibilityError(
            "trainer gradient_accumulation_steps must be a positive integer"
        )
    expected_micro_steps = optimizer_step * accumulation
    if micro_step != expected_micro_steps:
        raise CheckpointCompatibilityError(
            "checkpoint is not at a complete committed accumulation boundary: "
            f"micro_step={micro_step}, expected={expected_micro_steps}"
        )

    max_steps = current_config.get("max_steps")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise CheckpointCompatibilityError("trainer max_steps must be a positive integer")
    if optimizer_step > max_steps:
        raise CheckpointCompatibilityError(
            "checkpoint optimizer_step exceeds configured max_steps"
        )

    if manifest is not None:
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise CheckpointCompatibilityError("verified checkpoint identity is missing")
        if optimizer_step != identity.get("step"):
            raise CheckpointCompatibilityError(
                "trainer optimizer_step disagrees with checkpoint identity.step"
            )
        if tokens_seen != identity.get("tokens_seen"):
            raise CheckpointCompatibilityError(
                "trainer tokens_seen disagrees with checkpoint identity.tokens_seen"
            )

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        raise CheckpointCompatibilityError("trainer has no optimizer to restore")
    _preflight_optimizer_state(optimizer, state.get("optimizer"))

    scheduler = getattr(trainer, "scheduler", None)
    saved_scheduler = state.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise CheckpointCompatibilityError("scheduler state/config mismatch")
    if scheduler is not None:
        _preflight_stateful_target(scheduler, saved_scheduler, label="scheduler")

    scaler = getattr(trainer, "scaler", None)
    saved_scaler = state.get("scaler")
    if (scaler is None) != (saved_scaler is None):
        raise CheckpointCompatibilityError("scaler state/config mismatch")
    if scaler is not None:
        _preflight_stateful_target(scaler, saved_scaler, label="scaler")


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

    Canonical nested identity checks, trainer-owned optimizer/scheduler/scaler
    preflight, and the actual load all consume the same immutable verified byte
    snapshot. The source directory is never re-opened between verification and
    model mutation.
    """

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")

    verified = prepare_checkpoint_load(directory)
    manifest = verified.manifest
    _assert_bound_metadata(
        manifest,
        expected_init_spec_hash=expected_init_spec_hash,
        expected_split_identity=expected_split_identity,
        expected_packing_hash=expected_packing_hash,
        expected_packing_version=expected_packing_version,
        expected_training_config_hash=expected_training_config_hash,
        expected_environment_lock_hash=expected_environment_lock_hash,
        expected_seed=expected_seed,
    )

    # Decode the immutable in-memory snapshot once for trainer-owned preflight.
    # load_verified_checkpoint decodes the same retained bytes again before the
    # commit point; it never re-opens the source directory.
    _, combined_state = _decode_verified_state(verified)
    _preflight_trainer_state(
        trainer,
        combined_state.get("trainer"),
        manifest=manifest,
    )

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
