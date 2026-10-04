"""Trainer resume with exact positive-progress and D04 exposure binding.

This is the D05 convergence wrapper for the trainer-owned single-decode restore
path. It deliberately reuses trainer_adapter preflight helpers rather than
reimplementing D02 optimizer/scheduler/scaler semantics, and consumes D04's
ordered-exposure identities without recreating D04 dataloader semantics.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import core as _core
from .core import (
    LoadResult,
    _apply_model_weights,
    _decode_verified_state,
    _preflight_rng_state,
    _prepare_model_weights,
    assert_identity,
    prepare_checkpoint_load,
    restore_rng_state,
)
from .d04_resume_binding import assert_d04_resume_binding
from .progress_binding import _assert_progress
from .trainer_adapter import (
    _assert_bound_metadata,
    _preflight_trainer_state,
    _preflight_trainer_target,
)

_HEX = frozenset("0123456789abcdef")


def _require_expected_sha256(value: str | None, *, field: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise _core.CheckpointCompatibilityError(
            f"{field} must be an exact lowercase 64-hex SHA-256 or None"
        )


def _require_expected_nonempty_string(value: str | None, *, field: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise _core.CheckpointCompatibilityError(
            f"{field} must be a non-empty string or None"
        )


def _validate_expected_canonical_binding(
    *,
    expected_init_spec_hash: str | None,
    expected_split_identity: str | None,
    expected_packing_hash: str | None,
    expected_packing_version: str | None,
    expected_training_config_hash: str | None,
    expected_environment_lock_hash: str | None,
    expected_seed: int | None,
) -> None:
    """Reject malformed caller provenance before accepting equal malformed metadata."""

    for field, value in (
        ("expected_init_spec_hash", expected_init_spec_hash),
        ("expected_packing_hash", expected_packing_hash),
        ("expected_training_config_hash", expected_training_config_hash),
        ("expected_environment_lock_hash", expected_environment_lock_hash),
    ):
        _require_expected_sha256(value, field=field)
    _require_expected_nonempty_string(
        expected_split_identity,
        field="expected_split_identity",
    )
    _require_expected_nonempty_string(
        expected_packing_version,
        field="expected_packing_version",
    )
    if expected_seed is not None and (
        not isinstance(expected_seed, int)
        or isinstance(expected_seed, bool)
        or expected_seed < 0
    ):
        raise _core.CheckpointCompatibilityError(
            "expected_seed must be a non-negative integer or None"
        )


def load_trainer_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    trainer: Any,
    strict_model: bool = True,
    restore_rng: bool = True,
    expected_checkpoint_id: str | None = None,
    expected_manifest_sha256: str | None = None,
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
    expected_previous_run_id: str | None = None,
    expected_training_config_hash: str | None = None,
    expected_environment_lock_hash: str | None = None,
    expected_seed: int | None = None,
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
    expected_ledger_identity_sha256: str | None = None,
    expected_materialization_identity_sha256: str | None = None,
    expected_packing_identity_sha256: str | None = None,
    expected_exposure_plan_identity_sha256: str | None = None,
    expected_ordered_next_exposure_identity_sha256: str | None = None,
) -> LoadResult:
    """Verify/decode once and reject wrong progress/exposure before mutation."""

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")

    _require_expected_sha256(
        expected_checkpoint_id,
        field="expected_checkpoint_id",
    )
    _require_expected_sha256(
        expected_manifest_sha256,
        field="expected_manifest_sha256",
    )
    _require_expected_nonempty_string(
        expected_previous_run_id,
        field="expected_previous_run_id",
    )
    _validate_expected_canonical_binding(
        expected_init_spec_hash=expected_init_spec_hash,
        expected_split_identity=expected_split_identity,
        expected_packing_hash=expected_packing_hash,
        expected_packing_version=expected_packing_version,
        expected_training_config_hash=expected_training_config_hash,
        expected_environment_lock_hash=expected_environment_lock_hash,
        expected_seed=expected_seed,
    )

    # Refuse a previously poisoned instance before opening or decoding a
    # potentially model-scale checkpoint; post-decode preflight repeats this
    # guard before mutation in case the target state changed meanwhile.
    _preflight_trainer_target(trainer)
    verified = prepare_checkpoint_load(directory)
    manifest = verified.manifest
    if (
        expected_checkpoint_id is not None
        and manifest.get("checkpoint_id") != expected_checkpoint_id
    ):
        raise _core.CheckpointCompatibilityError(
            "checkpoint_id does not match the independently expected D05 identity"
        )
    if (
        expected_manifest_sha256 is not None
        and verified.manifest_sha256 != expected_manifest_sha256
    ):
        raise _core.CheckpointCompatibilityError(
            "manifest SHA-256 does not match the independently expected D05 identity"
        )
    if expected_previous_run_id is not None:
        identity = manifest.get("identity")
        training_config = identity.get("training_config") if isinstance(identity, dict) else None
        if (
            not isinstance(training_config, dict)
            or training_config.get("run_id") != expected_previous_run_id
        ):
            raise _core.CheckpointCompatibilityError(
                "checkpoint previous run id does not match the independently expected identity"
            )
    _assert_progress(
        _core,
        manifest,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )
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
    assert_d04_resume_binding(
        manifest,
        expected_ledger_identity_sha256=expected_ledger_identity_sha256,
        expected_materialization_identity_sha256=(
            expected_materialization_identity_sha256
        ),
        expected_packing_identity_sha256=expected_packing_identity_sha256,
        expected_exposure_plan_identity_sha256=(
            expected_exposure_plan_identity_sha256
        ),
        expected_ordered_next_exposure_identity_sha256=(
            expected_ordered_next_exposure_identity_sha256
        ),
    )
    assert_identity(
        manifest,
        git_sha=expected_git_sha,
        model_spec_hash=expected_model_spec_hash,
        tokenizer_hash=expected_tokenizer_hash,
        tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        dataset_manifest_hash=expected_dataset_manifest_hash,
        run_manifest_hash=expected_run_manifest_hash,
    )

    arrays, combined_state = _decode_verified_state(verified)
    del verified
    trainer_state = combined_state.get("trainer")
    _preflight_trainer_state(trainer, trainer_state, manifest=manifest)
    materialized = _prepare_model_weights(model, arrays, strict_model)
    if restore_rng:
        _preflight_rng_state(combined_state["rng"])
    del arrays

    # Preflight prevents known incompatibilities, but an application-time
    # model/RNG/optimizer failure can leave a mixed, non-replayable state.
    # Canonical D02 trainers must then refuse any further optimizer step or
    # in-place retry; avoid copying model-scale weights to attempt rollback.
    try:
        _apply_model_weights(model, materialized, strict_model)
        trainer.load_state_dict(trainer_state)
        # Trainer/optimizer/scheduler loaders may consume Python, NumPy or
        # torch RNG even on success. Restore the checkpoint streams last so
        # the first resumed batch sees the exact captured next draws.
        if restore_rng:
            restore_rng_state(combined_state["rng"])
    except BaseException:
        if hasattr(trainer, "_failure_reason") and hasattr(trainer, "_update_incomplete"):
            # D02 may already have recorded a more specific partial-load error
            # (including a second gradient-cleanup failure). Preserve it.
            if trainer._failure_reason is None:
                trainer._failure_reason = "checkpoint_restore_apply_failed"
            trainer._update_incomplete = True
        raise
    return LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=trainer_state,
        rng_state=combined_state["rng"],
    )
