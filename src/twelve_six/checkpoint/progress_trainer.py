"""Trainer resume with exact positive-progress binding.

This is the D05 convergence wrapper for the trainer-owned single-decode restore
path. It deliberately reuses trainer_adapter preflight helpers rather than
reimplementing D02 optimizer/scheduler/scaler semantics.
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
from .progress_binding import _assert_progress
from .trainer_adapter import _assert_bound_metadata, _preflight_trainer_state


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
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
) -> LoadResult:
    """Verify/decode once and reject wrong positive progress before mutation."""

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")

    verified = prepare_checkpoint_load(directory)
    manifest = verified.manifest
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

    _apply_model_weights(model, materialized, strict_model)
    if restore_rng:
        restore_rng_state(combined_state["rng"])
    trainer.load_state_dict(trainer_state)
    return LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=trainer_state,
        rng_state=combined_state["rng"],
    )
