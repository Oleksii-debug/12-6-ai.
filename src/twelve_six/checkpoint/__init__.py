"""Checkpointing, integrity, resume, and export primitives for 12-6 AI."""

from .core import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    LoadResult,
    VerifiedCheckpoint,
    assert_identity,
    capture_rng_state,
    detect_git_sha,
    environment_snapshot,
    hash_json,
    restore_rng_state,
    save_checkpoint,
    sha256_file,
)
from .hardening import (
    load_checkpoint,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    preflight_optimizer_state,
    preflight_trainer_state,
    verify_checkpoint,
)
from .hf_export import export_hf_directory
from .run_binding import bind_checkpoint_identity
from .trainer_adapter import load_trainer_checkpoint, save_trainer_checkpoint
from .core_guard import install_core_guards

install_core_guards()

__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointIdentity",
    "CheckpointIntegrityError",
    "LoadResult",
    "VerifiedCheckpoint",
    "assert_identity",
    "bind_checkpoint_identity",
    "capture_rng_state",
    "detect_git_sha",
    "environment_snapshot",
    "export_hf_directory",
    "hash_json",
    "load_checkpoint",
    "load_trainer_checkpoint",
    "load_verified_checkpoint",
    "prepare_checkpoint_load",
    "preflight_optimizer_state",
    "preflight_trainer_state",
    "restore_rng_state",
    "save_checkpoint",
    "save_trainer_checkpoint",
    "sha256_file",
    "verify_checkpoint",
]
