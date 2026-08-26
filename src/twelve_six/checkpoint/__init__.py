"""Checkpointing, integrity, resume, and export primitives for 12-6 AI."""

from . import core as _core
from .hardening import install_checkpoint_hardening

# D05 hardening is installed before downstream checkpoint adapters import core
# symbols, so direct ``twelve_six.checkpoint.core`` and package-level callers
# receive the same fail-closed loader behavior.
install_checkpoint_hardening(_core)

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
    load_checkpoint,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    restore_rng_state,
    save_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from .hf_export import export_hf_directory
from .run_binding import bind_checkpoint_identity
from .trainer_adapter import load_trainer_checkpoint, save_trainer_checkpoint

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
    "restore_rng_state",
    "save_checkpoint",
    "save_trainer_checkpoint",
    "sha256_file",
    "verify_checkpoint",
]
