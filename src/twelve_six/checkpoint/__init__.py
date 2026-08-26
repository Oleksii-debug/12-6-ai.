"""Checkpointing, integrity, resume, and export primitives for 12-6 AI."""

from . import core as _core
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
    prepare_checkpoint_load,
    restore_rng_state,
    save_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from .hardening import install_checkpoint_hardening

# Install D05 corruption guards before trainer_adapter imports the core load
# functions. This makes package-level imports and direct checkpoint.core imports
# converge on the same fail-closed runtime semantics.
install_checkpoint_hardening()
load_checkpoint = _core.load_checkpoint
load_verified_checkpoint = _core.load_verified_checkpoint

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
