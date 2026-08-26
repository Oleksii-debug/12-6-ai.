"""Checkpointing, integrity, resume, and export primitives for 12-6 AI."""

from . import core as _core
from . import trainer_adapter as _trainer_adapter
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
from .hf_export import export_hf_directory
from .run_binding import bind_checkpoint_identity

# Install fail-closed D05 compatibility checks, then make every package-level
# and trainer-adapter loader reference the same hardened implementation.
install_checkpoint_hardening()
load_checkpoint = _core.load_checkpoint
load_verified_checkpoint = _core.load_verified_checkpoint
_trainer_adapter.load_verified_checkpoint = _core.load_verified_checkpoint
load_trainer_checkpoint = _trainer_adapter.load_trainer_checkpoint
save_trainer_checkpoint = _trainer_adapter.save_trainer_checkpoint

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
