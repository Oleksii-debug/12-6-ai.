"""Checkpointing, integrity, resume, and export primitives for 12-6 AI."""

from . import core as _core
from .durability import install as _install_durable_save
from .progress_binding import install as _install_progress_binding
from .transactional_rng import install as _install_transactional_rng

_install_progress_binding(_core)
_install_transactional_rng(_core)
_install_durable_save(_core)
del _core, _install_durable_save, _install_progress_binding, _install_transactional_rng

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
from .progress_trainer import load_trainer_checkpoint
from .run_binding import bind_checkpoint_identity
from .trainer_adapter import save_trainer_checkpoint

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
