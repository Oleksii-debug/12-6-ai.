"""Crash-durable publication helpers for checkpoint-v1.

The checkpoint core already guarantees logical atomic publication and verified
read-before-mutate semantics. This module adds the stronger local-POSIX
persistence protocol needed before a successful save can be described as
crash-durable: sync every verified file, sync the checkpoint directory, rename
inside the destination parent, then sync that parent directory.
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import (
    MANIFEST_CHECKSUM_NAME,
    MANIFEST_NAME,
    STATE_TENSORS_NAME,
    STATE_TREE_NAME,
    WEIGHTS_NAME,
    CheckpointError,
    CheckpointIdentity,
    save_checkpoint,
    verify_checkpoint,
)

_SYNC_FILE_ORDER = (
    WEIGHTS_NAME,
    STATE_TENSORS_NAME,
    STATE_TREE_NAME,
    MANIFEST_NAME,
    MANIFEST_CHECKSUM_NAME,
)


class CheckpointDurabilityError(CheckpointError):
    """Raised when crash-durability cannot be established.

    ``published`` distinguishes a pre-publication failure from the important
    post-rename case: if syncing the destination parent fails, a fully verified
    checkpoint may already be visible and must not be deleted merely because
    durability is uncertain.
    """

    def __init__(self, message: str, *, published: bool) -> None:
        super().__init__(message)
        self.published = published


def _require_local_posix_directory_fsync() -> None:
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        raise CheckpointDurabilityError(
            "crash-durable checkpoint publication requires POSIX directory fsync support",
            published=False,
        )


def _safe_fsync_path(path: Path, *, directory: bool) -> None:
    """fsync one unchanged regular file or directory without following symlinks."""

    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise OSError(f"path disappeared before fsync: {path}") from exc

    expected = stat.S_ISDIR if directory else stat.S_ISREG
    kind = "directory" if directory else "regular file"
    if stat.S_ISLNK(before.st_mode) or not expected(before.st_mode):
        raise OSError(f"fsync target must be a non-symlink {kind}: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not expected(opened.st_mode):
            raise OSError(f"fsync target changed type while opening: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(f"fsync target changed identity while opening: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_verified_tree(checkpoint: Path) -> dict[str, Any]:
    manifest = verify_checkpoint(checkpoint)
    for name in _SYNC_FILE_ORDER:
        _safe_fsync_path(checkpoint / name, directory=False)
    _safe_fsync_path(checkpoint, directory=True)
    return manifest


def _publish_staged(staged: Path, destination: Path) -> None:
    os.replace(staged, destination)


def confirm_checkpoint_durability(directory: str | Path) -> dict[str, Any]:
    """Verify and synchronously persist an already-published checkpoint.

    This is an idempotent recovery operation for the case where publication
    succeeded but the final parent-directory fsync raised. A successful return
    means the local POSIX persistence protocol completed; it is not a claim
    about remote/object/distributed filesystems or physical power-cut testing.
    """

    _require_local_posix_directory_fsync()
    checkpoint = Path(directory)
    try:
        manifest = _sync_verified_tree(checkpoint)
        _safe_fsync_path(checkpoint.parent, directory=True)
    except OSError as exc:
        raise CheckpointDurabilityError(
            f"could not confirm checkpoint crash durability: {exc}",
            published=checkpoint.exists() and not checkpoint.is_symlink(),
        ) from exc
    return manifest


def save_durable_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    identity: CheckpointIdentity,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish checkpoint-v1 with local-POSIX crash-durability ordering.

    The visible destination is not created until the staged checkpoint has been
    fully serialized, verified, and fsynced. After the same-parent atomic
    rename, the parent directory is fsynced before success is returned.

    Checkpoint-v1 remains immutable. ``overwrite=True`` is accepted only for
    API compatibility and never permits replacement of an existing target.
    """

    _require_local_posix_directory_fsync()
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        if overwrite:
            raise FileExistsError(
                "checkpoint-v1 is immutable and cannot overwrite existing destination: "
                f"{destination}"
            )
        raise FileExistsError(f"checkpoint already exists: {destination}")

    parent = destination.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError as exc:
        raise CheckpointDurabilityError(
            "crash-durable publication requires a pre-existing destination parent",
            published=False,
        ) from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise CheckpointDurabilityError(
            "crash-durable publication requires a real non-symlink destination parent",
            published=False,
        )

    staged = parent / f".{destination.name}.durable-{uuid.uuid4().hex}"
    staged_owned = False
    published = False
    try:
        manifest = save_checkpoint(
            staged,
            model=model,
            identity=identity,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
            overwrite=False,
        )
        staged_owned = True
        try:
            _sync_verified_tree(staged)
        except OSError as exc:
            raise CheckpointDurabilityError(
                f"staged checkpoint fsync failed: {exc}", published=False
            ) from exc

        # Recheck immediately before publication. The destination parent is a
        # trusted single-writer namespace; checkpoint-v1 does not claim a
        # cross-process rename-no-replace primitive on every POSIX platform.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint already exists: {destination}")

        try:
            _publish_staged(staged, destination)
        except OSError as exc:
            raise CheckpointDurabilityError(
                f"atomic checkpoint publication failed: {exc}", published=False
            ) from exc
        staged_owned = False
        published = True

        try:
            _safe_fsync_path(parent, directory=True)
        except OSError as exc:
            raise CheckpointDurabilityError(
                "checkpoint is visible and verifies, but parent-directory fsync failed; "
                "durability is uncertain and the published checkpoint was preserved",
                published=True,
            ) from exc
        return manifest
    finally:
        if staged_owned and not published and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
