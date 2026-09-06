"""Crash-durability primitives for transactional checkpoint publication."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any


def _fsync_regular_file(path: Path) -> None:
    """Flush one existing regular non-symlink file to stable storage."""
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OSError(f"checkpoint durability requires regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError(f"checkpoint artifact changed type during fsync: {path.name}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(f"checkpoint artifact changed during fsync: {path.name}")
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_checkpoint_tree(directory: str | Path, *, expected_names: frozenset[str]) -> None:
    """Durably flush a fully verified unpublished checkpoint tree."""
    root = Path(directory)
    before = root.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise OSError("checkpoint durability root must be a real directory")
    names = {entry.name for entry in root.iterdir()}
    if names != expected_names:
        raise OSError(
            "checkpoint durability inventory mismatch: "
            f"missing={sorted(expected_names - names)}, unexpected={sorted(names - expected_names)}"
        )
    for name in sorted(expected_names):
        _fsync_regular_file(root / name)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(root, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise OSError("checkpoint durability root changed type during fsync")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError("checkpoint durability root changed during fsync")
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_parent_directory(path: str | Path) -> None:
    """Flush the parent directory after atomic publication of a checkpoint."""
    parent = Path(path).parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(parent, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("checkpoint parent is not a directory")
        os.fsync(fd)
    finally:
        os.close(fd)


def install(core: Any) -> None:
    """Wrap production save with payload+directory fsync before durable publication.

    The existing checkpoint-v1 writer first publishes into a private staging name.
    Only a fully verified staging tree is fsynced, atomically renamed to the caller's
    destination, and followed by a parent-directory fsync.  A failure before the
    final rename leaves no destination.  A parent-fsync failure is reported even
    though the rename may already be visible, because power-loss durability is then
    not proven.
    """
    if getattr(core, "_D05_DURABLE_SAVE_INSTALLED", False):
        return
    original_save = core.save_checkpoint

    def save_checkpoint(directory: str | Path, **kwargs: Any) -> dict[str, Any]:
        destination = Path(directory)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.durable-", dir=destination.parent)
        )
        staging = staging_root / "checkpoint"
        try:
            manifest = original_save(staging, **kwargs)
            fsync_checkpoint_tree(staging, expected_names=core._DIRECTORY_NAMES)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"checkpoint appeared before publication: {destination}")
            os.replace(staging, destination)
            fsync_parent_directory(destination)
            return manifest
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    core.save_checkpoint = save_checkpoint
    core._D05_DURABLE_SAVE_INSTALLED = True
