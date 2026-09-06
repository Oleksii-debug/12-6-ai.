"""Crash-durability primitives for transactional checkpoint publication."""

from __future__ import annotations

import os
import stat
from pathlib import Path


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
    """Durably flush a fully verified unpublished checkpoint tree.

    Call this after integrity verification and before the atomic directory rename.
    The exact inventory is rechecked so an unexpected file cannot be silently
    published between verification and the durability barrier.
    """
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
