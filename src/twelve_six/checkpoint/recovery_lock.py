"""Crash-releasing cross-process serialization for recovery publication."""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_NAME = ".publication.lock"


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_lock_file(root: Path) -> tuple[int, os.stat_result]:
    path = root / LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("recovery publication lock must be a regular file")
        visible = path.lstat()
        if stat.S_ISLNK(visible.st_mode) or not stat.S_ISREG(visible.st_mode):
            raise OSError("recovery publication lock path must be a regular non-symlink file")
        if not _same_object(opened, visible):
            raise OSError("recovery publication lock changed during open")
        return fd, opened
    except BaseException:
        os.close(fd)
        raise


def _lock_fd(fd: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
        return
    if os.name == "nt":
        import msvcrt

        if os.fstat(fd).st_size == 0:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, b"\0")
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        return
    raise OSError(f"unsupported platform for recovery publication lock: {os.name}")


def _unlock_fd(fd: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    raise OSError(f"unsupported platform for recovery publication lock: {os.name}")


@contextmanager
def exclusive_recovery_lock(root: str | Path) -> Iterator[None]:
    """Serialize publication/cleanup and release automatically on process death.

    The lock file is intentionally persistent. Removing it would allow two publishers
    to lock different inodes after a rename/unlink race. Kernel advisory locks are
    released when the owning process exits, including abnormal termination.
    """

    recovery_root = Path(root)
    recovery_root.mkdir(parents=True, exist_ok=True)
    before = recovery_root.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise OSError("recovery root must be a real directory before locking")

    fd, opened = _open_lock_file(recovery_root)
    locked = False
    try:
        _lock_fd(fd)
        locked = True
        after = recovery_root.lstat()
        if stat.S_ISLNK(after.st_mode) or not stat.S_ISDIR(after.st_mode):
            raise OSError("recovery root changed type while publication lock was acquired")
        if not _same_object(before, after):
            raise OSError("recovery root changed while publication lock was acquired")
        visible = (recovery_root / LOCK_NAME).lstat()
        if stat.S_ISLNK(visible.st_mode) or not _same_object(opened, visible):
            raise OSError("recovery publication lock path changed while locked")
        yield
    finally:
        try:
            if locked:
                _unlock_fd(fd)
        finally:
            os.close(fd)
