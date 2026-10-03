"""Crash-releasing cross-process serialization for recovery publication."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_NAME = ".publication.lock"
_PATH_LOCK_SUFFIX = ".publication-path.lock"
_NAMESPACE_LOCK_SUFFIX = ".publication-namespace.lock"


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_lock_file(
    root: Path, *, name: str = LOCK_NAME
) -> tuple[int, os.stat_result]:
    path = root / name
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    keep_open = False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("recovery publication lock must be a regular file")
        visible = path.lstat()
        if stat.S_ISLNK(visible.st_mode) or not stat.S_ISREG(visible.st_mode):
            raise OSError("recovery publication lock path must be a regular non-symlink file")
        if not _same_object(opened, visible):
            raise OSError("recovery publication lock changed during open")
        keep_open = True
        return fd, opened
    finally:
        if not keep_open:
            os.close(fd)


def _lock_fd_impl(fd: int) -> None:
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


def _lock_fd(fd: int) -> None:
    """Lock the in-root publication fd.

    Kept as a distinct seam because adversarial tests replace this function to
    exercise lock/root pathname races immediately after acquisition.
    """

    _lock_fd_impl(fd)


def _lock_path_guard_fd(fd: int) -> None:
    _lock_fd_impl(fd)


def _unlock_fd_impl(fd: int) -> None:
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


def _unlock_fd(fd: int) -> None:
    _unlock_fd_impl(fd)


def _unlock_path_guard_fd(fd: int) -> None:
    _unlock_fd_impl(fd)


def _open_root_fd(root: Path, expected: os.stat_result) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(root, flags)
    keep_open = False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise OSError("recovery root must stay a directory while locking")
        if not _same_object(expected, opened):
            raise OSError("recovery root changed while opening pinned directory")
        keep_open = True
        return fd
    finally:
        if not keep_open:
            os.close(fd)


def _stable_root_alias(fd: int, expected: os.stat_result) -> Path:
    """Return a pathname that resolves through the already-open directory fd.

    Recovery publication callbacks are Path-based.  On POSIX, `/proc/self/fd`
    (Linux) or `/dev/fd` (BSD/macOS where directory-fd traversal is exposed)
    lets those existing callbacks stay bound to the opened directory inode even
    if the user-visible recovery pathname is renamed after lock acquisition.
    Platforms without such a stable fd alias fail closed rather than falling
    back to the vulnerable pathname during a mutation critical section.
    """

    for base in (Path("/proc/self/fd"), Path("/dev/fd")):
        candidate = base / str(fd)
        try:
            observed = candidate.stat()
        except OSError:
            continue
        if stat.S_ISDIR(observed.st_mode) and _same_object(expected, observed):
            return candidate
    raise OSError("platform does not expose a stable recovery-root directory-fd path")


def _path_guard_name(root: Path) -> str:
    if not root.name:
        raise OSError("recovery root must have a stable basename")
    return f".{root.name}{_PATH_LOCK_SUFFIX}"


def _namespace_guard_name(root: Path) -> str:
    """Return a bounded guard name for the logical parent/root component pair."""

    if not root.name:
        raise OSError("recovery root must have a stable basename")
    parent_bytes = os.fsencode(root.parent.name)
    root_bytes = os.fsencode(root.name)
    payload = (
        b"twelve-six-recovery-namespace-v1\0"
        + len(parent_bytes).to_bytes(4, "big")
        + parent_bytes
        + len(root_bytes).to_bytes(4, "big")
        + root_bytes
    )
    digest = hashlib.sha256(payload).hexdigest()
    return f".{digest}{_NAMESPACE_LOCK_SUFFIX}"


def _real_directory(path: Path, *, role: str) -> os.stat_result:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise OSError(f"{role} must be a real directory")
    return observed


def _verify_visible_directory(
    path: Path, expected: os.stat_result, *, role: str, phase: str
) -> None:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise OSError(f"{role} changed type {phase}")
    if not _same_object(expected, observed):
        raise OSError(f"{role} changed {phase}")


@contextmanager
def exclusive_recovery_lock(root: str | Path) -> Iterator[Path]:
    """Serialize publication/cleanup and pin the root for the full critical section.

    The outer namespace guard is stored one level above the replaceable recovery
    parent and is keyed by the logical `(parent basename, recovery basename)` pair.
    It is acquired before the mutable parent/root is created or opened, so replacing
    the immediate parent cannot mint a second logical-path lock lane.  The existing
    sibling path guard and in-root lock remain in place below that authority.

    On POSIX, mutation callers receive a directory-fd alias, so descendant path
    operations stay on the opened root inode even if the visible pathname is renamed
    after this context yields.  The guarantee is intentionally scoped to immediate
    recovery-parent replacement under a stable containing namespace; it does not
    claim safety against replacement of that containing namespace itself.

    All lock files are intentionally persistent. Kernel advisory locks are released
    when the owning process exits, including abnormal termination.
    """

    recovery_root = Path(root)
    parent = recovery_root.parent
    guard_namespace = parent.parent
    guard_namespace.mkdir(parents=True, exist_ok=True)

    namespace_guard_fd, _ = _open_lock_file(
        guard_namespace, name=_namespace_guard_name(recovery_root)
    )
    namespace_guard_locked = False
    parent_fd: int | None = None
    path_guard_fd: int | None = None
    path_guard_locked = False
    root_fd: int | None = None
    lock_fd: int | None = None
    lock_locked = False
    try:
        _lock_path_guard_fd(namespace_guard_fd)
        namespace_guard_locked = True

        parent.mkdir(parents=True, exist_ok=True)
        parent_before = _real_directory(parent, role="recovery parent")
        if os.name == "posix":
            parent_fd = _open_root_fd(parent, parent_before)

        recovery_root.mkdir(exist_ok=True)
        _verify_visible_directory(
            parent,
            parent_before,
            role="recovery parent",
            phase="while recovery root was prepared",
        )

        path_guard_fd, _ = _open_lock_file(
            parent, name=_path_guard_name(recovery_root)
        )
        _lock_path_guard_fd(path_guard_fd)
        path_guard_locked = True

        before = _real_directory(recovery_root, role="recovery root")
        if os.name == "posix":
            root_fd = _open_root_fd(recovery_root, before)
            mutation_root = _stable_root_alias(root_fd, before)
        else:
            mutation_root = recovery_root

        lock_fd, opened = _open_lock_file(mutation_root)
        _lock_fd(lock_fd)
        lock_locked = True

        _verify_visible_directory(
            parent,
            parent_before,
            role="recovery parent",
            phase="while publication lock was acquired",
        )
        _verify_visible_directory(
            recovery_root,
            before,
            role="recovery root",
            phase="while publication lock was acquired",
        )
        visible = (mutation_root / LOCK_NAME).lstat()
        if stat.S_ISLNK(visible.st_mode) or not _same_object(opened, visible):
            raise OSError("recovery publication lock path changed while locked")

        yield mutation_root

        _verify_visible_directory(
            parent,
            parent_before,
            role="recovery parent",
            phase="during publication critical section",
        )
        _verify_visible_directory(
            recovery_root,
            before,
            role="recovery root",
            phase="during publication critical section",
        )
    finally:
        try:
            if lock_fd is not None and lock_locked:
                _unlock_fd(lock_fd)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            if root_fd is not None:
                os.close(root_fd)
            try:
                if path_guard_fd is not None and path_guard_locked:
                    _unlock_path_guard_fd(path_guard_fd)
            finally:
                if path_guard_fd is not None:
                    os.close(path_guard_fd)
                if parent_fd is not None:
                    os.close(parent_fd)
                try:
                    if namespace_guard_locked:
                        _unlock_path_guard_fd(namespace_guard_fd)
                finally:
                    os.close(namespace_guard_fd)
