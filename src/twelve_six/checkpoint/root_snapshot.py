"""Pin checkpoint-v1 root directory identity across verify/load on POSIX.

Core checkpoint-v1 already snapshots each artifact with lstat/open/fstat checks.
This module closes the remaining parent-path TOCTOU: once a checkpoint root is
accepted, every inventory and artifact lookup is relative to one opened
directory descriptor rather than repeatedly resolving the root pathname.
"""
from __future__ import annotations

import os
import stat
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_ROOT_SNAPSHOT: ContextVar[tuple[Path, int] | None] = ContextVar(
    "d05_checkpoint_root_snapshot",
    default=None,
)


def _supports_pinned_root() -> bool:
    return (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.listdir in os.supports_fd
    )


def _active_fd(root: Path) -> int | None:
    active = _ROOT_SNAPSHOT.get()
    if active is None:
        return None
    active_root, fd = active
    return fd if active_root == root else None


def _open_root_fd(root: Path, *, error_type: type[Exception]) -> int:
    try:
        before = root.lstat()
    except FileNotFoundError as exc:
        raise error_type(f"checkpoint directory does not exist: {root}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise error_type("checkpoint root must be a real directory, not a symlink")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(root, flags)
    except OSError as exc:
        raise error_type("checkpoint root cannot be opened safely") from exc

    owned = True
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise error_type("checkpoint root changed type while opening")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise error_type("checkpoint root changed while opening")
        owned = False
        return fd
    finally:
        if owned:
            os.close(fd)


def install(core_module: Any) -> None:
    """Install root-pinned acquisition while preserving non-POSIX behavior."""

    original_prepare = core_module.prepare_checkpoint_load
    original_require_directory = core_module._require_checkpoint_directory
    original_read_regular_bytes = core_module._read_regular_bytes

    def require_checkpoint_directory(root: Path) -> None:
        root = Path(root)
        fd = _active_fd(root)
        if fd is None:
            original_require_directory(root)
            return
        try:
            names = set(os.listdir(fd))
        except OSError as exc:
            raise core_module.CheckpointIntegrityError(
                "checkpoint root cannot be inventoried safely"
            ) from exc
        if names != core_module._DIRECTORY_NAMES:
            missing = sorted(core_module._DIRECTORY_NAMES - names)
            unexpected = sorted(names - core_module._DIRECTORY_NAMES)
            raise core_module.CheckpointIntegrityError(
                "checkpoint directory inventory mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def read_regular_bytes(root: Path, name: str) -> bytes:
        root = Path(root)
        root_fd = _active_fd(root)
        if root_fd is None:
            return original_read_regular_bytes(root, name)
        try:
            before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise core_module.CheckpointIntegrityError(
                f"missing checkpoint artifact: {name}"
            ) from exc
        except OSError as exc:
            raise core_module.CheckpointIntegrityError(
                f"cannot safely inspect checkpoint artifact: {name}"
            ) from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise core_module.CheckpointIntegrityError(
                f"checkpoint artifact must be a regular non-symlink file: {name}"
            )

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(name, flags, dir_fd=root_fd)
        except OSError as exc:
            raise core_module.CheckpointIntegrityError(
                f"cannot safely open checkpoint artifact: {name}"
            ) from exc
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise core_module.CheckpointIntegrityError(
                    f"checkpoint artifact changed type while opening: {name}"
                )
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise core_module.CheckpointIntegrityError(
                    f"checkpoint artifact changed while opening: {name}"
                )
            with os.fdopen(fd, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(fd)

    def prepare_checkpoint_load(directory: str | Path) -> Any:
        if not _supports_pinned_root():
            return original_prepare(directory)
        root = Path(directory)
        fd = _open_root_fd(
            root,
            error_type=core_module.CheckpointIntegrityError,
        )
        token = _ROOT_SNAPSHOT.set((root, fd))
        try:
            return original_prepare(root)
        finally:
            _ROOT_SNAPSHOT.reset(token)
            os.close(fd)

    core_module._require_checkpoint_directory = require_checkpoint_directory
    core_module._read_regular_bytes = read_regular_bytes
    core_module.prepare_checkpoint_load = prepare_checkpoint_load
