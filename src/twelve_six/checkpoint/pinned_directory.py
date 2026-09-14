"""Identity-pinned directory handles for recovery child namespaces.

Recovery's top-level root is already protected by a directory-handle snapshot,
but replaceable descendants such as ``checkpoints/``, ``generations/`` and
``resume-states/`` need the same property while they are read or mutated.
This module provides one cross-platform primitive instead of repeating
path-only lstat checks at every call site.
"""
from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class PinnedDirectoryError(RuntimeError):
    """Raised when a child directory cannot be identity-pinned safely."""


def _identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _require_real_directory(path: Path) -> os.stat_result:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise PinnedDirectoryError(f"directory does not exist: {path}") from exc
    except OSError as exc:
        raise PinnedDirectoryError(f"directory cannot be inspected safely: {path}") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise PinnedDirectoryError(f"directory must be a real non-symlink directory: {path}")
    return observed


def _prepare_directory(path: Path, *, create: bool) -> os.stat_result:
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PinnedDirectoryError(f"directory cannot be created safely: {path}") from exc
    return _require_real_directory(path)


@dataclass(frozen=True, slots=True)
class PinnedDirectory:
    """A visible directory plus a stable path to the opened directory object."""

    visible_path: Path
    path: Path
    identity: tuple[int, int]

    def verify_binding(self) -> None:
        """Fail closed if the visible pathname no longer names the pinned object."""

        visible = _require_real_directory(self.visible_path)
        if _identity(visible) != self.identity:
            raise PinnedDirectoryError(
                f"directory binding changed during critical section: {self.visible_path}"
            )
        try:
            pinned = self.path.stat()
        except OSError as exc:
            raise PinnedDirectoryError(
                f"pinned directory handle is no longer usable: {self.visible_path}"
            ) from exc
        if not stat.S_ISDIR(pinned.st_mode) or _identity(pinned) != self.identity:
            raise PinnedDirectoryError(
                f"pinned directory identity changed: {self.visible_path}"
            )


def _stable_posix_alias(fd: int, identity: tuple[int, int]) -> Path:
    for candidate in (Path(f"/proc/self/fd/{fd}"), Path(f"/dev/fd/{fd}")):
        try:
            observed = candidate.stat()
        except OSError:
            continue
        if stat.S_ISDIR(observed.st_mode) and _identity(observed) == identity:
            return candidate
    raise PinnedDirectoryError(
        "platform exposes no stable directory-fd alias; refusing pathname-only mutation"
    )


@contextmanager
def _pinned_posix(path: Path, *, create: bool) -> Iterator[PinnedDirectory]:
    before = _prepare_directory(path, create=create)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PinnedDirectoryError(f"directory cannot be opened safely: {path}") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(before):
            raise PinnedDirectoryError(f"directory changed while opening: {path}")
        identity = _identity(opened)
        pinned = PinnedDirectory(
            visible_path=path,
            path=_stable_posix_alias(fd, identity),
            identity=identity,
        )
        pinned.verify_binding()
        try:
            yield pinned
        finally:
            pinned.verify_binding()
    finally:
        os.close(fd)


def _open_windows_directory(path: Path) -> tuple[int, tuple[int, int]]:
    import ctypes
    from ctypes import wintypes

    file_read_attributes = 0x0080
    file_share_read = 0x00000001
    file_share_write = 0x00000002
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    file_attribute_reparse_point = 0x00000400

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        file_read_attributes,
        file_share_read | file_share_write,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    handle_value = ctypes.cast(handle, ctypes.c_void_p).value
    if handle_value in (None, invalid):
        raise PinnedDirectoryError(
            f"directory cannot be opened without delete sharing: {path}"
        )

    info = ByHandleFileInformation()
    if not get_info(handle, ctypes.byref(info)):
        kernel32.CloseHandle(handle)
        raise PinnedDirectoryError(f"directory identity cannot be read: {path}")
    if info.dwFileAttributes & file_attribute_reparse_point:
        kernel32.CloseHandle(handle)
        raise PinnedDirectoryError(f"directory must not be a Windows reparse point: {path}")

    file_index = (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow)
    return int(handle_value), (int(info.dwVolumeSerialNumber), file_index)


@contextmanager
def _pinned_windows(path: Path, *, create: bool) -> Iterator[PinnedDirectory]:
    import ctypes
    from ctypes import wintypes

    before = _prepare_directory(path, create=create)
    handle, _handle_identity = _open_windows_directory(path)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        # Python's Windows stat exposes the same stable file identity needed to
        # detect a pathname swap that happened between lstat and CreateFileW.
        after_open = _require_real_directory(path)
        if _identity(after_open) != _identity(before):
            raise PinnedDirectoryError(f"directory changed while opening: {path}")
        pinned = PinnedDirectory(
            visible_path=path,
            path=path,
            identity=_identity(after_open),
        )
        pinned.verify_binding()
        try:
            yield pinned
        finally:
            pinned.verify_binding()
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


@contextmanager
def pinned_real_directory(
    path: str | Path,
    *,
    create: bool = False,
) -> Iterator[PinnedDirectory]:
    """Pin one real directory for a complete read/mutation critical section.

    POSIX operations use a stable fd-backed alias, so even a hostile rename of
    the visible child root cannot redirect writes/deletes. Windows holds a
    directory HANDLE opened without ``FILE_SHARE_DELETE``; rename/delete of the
    pinned directory is therefore denied until the context exits. Both paths
    revalidate the visible binding before returning control to the caller.
    """

    target = Path(path)
    if os.name == "posix":
        with _pinned_posix(target, create=create) as pinned:
            yield pinned
        return
    if os.name == "nt":
        with _pinned_windows(target, create=create) as pinned:
            yield pinned
        return
    raise PinnedDirectoryError(
        f"identity-pinned recovery directories are unsupported on os.name={os.name!r}"
    )
