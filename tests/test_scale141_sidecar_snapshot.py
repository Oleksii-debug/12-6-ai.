from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import twelve_six.scale141_resume_sidecar as sidecar
from twelve_six.checkpoint import recovery_lock
from twelve_six.scale141_resume_sidecar import ResumeSidecarError, _read_payload


def _write(path: Path, payload: dict[str, object]) -> str:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_read_payload_uses_exact_opened_snapshot_when_path_changes_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    expected_payload = {"generation": "generation-00000001", "value": "verified"}
    expected_hash = _write(path, expected_payload)
    replacement = tmp_path / "replacement.json"
    _write(replacement, {"generation": "generation-00000001", "value": "tampered"})
    real_open = sidecar.os.open
    swapped = False

    def open_then_swap(target: os.PathLike[str] | str, flags: int, *args: object) -> int:
        nonlocal swapped
        fd = real_open(target, flags, *args)
        if Path(target) == path and not swapped:
            swapped = True
            os.replace(replacement, path)
        return fd

    monkeypatch.setattr(sidecar.os, "open", open_then_swap)

    observed = _read_payload(path, expected_hash)

    assert swapped is True
    assert observed == expected_payload
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == "tampered"


def test_read_payload_rejects_path_swap_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    expected_hash = _write(path, {"value": "verified"})
    replacement = tmp_path / "replacement.json"
    _write(replacement, {"value": "tampered"})
    real_open = sidecar.os.open
    swapped = False

    def swap_then_open(target: os.PathLike[str] | str, flags: int, *args: object) -> int:
        nonlocal swapped
        if Path(target) == path and not swapped:
            swapped = True
            os.replace(replacement, path)
        return real_open(target, flags, *args)

    monkeypatch.setattr(sidecar.os, "open", swap_then_open)

    with pytest.raises(ResumeSidecarError, match="changed while opening"):
        _read_payload(path, expected_hash)


def test_read_payload_rejects_symlink_even_when_target_hash_matches(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    expected_hash = _write(target, {"value": "verified"})
    link = tmp_path / "state.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(ResumeSidecarError, match="regular non-symlink"):
        _read_payload(link, expected_hash)


def test_publication_lock_rejects_symlink_path(tmp_path: Path) -> None:
    root = tmp_path / "recovery"
    root.mkdir()
    target = tmp_path / "outside.lock"
    target.write_bytes(b"")
    link = root / recovery_lock.LOCK_NAME
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(OSError), recovery_lock.exclusive_recovery_lock(root):
        pytest.fail("symlinked publication lock must never be acquired")


def test_publication_lock_rejects_path_replacement_after_fd_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "recovery"
    real_lock_fd = recovery_lock._lock_fd
    swapped = False

    def lock_then_replace(fd: int) -> None:
        nonlocal swapped
        real_lock_fd(fd)
        lock_path = root / recovery_lock.LOCK_NAME
        replacement = root / ".replacement.lock"
        replacement.write_bytes(b"")
        os.replace(replacement, lock_path)
        swapped = True

    monkeypatch.setattr(recovery_lock, "_lock_fd", lock_then_replace)

    with (
        pytest.raises(OSError, match="lock path changed while locked"),
        recovery_lock.exclusive_recovery_lock(root),
    ):
        pytest.fail("replaced publication lock path must never enter critical section")
    assert swapped is True


def test_publication_lock_rejects_recovery_root_replacement_while_locking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "recovery"
    real_lock_fd = recovery_lock._lock_fd
    swapped = False

    def lock_then_replace_root(fd: int) -> None:
        nonlocal swapped
        real_lock_fd(fd)
        moved = tmp_path / "moved-recovery"
        os.replace(root, moved)
        root.mkdir()
        swapped = True

    monkeypatch.setattr(recovery_lock, "_lock_fd", lock_then_replace_root)

    with (
        pytest.raises(
            OSError,
            match="recovery root changed while publication lock was acquired",
        ),
        recovery_lock.exclusive_recovery_lock(root),
    ):
        pytest.fail("replaced recovery root must never enter critical section")
    assert swapped is True
