from __future__ import annotations

from pathlib import Path

import pytest

import twelve_six.scale141_recovery as recovery


def test_pointer_snapshot_rejects_oversized_file_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = tmp_path / recovery.CURRENT_NAME
    original = b"x" * (recovery.MAX_POINTER_BYTES + 1)
    pointer.write_bytes(original)

    real_read = recovery.os.read
    read_calls = 0

    def counted_read(fd: int, size: int) -> bytes:
        nonlocal read_calls
        read_calls += 1
        return real_read(fd, size)

    monkeypatch.setattr(recovery.os, "read", counted_read)
    with pytest.raises(recovery.RecoveryLifecycleError, match="maximum supported size"):
        recovery._read_pointer_snapshot(pointer)

    assert read_calls == 0
    monkeypatch.setattr(recovery.os, "read", real_read)
    assert pointer.read_bytes() == original


def test_pointer_snapshot_caps_growth_after_small_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = tmp_path / recovery.CURRENT_NAME
    original = b"{}"
    pointer.write_bytes(original)
    assert pointer.stat().st_size < recovery.MAX_POINTER_BYTES

    real_read = recovery.os.read
    simulated_growth = b"y" * (recovery.MAX_POINTER_BYTES + 1)
    offset = 0
    requested: list[int] = []
    returned = 0

    def growing_read(_fd: int, size: int) -> bytes:
        nonlocal offset, returned
        requested.append(size)
        chunk = simulated_growth[offset : offset + size]
        offset += len(chunk)
        returned += len(chunk)
        return chunk

    monkeypatch.setattr(recovery.os, "read", growing_read)
    with pytest.raises(recovery.RecoveryLifecycleError, match="maximum supported size"):
        recovery._read_pointer_snapshot(pointer)

    assert returned == recovery.MAX_POINTER_BYTES + 1
    assert sum(requested) <= recovery.MAX_POINTER_BYTES + 1
    monkeypatch.setattr(recovery.os, "read", real_read)
    assert pointer.read_bytes() == original
