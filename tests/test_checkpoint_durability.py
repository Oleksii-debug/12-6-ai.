from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.checkpoint.durability import fsync_checkpoint_tree, fsync_parent_directory


def test_fsync_checkpoint_tree_flushes_exact_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkpoint.tmp"
    root.mkdir()
    (root / "a").write_bytes(b"alpha")
    (root / "b").write_bytes(b"beta")
    calls: list[int] = []
    monkeypatch.setattr("twelve_six.checkpoint.durability.os.fsync", calls.append)

    fsync_checkpoint_tree(root, expected_names=frozenset({"a", "b"}))

    assert len(calls) == 3  # two files plus containing directory


def test_fsync_checkpoint_tree_rejects_inventory_change(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint.tmp"
    root.mkdir()
    (root / "a").write_bytes(b"alpha")
    (root / "surprise").write_bytes(b"not verified")

    with pytest.raises(OSError, match="inventory mismatch"):
        fsync_checkpoint_tree(root, expected_names=frozenset({"a"}))


def test_fsync_checkpoint_tree_rejects_symlink_artifact(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint.tmp"
    root.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    link = root / "a"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(OSError, match="regular file"):
        fsync_checkpoint_tree(root, expected_names=frozenset({"a"}))


def test_fsync_parent_directory_flushes_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "checkpoint"
    calls: list[int] = []
    monkeypatch.setattr("twelve_six.checkpoint.durability.os.fsync", calls.append)

    fsync_parent_directory(destination)

    assert len(calls) == 1
