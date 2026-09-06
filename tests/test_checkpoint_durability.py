from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from twelve_six.checkpoint.durability import (
    fsync_checkpoint_tree,
    fsync_parent_directory,
    install,
)


def test_fsync_checkpoint_tree_flushes_exact_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkpoint.tmp"
    root.mkdir()
    (root / "a").write_bytes(b"alpha")
    (root / "b").write_bytes(b"beta")
    calls: list[int] = []
    monkeypatch.setattr("twelve_six.checkpoint.durability.os.fsync", calls.append)

    fsync_checkpoint_tree(root, expected_names=frozenset({"a", "b"}))

    assert len(calls) == 3


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


def _fake_core(events: list[str]) -> SimpleNamespace:
    names = frozenset({"payload", "manifest"})

    def original_save(directory: Path, **kwargs: object) -> dict[str, object]:
        del kwargs
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(name.encode())
        events.append("verified-write")
        return {"checkpoint_id": "fixture"}

    return SimpleNamespace(save_checkpoint=original_save, _DIRECTORY_NAMES=names)


def test_installed_save_orders_tree_fsync_rename_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    core = _fake_core(events)
    install(core)
    real_replace = __import__("os").replace

    def tree_barrier(directory: Path, *, expected_names: frozenset[str]) -> None:
        assert {path.name for path in directory.iterdir()} == expected_names
        events.append("tree-fsync")

    def replace(source: Path, destination: Path) -> None:
        events.append("rename")
        real_replace(source, destination)

    def parent_barrier(destination: Path) -> None:
        assert destination.exists()
        events.append("parent-fsync")

    monkeypatch.setattr("twelve_six.checkpoint.durability.fsync_checkpoint_tree", tree_barrier)
    monkeypatch.setattr("twelve_six.checkpoint.durability.os.replace", replace)
    monkeypatch.setattr("twelve_six.checkpoint.durability.fsync_parent_directory", parent_barrier)

    destination = tmp_path / "final"
    manifest = core.save_checkpoint(destination)

    assert manifest == {"checkpoint_id": "fixture"}
    assert destination.is_dir()
    assert events == ["verified-write", "tree-fsync", "rename", "parent-fsync"]


def test_tree_fsync_failure_never_publishes_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _fake_core([])
    install(core)

    def fail_tree(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected tree fsync failure")

    monkeypatch.setattr("twelve_six.checkpoint.durability.fsync_checkpoint_tree", fail_tree)
    destination = tmp_path / "final"

    with pytest.raises(OSError, match="injected tree fsync failure"):
        core.save_checkpoint(destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_parent_fsync_failure_reports_uncertain_durability_after_atomic_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _fake_core([])
    install(core)

    def fail_parent(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr("twelve_six.checkpoint.durability.fsync_checkpoint_tree", lambda *a, **k: None)
    monkeypatch.setattr("twelve_six.checkpoint.durability.fsync_parent_directory", fail_parent)
    destination = tmp_path / "final"

    with pytest.raises(OSError, match="injected parent fsync failure"):
        core.save_checkpoint(destination)

    assert destination.is_dir()


def test_destination_collision_after_staging_fails_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    core = _fake_core(events)
    install(core)
    destination = tmp_path / "final"

    def inject_collision(*args: object, **kwargs: object) -> None:
        del args, kwargs
        destination.mkdir()
        (destination / "owner").write_text("keep", encoding="utf-8")

    monkeypatch.setattr("twelve_six.checkpoint.durability.fsync_checkpoint_tree", inject_collision)

    with pytest.raises(FileExistsError, match="appeared before publication"):
        core.save_checkpoint(destination)

    assert (destination / "owner").read_text(encoding="utf-8") == "keep"
