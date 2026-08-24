from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import twelve_six.checkpoint.durability as durability
from twelve_six.checkpoint.core import CheckpointIdentity, verify_checkpoint
from twelve_six.checkpoint.durability import (
    CheckpointDurabilityError,
    confirm_checkpoint_durability,
    save_durable_checkpoint,
)


class Model:
    def __init__(self, value: list[float]) -> None:
        self.weights = np.asarray(value, dtype=np.float64).copy()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.weights = state["weights"].copy()


def identity(step: int = 0) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "durability-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float64",
        step=step,
        tokens_seen=step * 3,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _staging_entries(parent: Path, destination_name: str) -> list[Path]:
    prefix = f".{destination_name}.durable-"
    return [entry for entry in parent.iterdir() if entry.name.startswith(prefix)]


def test_durable_save_syncs_verified_tree_before_publish_and_parent_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    events: list[tuple[str, str]] = []
    real_sync = durability._safe_fsync_path
    real_publish = durability._publish_staged

    def record_sync(path: Path, *, directory: bool) -> None:
        events.append(("sync-dir" if directory else "sync-file", path.name))
        real_sync(path, directory=directory)

    def record_publish(staged: Path, destination: Path) -> None:
        events.append(("publish", destination.name))
        real_publish(staged, destination)

    monkeypatch.setattr(durability, "_safe_fsync_path", record_sync)
    monkeypatch.setattr(durability, "_publish_staged", record_publish)

    manifest = save_durable_checkpoint(
        checkpoint, model=Model([1, 2, 3]), identity=identity()
    )

    publish_index = events.index(("publish", "checkpoint"))
    before = events[:publish_index]
    after = events[publish_index + 1 :]
    assert [name for kind, name in before if kind == "sync-file"] == list(
        durability._SYNC_FILE_ORDER
    )
    assert before[-1][0] == "sync-dir"
    assert before[-1][1].startswith(".checkpoint.durable-")
    assert after == [("sync-dir", tmp_path.name)]
    assert verify_checkpoint(checkpoint)["checkpoint_id"] == manifest["checkpoint_id"]
    assert not _staging_entries(tmp_path, checkpoint.name)


def test_prepublish_sync_failure_never_creates_visible_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    real_sync = durability._safe_fsync_path

    def fail_state_tensor(path: Path, *, directory: bool) -> None:
        if path.name == "state.safetensors":
            raise OSError("injected file fsync failure")
        real_sync(path, directory=directory)

    monkeypatch.setattr(durability, "_safe_fsync_path", fail_state_tensor)

    with pytest.raises(CheckpointDurabilityError, match="staged checkpoint fsync") as caught:
        save_durable_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    assert caught.value.published is False
    assert not checkpoint.exists()
    assert not _staging_entries(tmp_path, checkpoint.name)


def test_atomic_publish_failure_is_classified_unpublished_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"

    def fail_publish(staged: Path, destination: Path) -> None:
        del staged, destination
        raise OSError("injected rename failure")

    monkeypatch.setattr(durability, "_publish_staged", fail_publish)

    with pytest.raises(CheckpointDurabilityError, match="atomic checkpoint") as caught:
        save_durable_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    assert caught.value.published is False
    assert not checkpoint.exists()
    assert not _staging_entries(tmp_path, checkpoint.name)


def test_parent_sync_failure_preserves_verified_published_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    real_sync = durability._safe_fsync_path

    def fail_final_parent(path: Path, *, directory: bool) -> None:
        if directory and path == tmp_path:
            raise OSError("injected parent fsync failure")
        real_sync(path, directory=directory)

    monkeypatch.setattr(durability, "_safe_fsync_path", fail_final_parent)

    with pytest.raises(CheckpointDurabilityError, match="durability is uncertain") as caught:
        save_durable_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    assert caught.value.published is True
    verify_checkpoint(checkpoint)
    assert not _staging_entries(tmp_path, checkpoint.name)

    monkeypatch.setattr(durability, "_safe_fsync_path", real_sync)
    manifest = confirm_checkpoint_durability(checkpoint)
    assert manifest["checkpoint_id"] == verify_checkpoint(checkpoint)["checkpoint_id"]


def test_existing_destination_remains_immutable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_durable_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}

    with pytest.raises(FileExistsError, match="immutable"):
        save_durable_checkpoint(
            checkpoint,
            model=Model([9, 9, 9]),
            identity=identity(1),
            overwrite=True,
        )

    after = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    assert before == after


def test_missing_parent_fails_closed_without_creating_ancestors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing" / "checkpoint"

    with pytest.raises(CheckpointDurabilityError, match="pre-existing") as caught:
        save_durable_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    assert caught.value.published is False
    assert not checkpoint.parent.exists()
