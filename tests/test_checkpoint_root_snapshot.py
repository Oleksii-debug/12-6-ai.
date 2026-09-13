from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    root_snapshot,
    save_checkpoint,
)


class Model:
    def __init__(self, value: list[float]) -> None:
        self.weights = np.asarray(value, dtype=np.float64).copy()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(
        self,
        state: dict[str, np.ndarray],
        strict: bool = True,
    ) -> None:
        assert not strict or set(state) == {"weights"}
        self.weights = state["weights"].copy()


def _identity(step: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "root-snapshot-test", "width": 3},
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


def _require_posix_root_snapshot() -> None:
    if not root_snapshot._supports_pinned_root():
        pytest.skip("directory-fd checkpoint snapshots are unavailable")


def test_checkpoint_load_keeps_opened_root_when_path_is_replaced_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_posix_root_snapshot()
    checkpoint = tmp_path / "checkpoint"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "opened-original"
    save_checkpoint(
        checkpoint,
        model=Model([1.0, 2.0, 3.0]),
        identity=_identity(1),
    )
    save_checkpoint(
        replacement,
        model=Model([9.0, 9.0, 9.0]),
        identity=_identity(2),
    )

    real_open_root = root_snapshot._open_root_fd
    swapped = False

    def open_then_swap(root: Path, *, error_type: type[Exception]) -> int:
        nonlocal swapped
        fd = real_open_root(root, error_type=error_type)
        if Path(root) == checkpoint and not swapped:
            swapped = True
            os.replace(checkpoint, moved)
            os.replace(replacement, checkpoint)
        return fd

    monkeypatch.setattr(root_snapshot, "_open_root_fd", open_then_swap)

    verified = prepare_checkpoint_load(checkpoint)
    target = Model([0.0, 0.0, 0.0])
    load_verified_checkpoint(verified, model=target, restore_rng=False)

    assert swapped is True
    assert verified.manifest["identity"]["step"] == 1
    np.testing.assert_array_equal(target.weights, [1.0, 2.0, 3.0])
    assert '"step":2' in (checkpoint / "manifest.json").read_text(encoding="utf-8")


def test_checkpoint_root_swap_between_lstat_and_open_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_posix_root_snapshot()
    checkpoint = tmp_path / "checkpoint"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "validated-original"
    save_checkpoint(
        checkpoint,
        model=Model([1.0, 2.0, 3.0]),
        identity=_identity(1),
    )
    save_checkpoint(
        replacement,
        model=Model([9.0, 9.0, 9.0]),
        identity=_identity(2),
    )

    real_lstat = Path.lstat
    swapped = False

    def lstat_then_swap(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = real_lstat(path)
        if path == checkpoint and not swapped:
            swapped = True
            os.replace(checkpoint, moved)
            os.replace(replacement, checkpoint)
        return observed

    monkeypatch.setattr(Path, "lstat", lstat_then_swap)

    with pytest.raises(CheckpointIntegrityError, match="root changed while opening"):
        prepare_checkpoint_load(checkpoint)
    assert swapped is True
