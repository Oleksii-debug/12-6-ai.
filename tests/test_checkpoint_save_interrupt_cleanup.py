"""An interrupted checkpoint write must not leave staged model weights behind.

Non-owning negative regression for the checkpoint core writer. A process kill
cannot run Python cleanup; these cases cover catchable BaseException exits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twelve_six.checkpoint import core


class _TinyModel:
    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weight": np.asarray([1.0, 2.0], dtype=np.float32)}


def _identity() -> core.CheckpointIdentity:
    return core.CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "interrupted-save-regression", "parameters": 2},
        parameter_count=2,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 1},
        seed=3,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["weights", "state", "verified", "publication"])
def test_interrupted_save_cleans_private_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
    stage: str,
) -> None:
    """Clean up even if a catchable interruption occurs after data is written."""

    destination = tmp_path / "checkpoint"
    staged: list[Path] = []
    real_mkdtemp = core.tempfile.mkdtemp

    def tracked_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        staged.append(Path(path))
        return path

    monkeypatch.setattr(core.tempfile, "mkdtemp", tracked_mkdtemp)
    thrown = interruption(f"interrupted after {stage}")
    saves = 0
    real_save = core.save_safetensors

    def interrupted_save(*args: Any, **kwargs: Any) -> None:
        nonlocal saves
        real_save(*args, **kwargs)
        saves += 1
        if (stage == "weights" and saves == 1) or (stage == "state" and saves == 2):
            raise thrown

    if stage in {"weights", "state"}:
        monkeypatch.setattr(core, "save_safetensors", interrupted_save)
    elif stage == "verified":
        real_verify = core.verify_checkpoint

        def interrupted_verify(directory: Path) -> dict[str, Any]:
            real_verify(directory)
            raise thrown

        monkeypatch.setattr(core, "verify_checkpoint", interrupted_verify)
    else:
        def interrupted_publish(*_args: Any, **_kwargs: Any) -> None:
            raise thrown

        monkeypatch.setattr(core.os, "replace", interrupted_publish)

    with pytest.raises(interruption) as caught:
        core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())

    assert caught.value is thrown
    assert len(staged) == 1
    assert saves == {"weights": 1, "state": 2, "verified": 0, "publication": 0}[stage]
    assert not destination.exists(), "an interrupted save must not publish a checkpoint"
    assert not staged[0].exists(), "staged checkpoint bytes must be deleted on interruption"
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_uninterrupted_save_still_publishes_a_verified_checkpoint(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())
    assert destination.is_dir()
    assert core.verify_checkpoint(destination)["identity"]["step"] == 0
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_after_interrupted_save_same_destination_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    destination = tmp_path / "checkpoint"
    real_save = core.save_safetensors
    thrown = interruption("first save interrupted after weights")

    def interrupted_first_write(*args: Any, **kwargs: Any) -> None:
        real_save(*args, **kwargs)
        raise thrown

    with monkeypatch.context() as patch:
        patch.setattr(core, "save_safetensors", interrupted_first_write)
        with pytest.raises(interruption) as caught:
            core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())
    assert caught.value is thrown
    assert not destination.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))

    core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())
    assert core.verify_checkpoint(destination)["identity"]["step"] == 0
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


@pytest.mark.parametrize("overwrite", [False, True])
def test_existing_checkpoint_is_never_modified_or_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool,
) -> None:
    destination = tmp_path / "checkpoint"
    core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())
    original = (destination / core.MANIFEST_NAME).read_bytes()

    def unexpected_save(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("existing destination must reject before any staged write")

    monkeypatch.setattr(core, "save_safetensors", unexpected_save)
    with pytest.raises(FileExistsError):
        core.save_checkpoint(
            destination, model=_TinyModel(), identity=_identity(), overwrite=overwrite,
        )
    assert (destination / core.MANIFEST_NAME).read_bytes() == original
    assert core.verify_checkpoint(destination)["identity"]["step"] == 0
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))

@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupt_after_atomic_rename_preserves_published_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    destination = tmp_path / "checkpoint"
    thrown = interruption("interrupted after atomic publication")
    real_replace = core.os.replace

    def published_then_interrupted(source: Path, target: Path) -> None:
        real_replace(source, target)
        raise thrown

    with monkeypatch.context() as patch:
        patch.setattr(core.os, "replace", published_then_interrupted)
        with pytest.raises(interruption) as caught:
            core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())
    assert caught.value is thrown
    assert destination.is_dir(), "a published checkpoint must survive late interruption"
    assert core.verify_checkpoint(destination)["identity"]["step"] == 0
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))
