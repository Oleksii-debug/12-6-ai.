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
@pytest.mark.parametrize("stage", ["weights", "state", "publication"])
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

    if stage != "publication":
        monkeypatch.setattr(core, "save_safetensors", interrupted_save)
    else:
        def interrupted_publish(*_args: Any, **_kwargs: Any) -> None:
            raise thrown

        monkeypatch.setattr(core.os, "replace", interrupted_publish)

    with pytest.raises(interruption) as caught:
        core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())

    assert caught.value is thrown
    assert len(staged) == 1
    assert saves == {"weights": 1, "state": 2, "publication": 0}[stage]
    assert not destination.exists(), "an interrupted save must not publish a checkpoint"
    assert not staged[0].exists(), "staged checkpoint bytes must be deleted on interruption"
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_uninterrupted_save_still_publishes_a_verified_checkpoint(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    core.save_checkpoint(destination, model=_TinyModel(), identity=_identity())
    assert destination.is_dir()
    assert core.verify_checkpoint(destination)["identity"]["step"] == 0
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))
