"""Rollback global torch policy on opt-out restore with an absent torch RNG stream.

The source artifact is genuinely sealed with the production checkpoint writer.
No project training data, optimizer updates or final-test access are involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity
from twelve_six.checkpoint import core, progress_trainer, trainer_adapter
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


def _save_without_torch_rng(
    directory: Path, monkeypatch: pytest.MonkeyPatch,
) -> TrainerConfig:
    config = TrainerConfig(
        max_steps=10, seed=703,
        deterministic_algorithms=True, deterministic_warn_only=True,
    )
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config)
    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d02-missing-torch-optout", "width": 3},
        parameter_count=12,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 10},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    rng = dict(core.capture_rng_state())
    rng.pop("torch")
    with monkeypatch.context() as patch:
        patch.setattr(core, "capture_rng_state", lambda: rng)
        trainer_adapter.save_trainer_checkpoint(
            directory, model=source_model, trainer=source, identity=identity,
        )
    core.verify_checkpoint(directory)
    _, state = core._decode_verified_state(core.prepare_checkpoint_load(directory))
    assert "torch" not in state["rng"]
    return config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_optout_failure_restores_policy_when_checkpoint_omits_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: Any,
) -> None:
    ambient = core.capture_rng_state()
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        checkpoint = tmp_path / "sealed-no-torch"
        config = _save_without_torch_rng(checkpoint, monkeypatch)
        model = torch.nn.Linear(3, 3)
        target = Trainer(model, config)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        primary = RuntimeError("interrupted model application")

        def failed_apply(*_args: Any, **_kwargs: Any) -> None:
            torch.use_deterministic_algorithms(False, warn_only=False)
            raise primary

        monkeypatch.setattr(loader, "_apply_model_weights", failed_apply)
        with pytest.raises(RuntimeError) as error:
            loader.load_trainer_checkpoint(
                checkpoint, model=model, trainer=target, restore_rng=False,
            )
        assert error.value is primary
        assert target._failure_reason is not None
        assert target._update_incomplete
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_optout_success_accepts_checkpoint_without_torch_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: Any,
) -> None:
    ambient = core.capture_rng_state()
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        checkpoint = tmp_path / "sealed-optout"
        config = _save_without_torch_rng(checkpoint, monkeypatch)
        model = torch.nn.Linear(3, 3)
        target = Trainer(model, config)
        result = loader.load_trainer_checkpoint(
            checkpoint, model=model, trainer=target, restore_rng=False,
        )
        assert result.manifest["identity"]["step"] == 0
        assert target._failure_reason is None
        assert not target._update_incomplete
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)
