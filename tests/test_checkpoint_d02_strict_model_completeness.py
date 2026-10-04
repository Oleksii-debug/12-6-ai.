"""D02 must not credit exact checkpoint recovery from partially restored weights.

Synthetic CPU acceptance cases only; not physical model-training evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIdentity
from twelve_six.checkpoint import core, progress_trainer, trainer_adapter
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


@pytest.fixture
def checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d02-model-completeness", "width": 3},
        parameter_count=12,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _sealed_source(
    checkpoint: Path,
    identity: CheckpointIdentity,
) -> TrainerConfig:
    config = TrainerConfig(max_steps=3, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config, device="cpu")
    trainer_adapter.save_trainer_checkpoint(
        checkpoint, model=source_model, trainer=source, identity=identity,
    )
    core.verify_checkpoint(checkpoint)
    return config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [True, False], ids=["replay", "opt-out"])
def test_canonical_d02_non_strict_restore_must_not_accept_missing_buffer(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    restore_rng: bool,
) -> None:
    checkpoint = tmp_path / "sealed-source"
    config = _sealed_source(checkpoint, checkpoint_identity)
    target_model = torch.nn.Linear(3, 3)
    target_model.register_buffer("resume_scale", torch.tensor(2.0))
    target = Trainer(target_model, config, device="cpu")
    initial_parameters = [p.detach().clone() for p in target_model.parameters()]
    initial_scale = target_model.resume_scale.detach().clone()

    # A valid checkpoint has all of the source's parameters, but the target
    # also has a persistent buffer absent from the sealed source model state.
    # PyTorch strict=False silently leaves that buffer at its live initializer.
    # Exact D02 resume must reject this *before* changing model/trainer state.
    with pytest.raises(CheckpointCompatibilityError, match="state_dict|model|key"):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
        )
    for current, before in zip(target_model.parameters(), initial_parameters, strict=True):
        torch.testing.assert_close(current.detach(), before, rtol=0, atol=0)
    torch.testing.assert_close(target_model.resume_scale, initial_scale, rtol=0, atol=0)
    assert target.optimizer_step == 0
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_canonical_d02_non_strict_mode_remains_valid_for_complete_model(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    checkpoint = tmp_path / "sealed-complete"
    config = _sealed_source(checkpoint, checkpoint_identity)
    target_model = torch.nn.Linear(3, 3)
    target = Trainer(target_model, config, device="cpu")
    result = loader.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target,
        strict_model=False,
        restore_rng=False,
    )
    assert result.manifest["identity"]["step"] == 0
    assert target._failure_reason is None
    assert target._update_incomplete is False
