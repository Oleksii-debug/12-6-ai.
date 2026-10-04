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
    *,
    source_has_buffer: bool = False,
) -> TrainerConfig:
    config = TrainerConfig(max_steps=3, seed=703)
    source_model = torch.nn.Linear(3, 3)
    if source_has_buffer:
        source_model.register_buffer("resume_scale", torch.tensor(3.0))
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
@pytest.mark.parametrize("missing_from", ["checkpoint", "target"])
def test_canonical_d02_non_strict_restore_must_not_accept_missing_buffer(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    restore_rng: bool,
    missing_from: str,
) -> None:
    checkpoint = tmp_path / "sealed-source"
    config = _sealed_source(
        checkpoint, checkpoint_identity, source_has_buffer=missing_from == "target",
    )
    target_model = torch.nn.Linear(3, 3)
    if missing_from == "checkpoint":
        target_model.register_buffer("resume_scale", torch.tensor(2.0))
    target = Trainer(target_model, config, device="cpu")
    initial_parameters = [p.detach().clone() for p in target_model.parameters()]
    initial_state = {
        name: value.detach().clone() for name, value in target_model.state_dict().items()
    }

    # A sealed checkpoint and fresh target may have mismatched persistent
    # buffers in either direction while model parameters still match exactly.
    # PyTorch strict=False accepts both, but exact D02 replay must not.
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
    for name, before in initial_state.items():
        torch.testing.assert_close(target_model.state_dict()[name], before, rtol=0, atol=0)
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


class _GenericTrainer:
    """Minimal non-D02 state owner retaining the public permissive API."""

    def state_dict(self) -> dict[str, Any]:
        return {
            "micro_step": 0,
            "optimizer_step": 0,
            "tokens_seen": 0,
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "config": {"kind": "generic-permissive"},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.restored = dict(state)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_generic_adapter_non_strict_remains_permissive(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    checkpoint = tmp_path / "sealed-generic"
    source_model = torch.nn.Linear(3, 3)
    trainer_adapter.save_trainer_checkpoint(
        checkpoint, model=source_model, trainer=_GenericTrainer(),
        identity=checkpoint_identity,
    )
    target_model = torch.nn.Linear(3, 3)
    target_model.register_buffer("extra", torch.tensor(9.0))
    trainer = _GenericTrainer()
    loader.load_trainer_checkpoint(
        checkpoint, model=target_model, trainer=trainer,
        strict_model=False, restore_rng=False,
    )
    assert trainer.restored["optimizer_step"] == 0
    torch.testing.assert_close(target_model.extra, torch.tensor(9.0), rtol=0, atol=0)
    for name, weight in source_model.state_dict().items():
        torch.testing.assert_close(target_model.state_dict()[name], weight, rtol=0, atol=0)
