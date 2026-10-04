"""D02 must not credit exact checkpoint recovery from partially restored weights.

Synthetic CPU acceptance cases only; not physical model-training evidence.
"""

from __future__ import annotations

from dataclasses import replace
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
@pytest.mark.parametrize("with_buffer", [False, True], ids=["parameters", "and-buffer"])
def test_canonical_d02_non_strict_mode_remains_valid_for_complete_model(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    with_buffer: bool,
) -> None:
    checkpoint = tmp_path / "sealed-complete"
    config = _sealed_source(
        checkpoint, checkpoint_identity, source_has_buffer=with_buffer,
    )
    saved_weights, _ = core._decode_verified_state(core.prepare_checkpoint_load(checkpoint))
    target_model = torch.nn.Linear(3, 3)
    if with_buffer:
        target_model.register_buffer("resume_scale", torch.tensor(-2.0))
    target = Trainer(target_model, config, device="cpu")
    result = loader.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target,
        strict_model=False,
        restore_rng=False,
    )
    assert result.manifest["identity"]["step"] == 0
    for name, saved in saved_weights.items():
        torch.testing.assert_close(
            target_model.state_dict()[name].detach().cpu(),
            torch.from_numpy(saved), rtol=0, atol=0,
        )
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


def _checkpoint_after_tiny_trainer_step(
    checkpoint: Path,
    identity: CheckpointIdentity,
    *,
    with_buffer: bool,
) -> tuple[torch.nn.Embedding, Trainer, TrainerConfig]:
    """Persist D02's genuine transition on tiny synthetic IDs, not corpus evidence."""
    config = TrainerConfig(max_steps=3, seed=703)
    # Exactly 12 trainable parameters and 3 output logits. This permits a
    # real D02 optimizer transition without consuming any project data.
    model = torch.nn.Embedding(4, 3)
    if with_buffer:
        model.register_buffer("resume_scale", torch.tensor(3.0))
    trainer = Trainer(model, config, device="cpu")
    metrics = trainer.train_microbatch(
        {"input_ids": torch.tensor([[0, 1, 2, 0]], dtype=torch.long)},
    )
    assert metrics.optimizer_stepped
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 3)
    trainer.assert_checkpoint_safe()
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=replace(identity, step=1, tokens_seen=3),
    )
    core.verify_checkpoint(checkpoint)
    assert any(trainer.optimizer.state.values())
    return model, trainer, config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [True, False], ids=["replay", "opt-out"])
@pytest.mark.parametrize("missing_from", ["checkpoint", "target"])
def test_nonzero_adamw_resume_rejects_incomplete_persistent_model(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    restore_rng: bool,
    missing_from: str,
) -> None:
    checkpoint = tmp_path / "committed-checkpoint"
    _, source, config = _checkpoint_after_tiny_trainer_step(
        checkpoint,
        checkpoint_identity,
        with_buffer=missing_from == "target",
    )
    assert source.optimizer_step == 1
    target_model = torch.nn.Embedding(4, 3)
    if missing_from == "checkpoint":
        target_model.register_buffer("resume_scale", torch.tensor(-2.0))
    target = Trainer(target_model, config, device="cpu")
    initial = {
        name: value.detach().clone()
        for name, value in target_model.state_dict().items()
    }
    extra = {"expected_step": 1, "expected_tokens_seen": 3} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(CheckpointCompatibilityError, match="state_dict|model|key"):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )
    for name, before in initial.items():
        torch.testing.assert_close(target_model.state_dict()[name], before, rtol=0, atol=0)
    assert not target.optimizer.state
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("with_buffer", [False, True], ids=["parameters", "and-buffer"])
def test_nonzero_adamw_complete_resume_preserves_next_optimizer_update(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    with_buffer: bool,
) -> None:
    checkpoint = tmp_path / "committed-complete"
    source_model, source, config = _checkpoint_after_tiny_trainer_step(
        checkpoint, checkpoint_identity, with_buffer=with_buffer,
    )
    target_model = torch.nn.Embedding(4, 3)
    if with_buffer:
        target_model.register_buffer("resume_scale", torch.tensor(-2.0))
    target = Trainer(target_model, config, device="cpu")
    extra = {"expected_step": 1, "expected_tokens_seen": 3} if (
        loader is progress_trainer
    ) else {}
    loader.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target,
        strict_model=False,
        restore_rng=True,
        **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 3)
    assert target._failure_reason is None
    assert target._update_incomplete is False
    for name, source_tensor in source_model.state_dict().items():
        torch.testing.assert_close(
            target_model.state_dict()[name], source_tensor, rtol=0, atol=0,
        )
    for source_param, target_param in zip(
        source_model.parameters(), target_model.parameters(), strict=True,
    ):
        source_state = source.optimizer.state[source_param]
        target_state = target.optimizer.state[target_param]
        assert source_state.keys() == target_state.keys()
        assert "exp_avg" in source_state and "exp_avg_sq" in source_state
        for key in source_state:
            torch.testing.assert_close(target_state[key], source_state[key], rtol=0, atol=0)
    # Advance through the real Trainer API on the same synthetic batch.
    batch = {"input_ids": torch.tensor([[0, 1, 2, 0]], dtype=torch.long)}
    source_metrics = source.train_microbatch(batch)
    target_metrics = target.train_microbatch(batch)
    assert source_metrics.optimizer_stepped and target_metrics.optimizer_stepped
    assert (source.micro_step, source.optimizer_step, source.tokens_seen) == (2, 2, 6)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (2, 2, 6)
    for source_param, target_param in zip(
        source_model.parameters(), target_model.parameters(), strict=True,
    ):
        torch.testing.assert_close(target_param, source_param, rtol=0, atol=0)
    target.assert_checkpoint_safe()
