from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import (
    NonFiniteTrainingError,
    Trainer,
    TrainingStateInvalidError,
)


class _TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 4) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, sequence = input_ids.shape
        return self.logits.view(1, 1, -1).expand(batch, sequence, -1)


def _trainer(*, max_steps: int = 1) -> Trainer:
    return Trainer(
        _TinyLanguageModel(),
        TrainerConfig(max_steps=max_steps, gradient_clip_norm=1.0, precision="fp32"),
    )


def _install_finite_adamw_state(trainer: Trainer) -> dict[str, torch.Tensor]:
    parameter = next(iter(trainer.model.parameters()))
    state = trainer.optimizer.state[parameter]
    state["step"] = torch.tensor(1.0)
    state["exp_avg"] = torch.zeros_like(parameter)
    state["exp_avg_sq"] = torch.zeros_like(parameter)
    return state


@pytest.mark.parametrize(
    ("field", "poison"),
    [
        ("exp_avg", float("nan")),
        ("exp_avg_sq", float("inf")),
    ],
)
def test_checkpoint_publication_rejects_nonfinite_optimizer_state(
    field: str,
    poison: float,
) -> None:
    trainer = _trainer()
    optimizer_state = _install_finite_adamw_state(trainer)
    optimizer_state[field].view(-1)[0] = poison

    with pytest.raises(
        NonFiniteTrainingError,
        match="non-finite optimizer state blocks checkpoint publication",
    ):
        trainer.state_dict()

    with pytest.raises(TrainingStateInvalidError):
        trainer.assert_checkpoint_safe()


def test_checkpoint_publication_rejects_nonfinite_optimizer_param_group() -> None:
    trainer = _trainer()
    trainer.optimizer.param_groups[0]["lr"] = float("inf")

    with pytest.raises(
        NonFiniteTrainingError,
        match="non-finite optimizer state blocks checkpoint publication",
    ):
        trainer.state_dict()


def test_clean_finite_optimizer_state_save_and_restore_passes() -> None:
    source = _trainer()
    _install_finite_adamw_state(source)
    checkpoint = source.state_dict()

    destination = _trainer()
    destination.load_state_dict(checkpoint)

    assert destination.state_dict().optimizer["state"]
    assert destination.optimizer_step == 0
    assert destination.micro_step == 0


@pytest.mark.parametrize(
    ("field", "poison"),
    [
        ("exp_avg", float("nan")),
        ("exp_avg_sq", float("inf")),
    ],
)
def test_restore_rejects_rehashed_nonfinite_optimizer_payload_without_mutation(
    field: str,
    poison: float,
) -> None:
    source = _trainer()
    _install_finite_adamw_state(source)
    checkpoint = copy.deepcopy(source.state_dict())
    optimizer_state = next(iter(checkpoint.optimizer["state"].values()))
    optimizer_state[field].view(-1)[0] = poison

    destination = _trainer()
    before = copy.deepcopy(destination.optimizer.state_dict())

    with pytest.raises(
        NonFiniteTrainingError,
        match="trainer checkpoint optimizer state contains NaN/Inf",
    ):
        destination.load_state_dict(checkpoint)

    assert destination.optimizer.state_dict() == before
    assert destination.optimizer_step == 0
    assert destination.micro_step == 0
    destination.assert_checkpoint_safe()


def test_restore_cannot_repair_live_nonfinite_trainer_in_place() -> None:
    source = _trainer()
    _install_finite_adamw_state(source)
    checkpoint = source.state_dict()

    destination = _trainer()
    optimizer_state = _install_finite_adamw_state(destination)
    optimizer_state["exp_avg"].view(-1)[0] = float("nan")

    with pytest.raises(
        TrainingStateInvalidError,
        match="non-finite live Trainer cannot be repaired in place",
    ):
        destination.load_state_dict(checkpoint)

    with pytest.raises(TrainingStateInvalidError):
        destination.assert_checkpoint_safe()


def test_post_step_optimizer_poison_never_commits_trainer_step() -> None:
    trainer = _trainer()
    original_step = trainer.optimizer.step

    def poisoning_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        optimizer_state = next(iter(trainer.optimizer.state.values()))
        optimizer_state["exp_avg_sq"].view(-1)[0] = float("inf")
        return result

    trainer.optimizer.step = poisoning_step  # type: ignore[method-assign]
    batch = {"input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long)}

    with pytest.raises(
        NonFiniteTrainingError,
        match="non-finite optimizer state at micro_step=1",
    ):
        trainer.train_microbatch(batch)

    assert trainer.optimizer_step == 0
    assert trainer.micro_step == 1
    assert trainer.tokens_seen == 2
    with pytest.raises(TrainingStateInvalidError):
        trainer.train_microbatch(batch)
