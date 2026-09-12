from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.optim import SGD

from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import (
    Trainer,
    TrainerState,
    TrainingStateInvalidError,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("resume integrity tests must not execute model forward")


class CleanupFailOptimizer(SGD):
    def __init__(self, params) -> None:
        super().__init__(params, lr=0.1, momentum=0.9)
        self.fail_cleanup = False
        self.fail_rollback = False
        self.cleanup_failed = False

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.fail_cleanup:
            self.fail_cleanup = False
            first_parameter = self.param_groups[0]["params"][0]
            first_parameter.grad = None
            self.cleanup_failed = True
            raise RuntimeError("injected resume cleanup failure")
        super().zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state_dict) -> None:
        if self.fail_rollback and self.cleanup_failed:
            raise RuntimeError("injected resume rollback failure")
        super().load_state_dict(state_dict)


def _trainer(*, optimizer_cls=SGD) -> Trainer:
    model = TinyModel()
    optimizer = optimizer_cls(model.parameters(), lr=0.1, momentum=0.9)
    return Trainer(
        model,
        TrainerConfig(max_steps=4),
        optimizer=optimizer,
        device="cpu",
    )


def _state(trainer: Trainer) -> TrainerState:
    state = trainer.state_dict()
    assert isinstance(state, TrainerState)
    return state


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("micro_step", False),
        ("micro_step", 0.0),
        ("optimizer_step", False),
        ("optimizer_step", 0.0),
        ("tokens_seen", False),
        ("tokens_seen", 0.0),
    ],
)
def test_resume_rejects_numeric_alias_counters_before_mutation(
    field: str,
    bad_value: object,
) -> None:
    trainer = _trainer()
    state = _state(trainer)
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())

    with pytest.raises(ValueError, match="must be an exact integer"):
        trainer.load_state_dict(replace(state, **{field: bad_value}))

    assert trainer.optimizer.state_dict() == optimizer_before
    assert trainer.micro_step == 0
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_steps", 4.0),
        ("deterministic_algorithms", 1),
        ("learning_rate", 0.0003 if False else 3e-4),
    ],
)
def test_resume_config_identity_is_type_aware(field: str, bad_value: object) -> None:
    trainer = _trainer()
    state = _state(trainer)
    config = copy.deepcopy(state.config)
    if field == "learning_rate":
        config[field] = int(config[field] == 3e-4)
    else:
        config[field] = bad_value

    with pytest.raises(ValueError, match="trainer config mismatch"):
        trainer.load_state_dict(replace(state, config=config))

    assert trainer.micro_step == 0
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0


def test_resume_cleanup_failure_rolls_back_components_and_gradients() -> None:
    trainer = _trainer(optimizer_cls=CleanupFailOptimizer)
    state = _state(trainer)
    parameter = next(trainer.model.parameters())
    parameter.grad = torch.tensor([7.0, 8.0])
    gradient_before = parameter.grad.detach().clone()
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    trainer.optimizer.fail_cleanup = True

    with pytest.raises(RuntimeError, match="injected resume cleanup failure"):
        trainer.load_state_dict(state)

    assert trainer.optimizer.state_dict() == optimizer_before
    assert parameter.grad is not None
    torch.testing.assert_close(parameter.grad, gradient_before)
    assert trainer.micro_step == 0
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0
    trainer.assert_checkpoint_safe()


def test_resume_cleanup_with_unprovable_rollback_poisons_trainer() -> None:
    trainer = _trainer(optimizer_cls=CleanupFailOptimizer)
    state = _state(trainer)
    parameter = next(trainer.model.parameters())
    parameter.grad = torch.tensor([3.0, 4.0])
    trainer.optimizer.fail_cleanup = True
    trainer.optimizer.fail_rollback = True

    with pytest.raises(TrainingStateInvalidError, match="rollback could not prove a clean state"):
        trainer.load_state_dict(state)

    with pytest.raises(TrainingStateInvalidError, match="trainer state is invalid"):
        trainer.assert_checkpoint_safe()


def test_valid_exact_resume_remains_accepted() -> None:
    source = _trainer()
    state = _state(source)
    target = _trainer()
    parameter = next(target.model.parameters())
    parameter.grad = torch.tensor([5.0, 6.0])

    target.load_state_dict(state)

    assert target.micro_step == state.micro_step
    assert target.optimizer_step == state.optimizer_step
    assert target.tokens_seen == state.tokens_seen
    assert parameter.grad is None
    target.assert_checkpoint_safe()
