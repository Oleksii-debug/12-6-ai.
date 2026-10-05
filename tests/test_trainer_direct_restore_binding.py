from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from twelve_six.training import (
    Trainer,
    TrainerConfig,
    TrainingStateInvalidError,
    build_optimizer,
)


class LookupMutatingAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.armed = False

    def __getattribute__(self, name: str):
        if name == "load_state_dict":
            armed = object.__getattribute__(self, "armed")
            if armed:
                object.__setattr__(self, "armed", False)
                owner = object.__getattribute__(self, "owner")
                replacement = object.__getattribute__(self, "replacement")
                assert owner is not None
                assert replacement is not None
                owner.optimizer = replacement
        return super().__getattribute__(name)


class LookupStateMutatingAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.armed = False

    def __getattribute__(self, name: str):
        if name == "load_state_dict":
            armed = object.__getattribute__(self, "armed")
            if armed:
                object.__setattr__(self, "armed", False)
                owner = object.__getattribute__(self, "owner")
                assert owner is not None
                owner._failure_reason = "descriptor poisoned target"
        return super().__getattribute__(name)


class LookupModelMutatingAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.armed = False

    def __getattribute__(self, name: str):
        if name == "load_state_dict":
            armed = object.__getattribute__(self, "armed")
            if armed:
                object.__setattr__(self, "armed", False)
                owner = object.__getattribute__(self, "owner")
                assert owner is not None
                owner.model.weight = nn.Parameter(owner.model.weight.detach().clone() + 1.0)
        return super().__getattribute__(name)


class ApplyMutatingAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.armed = False

    def load_state_dict(self, state_dict):
        if self.armed:
            self.armed = False
            assert self.owner is not None
            assert self.replacement is not None
            self.owner.optimizer = self.replacement
        return super().load_state_dict(state_dict)


def _config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="constant",
        gradient_accumulation_steps=1,
        seed=17,
    )


def _clean_state(config: TrainerConfig):
    source_model = nn.Linear(3, 2)
    source = Trainer(source_model, config)
    return source.state_dict()


def _target_with_optimizer(
    optimizer_type: type[AdamW],
    config: TrainerConfig,
) -> tuple[Trainer, AdamW]:
    model = nn.Linear(3, 2)
    optimizer = optimizer_type(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    trainer = Trainer(model, config, optimizer=optimizer, scheduler=None)
    replacement = build_optimizer(model, config)
    optimizer.owner = trainer
    optimizer.replacement = replacement
    optimizer.armed = True
    return trainer, optimizer


def test_direct_restore_rejects_optimizer_rebind_during_loader_lookup() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(LookupMutatingAdamW, config)

    with pytest.raises(
        TrainingStateInvalidError,
        match="component binding changed during loader lookup",
    ):
        trainer.load_state_dict(state)

    assert trainer.optimizer is optimizer.replacement
    assert trainer._failure_reason is not None
    assert trainer._update_incomplete is False
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)


def test_direct_restore_rejects_optimizer_rebind_during_loader_apply() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(ApplyMutatingAdamW, config)

    with pytest.raises(
        TrainingStateInvalidError,
        match="component binding changed during load",
    ):
        trainer.load_state_dict(state)

    assert trainer.optimizer is optimizer.replacement
    assert trainer._failure_reason is not None
    assert trainer._update_incomplete is True
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)


def test_direct_restore_rejects_poison_marker_drift_during_loader_lookup() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(LookupStateMutatingAdamW, config)

    with pytest.raises(
        TrainingStateInvalidError,
        match="restore state changed during loader lookup",
    ):
        trainer.load_state_dict(state)

    assert trainer.optimizer is optimizer
    assert trainer._failure_reason == "descriptor poisoned target"
    assert trainer._update_incomplete is False
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)


def test_direct_restore_rejects_model_drift_during_loader_lookup() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(LookupModelMutatingAdamW, config)
    before = trainer.model.weight.detach().clone()

    with pytest.raises(
        TrainingStateInvalidError,
        match="model changed during loader lookup",
    ):
        trainer.load_state_dict(state)

    assert trainer.optimizer is optimizer
    assert not torch.equal(trainer.model.weight.detach(), before)
    assert trainer._failure_reason is not None
    assert trainer._update_incomplete is False
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)
