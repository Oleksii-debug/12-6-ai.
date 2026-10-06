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


class LookupDriftAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.mutation: str | None = None
        self.armed = False
        self.apply_calls = 0

    def __getattribute__(self, name: str):
        if name == "load_state_dict":
            armed = object.__getattribute__(self, "armed")
            if armed:
                object.__setattr__(self, "armed", False)
                owner = object.__getattribute__(self, "owner")
                mutation = object.__getattribute__(self, "mutation")
                assert owner is not None
                if mutation == "counter":
                    owner.tokens_seen = 1
                elif mutation == "config":
                    object.__setattr__(
                        owner.config,
                        "learning_rate",
                        owner.config.learning_rate * 2.0,
                    )
                elif mutation == "auxiliary":
                    owner.optimizer.param_groups[0]["lr"] *= 0.5
                else:
                    raise AssertionError(f"unknown lookup mutation: {mutation}")
        return super().__getattribute__(name)

    def load_state_dict(self, state_dict):
        self.apply_calls += 1
        return super().load_state_dict(state_dict)


class LookupRaisingAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.mutate_before_raise = False
        self.armed = False

    def __getattribute__(self, name: str):
        if name == "load_state_dict":
            armed = object.__getattribute__(self, "armed")
            if armed:
                object.__setattr__(self, "armed", False)
                owner = object.__getattribute__(self, "owner")
                assert owner is not None
                if object.__getattribute__(self, "mutate_before_raise"):
                    owner.tokens_seen = 1
                raise RuntimeError("loader lookup exploded")
        return super().__getattribute__(name)


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        ("optimizer-load", "trainer optimizer must provide load_state_dict"),
        ("optimizer-zero-grad", "trainer optimizer must provide zero_grad"),
        ("scheduler-load", "trainer scheduler must provide load_state_dict"),
        ("scaler-load", "trainer gradient scaler must provide load_state_dict"),
    ],
)
def test_direct_restore_missing_interface_is_preapply_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    message: str,
) -> None:
    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="cosine",
        warmup_steps=1,
        gradient_accumulation_steps=1,
        seed=17,
    )
    state = _clean_state(config)
    trainer = Trainer(nn.Linear(3, 2), config)

    if surface == "optimizer-load":
        monkeypatch.setattr(trainer.optimizer, "load_state_dict", None)
    elif surface == "optimizer-zero-grad":
        monkeypatch.setattr(trainer.optimizer, "zero_grad", None)
    elif surface == "scheduler-load":
        assert trainer.scheduler is not None
        monkeypatch.setattr(trainer.scheduler, "load_state_dict", None)
    elif surface == "scaler-load":
        assert trainer.scaler is not None
        monkeypatch.setattr(trainer.scaler, "load_state_dict", None)
    else:
        raise AssertionError(f"unknown restore surface: {surface}")

    with pytest.raises(TrainingStateInvalidError, match=message):
        trainer.load_state_dict(state)

    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)
    assert not trainer.optimizer.state

    monkeypatch.undo()
    trainer.load_state_dict(state)
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    assert (
        trainer.micro_step,
        trainer.optimizer_step,
        trainer.tokens_seen,
    ) == (
        state.micro_step,
        state.optimizer_step,
        state.tokens_seen,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("counter", "restore state changed during loader lookup"),
        ("config", "restore config changed during loader lookup"),
        ("auxiliary", "trainer auxiliary state changed during loader lookup"),
    ],
)
def test_direct_restore_rejects_stable_binding_lookup_drift(
    mutation: str,
    message: str,
) -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(LookupDriftAdamW, config)
    assert isinstance(optimizer, LookupDriftAdamW)
    optimizer.mutation = mutation

    with pytest.raises(TrainingStateInvalidError, match=message):
        trainer.load_state_dict(state)

    assert trainer.optimizer is optimizer
    assert optimizer.apply_calls == 0
    assert trainer._failure_reason is not None
    assert trainer._update_incomplete is False


def test_direct_restore_preserves_lookup_exception_without_drift_and_allows_retry() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(LookupRaisingAdamW, config)
    assert isinstance(optimizer, LookupRaisingAdamW)

    with pytest.raises(RuntimeError, match="loader lookup exploded"):
        trainer.load_state_dict(state)

    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)

    trainer.load_state_dict(state)
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


def test_direct_restore_preserves_lookup_exception_and_poisons_detected_drift() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(LookupRaisingAdamW, config)
    assert isinstance(optimizer, LookupRaisingAdamW)
    optimizer.mutate_before_raise = True

    with pytest.raises(RuntimeError, match="loader lookup exploded"):
        trainer.load_state_dict(state)

    assert trainer.tokens_seen == 1
    assert trainer._failure_reason == "trainer restore state changed during loader lookup"
    assert trainer._update_incomplete is False
