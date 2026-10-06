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
                elif mutation == "policy":
                    owner._canonical_unscheduled_default_optimizer = True
                elif mutation == "device":
                    owner.device = torch.device("meta")
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
        ("policy", "restore policy changed during loader lookup"),
        ("device", "component binding changed during loader lookup"),
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


class ApplyStateDriftAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.mutation: str | None = None
        self.armed = False

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        if self.armed:
            self.armed = False
            assert self.owner is not None
            if self.mutation == "counter":
                self.owner.tokens_seen = 1
            elif self.mutation == "config":
                object.__setattr__(
                    self.owner.config,
                    "learning_rate",
                    self.owner.config.learning_rate * 2.0,
                )
            elif self.mutation == "model":
                with torch.no_grad():
                    self.owner.model.weight.add_(1.0)
            elif self.mutation == "auxiliary":
                self.param_groups[0]["lr"] *= 0.5
            elif self.mutation == "policy":
                self.owner._canonical_unscheduled_default_optimizer = True
            elif self.mutation == "device":
                self.owner.device = torch.device("meta")
            elif self.mutation == "gradient":
                self.owner.model.weight.grad = torch.ones_like(self.owner.model.weight)
            else:
                raise AssertionError(f"unknown apply mutation: {self.mutation}")
        return result


class ZeroGradAuxiliaryDriftAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.armed = False

    def zero_grad(self, *args, **kwargs):
        result = super().zero_grad(*args, **kwargs)
        if self.armed:
            self.armed = False
            self.param_groups[0]["lr"] *= 0.5
        return result


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("counter", "trainer restore counters changed during load"),
        ("config", "trainer restore config changed during load"),
        ("model", "trainer model changed during load"),
        ("auxiliary", "optimizer export hyperparameters differ"),
        ("policy", "trainer restore policy changed during load"),
        ("device", "trainer restore component binding changed during load"),
        ("gradient", "completed optimizer step left residual model gradients"),
    ],
)
def test_direct_restore_rejects_apply_time_state_drift(
    mutation: str,
    message: str,
) -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(ApplyStateDriftAdamW, config)
    assert isinstance(optimizer, ApplyStateDriftAdamW)
    optimizer.mutation = mutation

    with pytest.raises((TrainingStateInvalidError, RuntimeError), match=message):
        trainer.load_state_dict(state)

    assert trainer._failure_reason == (
        "trainer state restore failed after possible partial apply"
    )
    assert trainer._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
    with pytest.raises(TrainingStateInvalidError, match="verified model"):
        trainer.load_state_dict(state)


def test_direct_restore_rejects_zero_grad_auxiliary_drift() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(ZeroGradAuxiliaryDriftAdamW, config)
    assert isinstance(optimizer, ZeroGradAuxiliaryDriftAdamW)

    with pytest.raises(
        TrainingStateInvalidError,
        match="optimizer export hyperparameters differ",
    ):
        trainer.load_state_dict(state)

    assert trainer._failure_reason == (
        "trainer state restore failed after possible partial apply"
    )
    assert trainer._update_incomplete is True


@pytest.mark.parametrize(
    "authority",
    [
        "__getattribute__",
        "__setattr__",
        "_require_finite_committed_update",
    ],
)
def test_direct_restore_rejects_subclass_safety_authority_override(
    authority: str,
) -> None:
    config = _config()
    state = _clean_state(config)

    def passthrough_getattribute(self, name):
        return object.__getattribute__(self, name)

    def passthrough_setattr(self, name, value):
        object.__setattr__(self, name, value)

    if authority == "__getattribute__":
        override = passthrough_getattribute
    elif authority == "__setattr__":
        override = passthrough_setattr
    else:
        def no_op_authority(self):
            return None

        override = no_op_authority

    unsafe_type = type(
        f"DirectRestoreUnsafe_{authority}",
        (Trainer,),
        {authority: override},
    )
    target = unsafe_type(nn.Linear(3, 2), config, scheduler=None)

    with pytest.raises(
        TrainingStateInvalidError,
        match=f"native D02 safety authority must remain canonical: {authority}",
    ):
        Trainer.load_state_dict(target, state)

    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert not target.optimizer.state


def test_direct_restore_rejects_instance_safety_shadow_then_allows_retry() -> None:
    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)

    def no_op_authority():
        return None

    target._require_finite_committed_update = no_op_authority

    with pytest.raises(
        TrainingStateInvalidError,
        match=(
            "native D02 safety authority must remain canonical: "
            "_require_finite_committed_update"
        ),
    ):
        target.load_state_dict(state)

    assert target._failure_reason is None
    assert target._update_incomplete is False
    del vars(target)["_require_finite_committed_update"]

    target.load_state_dict(state)
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_direct_restore_allows_subclass_with_export_only_override() -> None:
    class ExportOnlyTrainer(Trainer):
        def state_dict(self):
            return super().state_dict()

    config = _config()
    state = _clean_state(config)
    target = ExportOnlyTrainer(nn.Linear(3, 2), config, scheduler=None)

    Trainer.load_state_dict(target, state)

    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert (
        target.micro_step,
        target.optimizer_step,
        target.tokens_seen,
    ) == (
        state.micro_step,
        state.optimizer_step,
        state.tokens_seen,
    )


def test_d05_uses_trainer_checkpoint_safety_authority_source() -> None:
    from twelve_six.checkpoint import trainer_adapter

    assert (
        trainer_adapter._NATIVE_D02_CHECKPOINT_SAFETY_AUTHORITIES
        is Trainer._CHECKPOINT_SAFETY_AUTHORITIES
    )


def test_direct_restore_rejects_dict_descriptor_without_dispatch() -> None:
    observed: list[str] = []

    class DictSpoofTrainer(Trainer):
        @property
        def __dict__(self):
            observed.append("__dict__")
            return {}

    config = _config()
    state = _clean_state(config)
    target = DictSpoofTrainer(nn.Linear(3, 2), config, scheduler=None)

    with pytest.raises(
        TrainingStateInvalidError,
        match="native D02 safety authority must remain canonical: __dict__",
    ):
        Trainer.load_state_dict(target, state)

    assert observed == []
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_direct_restore_uses_real_mro_for_safety_lineage() -> None:
    class LyingMroMeta(type):
        def __getattribute__(cls, name):
            if name == "__mro__":
                return (Trainer, object)
            return type.__getattribute__(cls, name)

    class LyingTrainer(Trainer, metaclass=LyingMroMeta):
        def _require_finite_committed_update(self):
            return None

    config = _config()
    state = _clean_state(config)
    target = LyingTrainer(nn.Linear(3, 2), config, scheduler=None)

    with pytest.raises(
        TrainingStateInvalidError,
        match=(
            "native D02 safety authority must remain canonical: "
            "_require_finite_committed_update"
        ),
    ):
        Trainer.load_state_dict(target, state)

    assert target._failure_reason is None
    assert target._update_incomplete is False


class LookupAuthorityMutatingAdamW(AdamW):
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
                assert owner is not None

                def no_op_authority():
                    return None

                vars(owner)["_require_finite_committed_update"] = no_op_authority
        return super().__getattribute__(name)


class ApplyAuthorityMutatingAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.armed = False

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        if self.armed:
            self.armed = False
            assert self.owner is not None

            def suppress_poison(*args, **kwargs):
                return None

            vars(self.owner)["_mark_failed"] = suppress_poison
        return result


def test_direct_restore_rejects_lookup_time_safety_shadow() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(LookupAuthorityMutatingAdamW, config)
    assert isinstance(optimizer, LookupAuthorityMutatingAdamW)

    with pytest.raises(
        TrainingStateInvalidError,
        match="restore safety authority changed during loader lookup",
    ):
        trainer.load_state_dict(state)

    assert "_require_finite_committed_update" in vars(trainer)
    assert trainer._failure_reason == (
        "trainer restore safety authority changed during loader lookup"
    )
    assert trainer._update_incomplete is False


def test_direct_restore_partial_apply_cannot_shadow_poison_authority() -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(ApplyAuthorityMutatingAdamW, config)
    assert isinstance(optimizer, ApplyAuthorityMutatingAdamW)

    with pytest.raises(
        TrainingStateInvalidError,
        match="native D02 safety authority must remain canonical: _mark_failed",
    ):
        trainer.load_state_dict(state)

    assert "_mark_failed" in vars(trainer)
    assert trainer._failure_reason == (
        "trainer state restore failed after possible partial apply"
    )
    assert trainer._update_incomplete is True


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("device", "trainer restore binding fields are unavailable"),
        (
            "_canonical_default_optimizer_options",
            "trainer restore policy fields are unavailable",
        ),
    ],
)
def test_direct_restore_rejects_missing_native_restore_contract_field(
    field: str,
    message: str,
) -> None:
    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    attrs = vars(target)
    saved = attrs.pop(field)

    with pytest.raises(TrainingStateInvalidError, match=message):
        target.load_state_dict(state)

    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert not target.optimizer.state

    attrs[field] = saved
    target.load_state_dict(state)
    assert target._failure_reason is None
    assert target._update_incomplete is False
