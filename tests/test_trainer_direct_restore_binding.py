from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

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


class _PayloadOwnershipSafetyShadow:
    deepcopy_calls = 0

    def __init__(self, owner: Trainer) -> None:
        self.owner = owner

    def __deepcopy__(self, memo: dict[int, object]) -> int:
        del memo
        type(self).deepcopy_calls += 1

        def forged_storage():
            del vars(self.owner)["_canonical_optimizer_storage"]
            return {}, []

        self.owner._canonical_optimizer_storage = forged_storage
        return 0


def test_direct_restore_seals_each_payload_ownership_phase() -> None:
    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    shadow = _PayloadOwnershipSafetyShadow(target)
    _PayloadOwnershipSafetyShadow.deepcopy_calls = 0
    state.optimizer["payload_shadow"] = shadow

    with pytest.raises(
        TrainingStateInvalidError,
        match=(
            "trainer restore safety authority changed during "
            "optimizer payload ownership"
        ),
    ):
        target.load_state_dict(state)

    assert _PayloadOwnershipSafetyShadow.deepcopy_calls == 1
    assert "_canonical_optimizer_storage" in vars(target)
    assert target._failure_reason == (
        "trainer restore safety authority changed during "
        "optimizer payload ownership"
    )
    assert target._update_incomplete is True
    assert not target.optimizer.state


class _RestoreConfigDriftMapping(Mapping[str, object]):
    def __init__(
        self,
        payload: dict[str, object],
        owner: Trainer,
    ) -> None:
        self.payload = payload
        self.owner = owner
        self.iterations = 0

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        object.__setattr__(
            self.owner.config,
            "gradient_accumulation_steps",
            2,
        )
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    def __getitem__(self, key: str) -> object:
        return self.payload[key]


class _RestoreConfigDriftReset:
    deepcopy_calls = 0

    def __init__(self, owner: Trainer) -> None:
        self.owner = owner

    def __deepcopy__(self, memo: dict[int, object]) -> int:
        del memo
        type(self).deepcopy_calls += 1
        object.__setattr__(
            self.owner.config,
            "gradient_accumulation_steps",
            1,
        )
        return 0


def test_direct_restore_uses_entry_config_for_counter_preflight() -> None:
    config = _config()
    state = _clean_state(config)
    state.micro_step = 2
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    reset = _RestoreConfigDriftReset(target)
    _RestoreConfigDriftReset.deepcopy_calls = 0
    state.optimizer["restore_config_after_counter_check"] = reset
    payload = _RestoreConfigDriftMapping(
        {
            "micro_step": state.micro_step,
            "optimizer_step": state.optimizer_step,
            "tokens_seen": state.tokens_seen,
            "optimizer": state.optimizer,
            "scheduler": state.scheduler,
            "scaler": state.scaler,
            "config": state.config,
        },
        target,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint is not at a complete committed accumulation boundary",
    ):
        target.load_state_dict(payload)

    assert payload.iterations >= 1
    assert _RestoreConfigDriftReset.deepcopy_calls == 0
    assert target.config.gradient_accumulation_steps == 2
    assert target._failure_reason == (
        "trainer restore config changed during checkpoint preflight"
    )
    assert target._update_incomplete is True
    assert not target.optimizer.state


class _ForbiddenRestoreControlValue:
    bool_calls = 0
    eq_calls = 0
    deepcopy_calls = 0

    def __bool__(self) -> bool:
        type(self).bool_calls += 1
        return False

    def __eq__(self, other: object) -> bool:
        del other
        type(self).eq_calls += 1
        return True

    def __deepcopy__(self, memo: dict[int, object]) -> "_ForbiddenRestoreControlValue":
        del memo
        type(self).deepcopy_calls += 1
        return self


@pytest.mark.parametrize(
    "field",
    ["_update_incomplete", "micro_step", "_pending_loss_sum"],
)
def test_direct_restore_rejects_noncanonical_fresh_control_without_callbacks(
    field: str,
) -> None:
    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    value = _ForbiddenRestoreControlValue()
    _ForbiddenRestoreControlValue.bool_calls = 0
    _ForbiddenRestoreControlValue.eq_calls = 0
    _ForbiddenRestoreControlValue.deepcopy_calls = 0
    setattr(target, field, value)

    with pytest.raises(TrainingStateInvalidError):
        target.load_state_dict(state)

    assert _ForbiddenRestoreControlValue.bool_calls == 0
    assert _ForbiddenRestoreControlValue.eq_calls == 0
    assert _ForbiddenRestoreControlValue.deepcopy_calls == 0
    assert target._failure_reason is None
    assert not target.optimizer.state


class _ForbiddenRestoreContractValue:
    deepcopy_calls = 0
    eq_calls = 0

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> "_ForbiddenRestoreContractValue":
        del memo
        type(self).deepcopy_calls += 1
        return self

    def __eq__(self, other: object) -> bool:
        del other
        type(self).eq_calls += 1
        return True


def test_direct_restore_rejects_noncanonical_config_without_callbacks() -> None:
    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    value = _ForbiddenRestoreContractValue()
    _ForbiddenRestoreContractValue.deepcopy_calls = 0
    _ForbiddenRestoreContractValue.eq_calls = 0
    object.__setattr__(target.config, "seed", value)

    with pytest.raises(
        TrainingStateInvalidError,
        match="checkpoint config field seed contains non-canonical value type",
    ):
        target.load_state_dict(state)

    assert _ForbiddenRestoreContractValue.deepcopy_calls == 0
    assert _ForbiddenRestoreContractValue.eq_calls == 0
    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert not target.optimizer.state


def test_direct_restore_rejects_noncanonical_policy_without_callbacks() -> None:
    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    value = _ForbiddenRestoreContractValue()
    _ForbiddenRestoreContractValue.deepcopy_calls = 0
    _ForbiddenRestoreContractValue.eq_calls = 0
    target._canonical_default_optimizer_options["eps"] = value

    with pytest.raises(
        TrainingStateInvalidError,
        match="trainer restore policy .* contains non-canonical value type",
    ):
        target.load_state_dict(state)

    assert _ForbiddenRestoreContractValue.deepcopy_calls == 0
    assert _ForbiddenRestoreContractValue.eq_calls == 0
    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert not target.optimizer.state


def test_direct_restore_requires_training_mode_and_allows_retry() -> None:
    config = _config()
    state = _clean_state(config)
    model = nn.Linear(3, 2)
    target = Trainer(model, config, scheduler=None)
    model.eval()

    with pytest.raises(
        TrainingStateInvalidError,
        match="checkpoint model training mode must remain enabled",
    ):
        target.load_state_dict(state)

    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert not target.optimizer.state

    model.train()
    target.load_state_dict(state)
    assert target._failure_reason is None
    assert target._update_incomplete is False


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


class LookupNonCallableDriftAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.mutation: str | None = None
        self.armed = False

    def __getattribute__(self, name: str):
        if name == "load_state_dict":
            armed = object.__getattribute__(self, "armed")
            if armed:
                object.__setattr__(self, "armed", False)
                owner = object.__getattribute__(self, "owner")
                mutation = object.__getattribute__(self, "mutation")
                assert owner is not None
                if mutation == "model":
                    owner.model.weight = nn.Parameter(
                        owner.model.weight.detach().clone() + 1.0
                    )
                elif mutation == "counter":
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
                    raise AssertionError(
                        f"unknown non-callable lookup mutation: {mutation}"
                    )
                return None
        return super().__getattribute__(name)


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "model changed during loader lookup"),
        ("counter", "restore state changed during loader lookup"),
        ("config", "restore config changed during loader lookup"),
        ("auxiliary", "trainer auxiliary state changed during loader lookup"),
    ],
)
def test_direct_restore_poisons_noncallable_lookup_drift_before_interface_error(
    mutation: str,
    message: str,
) -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(
        LookupNonCallableDriftAdamW,
        config,
    )
    assert isinstance(optimizer, LookupNonCallableDriftAdamW)
    optimizer.mutation = mutation

    with pytest.raises(TrainingStateInvalidError, match=message):
        trainer.load_state_dict(state)

    assert trainer.optimizer is optimizer
    assert trainer._failure_reason is not None
    assert trainer._update_incomplete is False

    with pytest.raises(
        TrainingStateInvalidError,
        match="failed trainer cannot be repaired in place",
    ):
        trainer.load_state_dict(state)


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


class ApplyReproducibilityDriftAdamW(AdamW):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Trainer | None = None
        self.replacement: AdamW | None = None
        self.mutation: str | None = None
        self.armed = False
        self.guarded_zero_grad_calls = 0

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        if self.armed:
            assert self.owner is not None
            if self.mutation == "determinism":
                torch.use_deterministic_algorithms(
                    not self.owner.config.deterministic_algorithms,
                    warn_only=self.owner.config.deterministic_warn_only,
                )
            elif self.mutation == "training_mode":
                self.owner.model.eval()
            elif self.mutation == "grad_mode":
                torch.set_grad_enabled(not torch.is_grad_enabled())
            else:
                raise AssertionError(
                    f"unknown reproducibility mutation: {self.mutation}"
                )
        return result

    def zero_grad(self, *args, **kwargs):
        if self.armed:
            self.guarded_zero_grad_calls += 1
        return super().zero_grad(*args, **kwargs)


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("determinism", "live PyTorch deterministic policy disagrees"),
        ("training_mode", "checkpoint model training mode must remain enabled"),
        ("grad_mode", "trainer autograd mode changed during load"),
    ],
)
def test_direct_restore_rejects_post_component_reproducibility_drift_before_zero_grad(
    mutation: str,
    message: str,
) -> None:
    config = _config()
    state = _clean_state(config)
    trainer, optimizer = _target_with_optimizer(
        ApplyReproducibilityDriftAdamW,
        config,
    )
    assert isinstance(optimizer, ApplyReproducibilityDriftAdamW)
    optimizer.mutation = mutation
    expected_enabled = torch.are_deterministic_algorithms_enabled()
    expected_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    expected_grad_enabled = torch.is_grad_enabled()

    try:
        with pytest.raises(TrainingStateInvalidError, match=message):
            trainer.load_state_dict(state)
    finally:
        torch.use_deterministic_algorithms(
            expected_enabled,
            warn_only=expected_warn_only,
        )
        torch.set_grad_enabled(expected_grad_enabled)

    assert optimizer.guarded_zero_grad_calls == 0
    assert trainer._failure_reason == (
        "trainer state restore failed after possible partial apply"
    )
    assert trainer._update_incomplete is True


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


class FinalValidationAuxiliaryObserver:
    owner: Trainer | None = None
    armed = False

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).armed = False
            owner = type(self).owner
            assert owner is not None
            owner.optimizer.param_groups[0]["lr"] *= 0.5
        return type(other) is type(self)


class FinalValidationMutatingLambdaLR(LambdaLR):
    owner: Trainer | None = None
    armed = False

    def __init__(self, optimizer: AdamW) -> None:
        super().__init__(optimizer, lr_lambda=lambda _step: 1.0)
        self.observer = FinalValidationAuxiliaryObserver()

    def state_dict(self):
        result = super().state_dict()
        if type(self).armed:
            type(self).armed = False
            owner = type(self).owner
            assert owner is not None
            with torch.no_grad():
                owner.model.weight.add_(1.0)
        return result


def _target_with_final_validation_scheduler(
    config: TrainerConfig,
) -> tuple[Trainer, FinalValidationMutatingLambdaLR]:
    model = nn.Linear(3, 2)
    optimizer = build_optimizer(model, config)
    scheduler = FinalValidationMutatingLambdaLR(optimizer)
    trainer = Trainer(
        model,
        config,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return trainer, scheduler


def _state_with_final_validation_scheduler(config: TrainerConfig):
    trainer, _ = _target_with_final_validation_scheduler(config)
    return trainer.state_dict()


def test_direct_restore_final_observer_cannot_mutate_model() -> None:
    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="cosine",
        warmup_steps=1,
        gradient_accumulation_steps=1,
        seed=17,
    )
    FinalValidationMutatingLambdaLR.owner = None
    FinalValidationMutatingLambdaLR.armed = False
    FinalValidationAuxiliaryObserver.owner = None
    FinalValidationAuxiliaryObserver.armed = False
    state = _state_with_final_validation_scheduler(config)
    trainer, _ = _target_with_final_validation_scheduler(config)
    before = trainer.model.weight.detach().clone()

    FinalValidationMutatingLambdaLR.owner = trainer
    FinalValidationMutatingLambdaLR.armed = True
    try:
        with pytest.raises(
            TrainingStateInvalidError,
            match="trainer model changed during final restore validation",
        ):
            trainer.load_state_dict(state)
    finally:
        FinalValidationMutatingLambdaLR.owner = None
        FinalValidationMutatingLambdaLR.armed = False

    assert not torch.equal(trainer.model.weight.detach(), before)
    assert trainer._failure_reason == (
        "trainer state restore failed after possible partial apply"
    )
    assert trainer._update_incomplete is True


def test_direct_restore_final_observer_cannot_mutate_auxiliary_state() -> None:
    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="cosine",
        warmup_steps=1,
        gradient_accumulation_steps=1,
        seed=17,
    )
    FinalValidationMutatingLambdaLR.owner = None
    FinalValidationMutatingLambdaLR.armed = False
    FinalValidationAuxiliaryObserver.owner = None
    FinalValidationAuxiliaryObserver.armed = False
    state = _state_with_final_validation_scheduler(config)
    trainer, _ = _target_with_final_validation_scheduler(config)

    FinalValidationAuxiliaryObserver.owner = trainer
    FinalValidationAuxiliaryObserver.armed = True
    try:
        with pytest.raises(
            TrainingStateInvalidError,
            match="trainer auxiliary state changed during final restore validation",
        ):
            trainer.load_state_dict(state)
    finally:
        FinalValidationAuxiliaryObserver.owner = None
        FinalValidationAuxiliaryObserver.armed = False

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
    assert (
        trainer_adapter._NATIVE_D02_CHECKPOINT_STORAGE_FIELDS
        is Trainer._CHECKPOINT_STORAGE_FIELDS
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


def test_direct_restore_rejects_effectful_state_deepcopy_before_loader_lookup() -> None:
    from dataclasses import replace

    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="cosine",
        warmup_steps=1,
        gradient_accumulation_steps=1,
        seed=17,
    )
    state = _clean_state(config)
    assert state.scheduler is not None
    target = Trainer(nn.Linear(3, 2), config)
    before = target.model.weight.detach().clone()

    class EffectfulSchedulerState(dict):
        def __deepcopy__(self, memo):
            del memo
            with torch.no_grad():
                target.model.weight.add_(1.0)
            return dict(self)

    effectful = EffectfulSchedulerState(state.scheduler)

    with pytest.raises(
        TrainingStateInvalidError,
        match="trainer model changed during checkpoint preflight",
    ):
        target.load_state_dict(replace(state, scheduler=effectful))

    assert not torch.equal(target.model.weight.detach(), before)
    assert target._failure_reason == "trainer model changed during checkpoint preflight"
    assert target._update_incomplete is False
    assert not target.optimizer.state


def test_direct_restore_poison_target_when_state_mapping_decode_mutates_then_raises() -> None:
    from collections.abc import Mapping

    config = _config()
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)

    class ExplodingStateMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

        def keys(self):
            target.tokens_seen = 1
            raise RuntimeError("state mapping decode exploded")

    with pytest.raises(RuntimeError, match="state mapping decode exploded"):
        target.load_state_dict(ExplodingStateMapping())

    assert target.tokens_seen == 1
    assert target._failure_reason == (
        "trainer restore state changed during checkpoint preflight"
    )
    assert target._update_incomplete is False
    assert not target.optimizer.state


def test_direct_restore_poison_target_when_optimizer_preflight_mutates_then_raises() -> None:
    from dataclasses import replace

    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)

    class ExplodingOptimizerState(dict):
        armed = True

        def get(self, key, default=None):
            if self.armed and key == "param_groups":
                self.armed = False
                target.tokens_seen = 1
                raise RuntimeError("optimizer preflight exploded")
            return super().get(key, default)

    hostile_state = replace(state, optimizer=ExplodingOptimizerState(state.optimizer))

    with pytest.raises(RuntimeError, match="optimizer preflight exploded"):
        target.load_state_dict(hostile_state)

    assert target.tokens_seen == 1
    assert target._failure_reason == (
        "trainer restore state changed during checkpoint preflight"
    )
    assert target._update_incomplete is False
    assert not target.optimizer.state


def test_checkpoint_rng_fingerprint_is_observer_only() -> None:
    import random

    import numpy as np

    config = _config()
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cuda_was_initialized = torch.cuda.is_initialized()

    first = target._checkpoint_rng_fingerprint()
    second = target._checkpoint_rng_fingerprint()

    assert first == second
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert torch.cuda.is_initialized() is cuda_was_initialized


def test_direct_restore_preflight_seals_autograd_mode() -> None:
    from dataclasses import replace

    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="cosine",
        warmup_steps=1,
        gradient_accumulation_steps=1,
        seed=17,
    )
    state = _clean_state(config)
    assert state.scheduler is not None
    target = Trainer(nn.Linear(3, 2), config)
    expected_grad_enabled = torch.is_grad_enabled()

    class GradModeMutatingSchedulerState(dict):
        def __deepcopy__(self, memo):
            del memo
            torch.set_grad_enabled(not expected_grad_enabled)
            return dict(self)

    hostile = replace(
        state,
        scheduler=GradModeMutatingSchedulerState(state.scheduler),
    )

    try:
        with pytest.raises(
            TrainingStateInvalidError,
            match="trainer autograd mode changed during scheduler payload ownership",
        ):
            target.load_state_dict(hostile)

        assert torch.is_grad_enabled() is not expected_grad_enabled
        assert target._failure_reason == (
            "trainer autograd mode changed during scheduler payload ownership"
        )
        assert target._update_incomplete is False
        assert not target.optimizer.state
    finally:
        torch.set_grad_enabled(expected_grad_enabled)


def test_direct_restore_poison_target_when_checkpoint_preflight_consumes_rng() -> None:
    from dataclasses import replace

    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="cosine",
        warmup_steps=1,
        gradient_accumulation_steps=1,
        seed=17,
    )
    state = _clean_state(config)
    assert state.scheduler is not None
    target = Trainer(nn.Linear(3, 2), config)
    rng_before = torch.get_rng_state().clone()

    class RngMutatingSchedulerState(dict):
        def __deepcopy__(self, memo):
            del memo
            torch.rand(1)
            return dict(self)

    hostile = replace(
        state,
        scheduler=RngMutatingSchedulerState(state.scheduler),
    )

    try:
        with pytest.raises(
            TrainingStateInvalidError,
            match="trainer RNG state changed during scheduler payload ownership",
        ):
            target.load_state_dict(hostile)

        assert not torch.equal(torch.get_rng_state(), rng_before)
        assert target._failure_reason == (
            "trainer RNG state changed during scheduler payload ownership"
        )
        assert target._update_incomplete is False
        assert not target.optimizer.state
    finally:
        torch.set_rng_state(rng_before)


def test_direct_restore_preflight_seals_python_and_numpy_rng() -> None:
    import random

    import numpy as np
    from dataclasses import replace

    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=4,
        scheduler="cosine",
        warmup_steps=1,
        gradient_accumulation_steps=1,
        seed=23,
    )
    state = _clean_state(config)
    assert state.scheduler is not None
    target = Trainer(nn.Linear(3, 2), config)
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    class PythonNumpyRngMutatingSchedulerState(dict):
        def __deepcopy__(self, memo):
            del memo
            random.random()
            np.random.random()
            return dict(self)

    hostile = replace(
        state,
        scheduler=PythonNumpyRngMutatingSchedulerState(state.scheduler),
    )

    try:
        with pytest.raises(
            TrainingStateInvalidError,
            match="trainer RNG state changed during scheduler payload ownership",
        ):
            target.load_state_dict(hostile)

        assert random.getstate() != python_before
        numpy_after = np.random.get_state()
        assert (
            numpy_after[2] != numpy_before[2]
            or not np.array_equal(numpy_after[1], numpy_before[1])
        )
        assert target._failure_reason == (
            "trainer RNG state changed during scheduler payload ownership"
        )
        assert target._update_incomplete is False
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)


def test_direct_restore_poison_target_when_component_load_consumes_rng() -> None:
    config = _config()
    state = _clean_state(config)
    model = nn.Linear(3, 2)

    class RngMutatingAdamW(AdamW):
        def load_state_dict(self, state_dict):
            result = super().load_state_dict(state_dict)
            torch.rand(1)
            return result

    optimizer = RngMutatingAdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    target = Trainer(
        model,
        config,
        optimizer=optimizer,
        scheduler=None,
    )
    rng_before = torch.get_rng_state().clone()

    try:
        with pytest.raises(
            TrainingStateInvalidError,
            match="trainer RNG state changed during load",
        ):
            target.load_state_dict(state)

        assert not torch.equal(torch.get_rng_state(), rng_before)
        assert target._failure_reason == (
            "trainer state restore failed after possible partial apply"
        )
        assert target._update_incomplete is True
        with pytest.raises(
            TrainingStateInvalidError,
            match="failed trainer cannot be repaired in place",
        ):
            target.load_state_dict(state)
    finally:
        torch.set_rng_state(rng_before)


def test_direct_restore_rejects_entry_deterministic_policy_drift_before_state_access() -> None:
    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    expected_enabled = torch.are_deterministic_algorithms_enabled()
    expected_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    class StateAccessTrap(dict):
        def keys(self):
            raise AssertionError("checkpoint state must not be accessed")

    hostile_state = StateAccessTrap(
        micro_step=state.micro_step,
        optimizer_step=state.optimizer_step,
        tokens_seen=state.tokens_seen,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        scaler=state.scaler,
        config=state.config,
    )

    try:
        torch.use_deterministic_algorithms(
            not expected_enabled,
            warn_only=expected_warn_only,
        )
        with pytest.raises(
            TrainingStateInvalidError,
            match="live PyTorch deterministic policy disagrees",
        ):
            target.load_state_dict(hostile_state)

        assert target._failure_reason is None
        assert target._update_incomplete is False
        assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
        assert not target.optimizer.state
    finally:
        torch.use_deterministic_algorithms(
            expected_enabled,
            warn_only=expected_warn_only,
        )

    target.load_state_dict(state)
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_direct_restore_poison_target_when_payload_drifts_deterministic_policy() -> None:
    from dataclasses import replace

    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    expected_enabled = torch.are_deterministic_algorithms_enabled()
    expected_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    class PolicyMutatingOptimizerState(dict):
        armed = True

        def get(self, key, default=None):
            if self.armed and key == "param_groups":
                self.armed = False
                torch.use_deterministic_algorithms(
                    not expected_enabled,
                    warn_only=expected_warn_only,
                )
                raise RuntimeError("optimizer preflight changed deterministic policy")
            return super().get(key, default)

    hostile_state = replace(
        state,
        optimizer=PolicyMutatingOptimizerState(state.optimizer),
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="optimizer preflight changed deterministic policy",
        ):
            target.load_state_dict(hostile_state)
    finally:
        torch.use_deterministic_algorithms(
            expected_enabled,
            warn_only=expected_warn_only,
        )

    assert target._failure_reason == (
        "trainer deterministic policy changed during checkpoint preflight"
    )
    assert target._update_incomplete is False
    assert not target.optimizer.state


def test_direct_restore_rejects_late_marker_storage_descriptor_before_mutation() -> None:
    from twelve_six.checkpoint import trainer_adapter

    class LateMarkerDescriptor:
        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            return vars(instance).get("_update_incomplete", False)

        def __set__(self, instance, value):
            attrs = vars(instance)
            if attrs.get("_descriptor_armed", False) and value is False:
                with torch.no_grad():
                    instance.model.weight.add_(1.0)
            attrs["_update_incomplete"] = value

    class DescriptorTrainer(Trainer):
        _update_incomplete = LateMarkerDescriptor()

    config = _config()
    state = _clean_state(config)
    target = DescriptorTrainer(nn.Linear(3, 2), config, scheduler=None)
    target._descriptor_armed = True
    before = target.model.weight.detach().clone()
    message = (
        "native D02 restore storage descriptor must remain canonical: "
        "_update_incomplete"
    )

    with pytest.raises(TrainingStateInvalidError, match=message):
        Trainer.load_state_dict(target, state)

    assert torch.equal(target.model.weight.detach(), before)
    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert not target.optimizer.state

    with pytest.raises(trainer_adapter.CheckpointCompatibilityError, match=message):
        trainer_adapter._assert_native_d02_checkpoint_safety_lineage(target)

    assert torch.equal(target.model.weight.detach(), before)


def test_direct_restore_rejects_config_subclass_before_attribute_dispatch() -> None:
    observed: list[str] = []

    class HostileConfig(TrainerConfig):
        def __getattribute__(self, name: str):
            if name in {
                "learning_rate",
                "max_steps",
                "gradient_accumulation_steps",
                "deterministic_algorithms",
                "deterministic_warn_only",
            }:
                observed.append(name)
            return super().__getattribute__(name)

    canonical = _config()
    hostile = HostileConfig(
        learning_rate=canonical.learning_rate,
        weight_decay=canonical.weight_decay,
        betas=canonical.betas,
        eps=canonical.eps,
        max_steps=canonical.max_steps,
        warmup_steps=canonical.warmup_steps,
        scheduler=canonical.scheduler,
        gradient_accumulation_steps=canonical.gradient_accumulation_steps,
        gradient_clip_norm=canonical.gradient_clip_norm,
        precision=canonical.precision,
        seed=canonical.seed,
        deterministic_algorithms=canonical.deterministic_algorithms,
        deterministic_warn_only=canonical.deterministic_warn_only,
    )
    state = _clean_state(canonical)
    target = Trainer(nn.Linear(3, 2), hostile, scheduler=None)
    observed.clear()

    with pytest.raises(
        TrainingStateInvalidError,
        match="trainer restore config must use canonical TrainerConfig",
    ):
        target.load_state_dict(state)

    assert observed == []
    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert not target.optimizer.state


def test_direct_restore_rejects_duck_typed_state_without_attribute_dispatch() -> None:
    config = _config()
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    observed: list[str] = []

    class DuckState:
        def __getattribute__(self, name: str):
            if name.startswith("_"):
                return object.__getattribute__(self, name)
            observed.append(name)
            raise AssertionError(f"unexpected state attribute access: {name}")

    with pytest.raises(
        TypeError,
        match="trainer state must be TrainerState or a mapping",
    ):
        target.load_state_dict(DuckState())

    assert observed == []
    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert not target.optimizer.state


def test_direct_restore_validates_owned_optimizer_snapshot_before_apply() -> None:
    import copy
    from dataclasses import replace

    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)

    class DriftingGroups(list):
        def __deepcopy__(self, memo):
            copied = copy.deepcopy(list(self), memo)
            copied[0] = dict(copied[0])
            copied[0]["lr"] = -1.0
            return copied

    optimizer_state = dict(state.optimizer)
    optimizer_state["param_groups"] = DriftingGroups(state.optimizer["param_groups"])
    hostile = replace(state, optimizer=optimizer_state)

    with pytest.raises(
        FloatingPointError,
        match="optimizer learning rate must be finite and >= 0",
    ):
        target.load_state_dict(hostile)

    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert not target.optimizer.state
    assert target.optimizer.param_groups[0]["lr"] == config.learning_rate

    target.load_state_dict(state)
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_direct_restore_rejects_nonfinite_owned_optimizer_state_before_apply() -> None:
    import copy
    from dataclasses import replace

    config = _config()
    state = _clean_state(config)
    target = Trainer(nn.Linear(3, 2), config, scheduler=None)
    bad_optimizer = copy.deepcopy(state.optimizer)
    bad_optimizer["state"][0] = {"diagnostic": float("nan")}
    hostile = replace(state, optimizer=bad_optimizer)

    with pytest.raises(
        FloatingPointError,
        match="checkpoint optimizer has non-finite state",
    ):
        target.load_state_dict(hostile)

    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert not target.optimizer.state

    target.load_state_dict(state)
    assert target._failure_reason is None
    assert target._update_incomplete is False
