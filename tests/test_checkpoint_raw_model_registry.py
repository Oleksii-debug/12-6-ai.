"""Checkpoint/training safety must use registered Module state, not overridable iterators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from twelve_six.checkpoint import CheckpointCompatibilityError, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


class _TwoParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.ones(3))
        self.right = torch.nn.Parameter(torch.full((3,), 2.0))


class _HiddenRightParameters(_TwoParameters):
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        del recurse
        yield self.left


class _LyingNamedParameters(_TwoParameters):
    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        del prefix, recurse, remove_duplicate
        yield "forged-right", self.right
        yield "forged-left", self.left


class _ArmedDictModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
        self._dict_spoof_armed = False
        self.dict_reads: list[str] = []

    @property
    def __dict__(self) -> dict[str, Any]:
        descriptor = torch.nn.Module.__dict__["__dict__"]
        real = descriptor.__get__(self, type(self))
        if not real.get("_dict_spoof_armed", False):
            return real
        real["dict_reads"].append("__dict__")
        return {
            "training": True,
            "_parameters": {},
            "_buffers": {},
            "_modules": {},
            "_non_persistent_buffers_set": set(),
        }


class _ArmedDictAdamW(torch.optim.AdamW):
    def __init__(self, params: Any, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        raw = torch.optim.Optimizer.__dict__["__dict__"].__get__(self, type(self))
        raw["_dict_spoof_armed"] = False
        raw["dict_reads"] = []

    @property
    def __dict__(self) -> dict[str, Any]:
        descriptor = torch.optim.Optimizer.__dict__["__dict__"]
        real = descriptor.__get__(self, type(self))
        if not real.get("_dict_spoof_armed", False):
            return real
        real["dict_reads"].append("__dict__")
        return {
            "state": {},
            "param_groups": [],
            "_dict_spoof_armed": True,
            "dict_reads": real["dict_reads"],
        }


class _TypeIdentityMeta(type):
    reads: list[str] = []

    def __getattribute__(cls, name: str) -> Any:
        if name in {"__module__", "__qualname__"}:
            type.__getattribute__(cls, "reads").append(name)
            return "forged.type"
        return type.__getattribute__(cls, name)


class _MetaclassAdamW(torch.optim.AdamW, metaclass=_TypeIdentityMeta):
    pass


def _raw_module_dict(module: torch.nn.Module) -> dict[str, Any]:
    descriptor = torch.nn.Module.__dict__["__dict__"]
    attrs = descriptor.__get__(module, type(module))
    assert type(attrs) is dict
    return attrs


def test_default_optimizer_cannot_accept_hidden_registered_parameter() -> None:
    model = _HiddenRightParameters()

    with pytest.raises(ValueError, match="optimizer omits trainable model parameters"):
        Trainer(model, TrainerConfig(seed=703, max_steps=2), device="cpu")


def test_optimizer_names_ignore_named_parameters_override() -> None:
    model = _LyingNamedParameters()
    optimizer = torch.optim.AdamW(
        [model.left, model.right],
        lr=1e-3,
    )
    trainer = Trainer(
        model,
        TrainerConfig(seed=703, max_steps=2),
        optimizer=optimizer,
        device="cpu",
    )

    assert trainer._optimizer_parameter_name_groups() == [["left", "right"]]


def test_model_fingerprint_ignores_armed_dict_descriptor() -> None:
    model = _ArmedDictModel()
    trainer = Trainer(
        model,
        TrainerConfig(seed=703, max_steps=2),
        device="cpu",
    )
    raw = _raw_module_dict(model)
    raw["_dict_spoof_armed"] = True

    before = trainer._model_export_fingerprint()
    with torch.no_grad():
        raw["_parameters"]["weight"].add_(1.0)
    after = trainer._model_export_fingerprint()

    assert before != after
    assert raw["dict_reads"] == []


def test_auxiliary_fingerprint_ignores_optimizer_dict_descriptor() -> None:
    model = _TwoParameters()
    optimizer = _ArmedDictAdamW(
        [model.left, model.right],
        lr=1e-3,
    )
    trainer = Trainer(
        model,
        TrainerConfig(seed=703, max_steps=2),
        optimizer=optimizer,
        device="cpu",
    )
    raw = torch.optim.Optimizer.__dict__["__dict__"].__get__(
        optimizer,
        type(optimizer),
    )
    raw["_dict_spoof_armed"] = True

    before = trainer._checkpoint_auxiliary_fingerprint()
    raw["param_groups"][0]["lr"] *= 0.5
    after = trainer._checkpoint_auxiliary_fingerprint()

    assert before != after
    assert raw["dict_reads"] == []


def test_auxiliary_type_identity_ignores_custom_metaclass() -> None:
    _TypeIdentityMeta.reads.clear()
    model = _TwoParameters()
    optimizer = _MetaclassAdamW(
        [model.left, model.right],
        lr=1e-3,
    )
    trainer = Trainer(
        model,
        TrainerConfig(seed=703, max_steps=2),
        optimizer=optimizer,
        device="cpu",
    )

    trainer._checkpoint_auxiliary_fingerprint()

    assert _TypeIdentityMeta.reads == []


def test_restore_preflight_sees_gradient_hidden_from_parameters_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _HiddenRightParameters()
    optimizer = torch.optim.AdamW(
        [model.left, model.right],
        lr=1e-3,
    )
    trainer = Trainer(
        model,
        TrainerConfig(seed=703, max_steps=2),
        optimizer=optimizer,
        device="cpu",
    )
    model.right.grad = torch.ones_like(model.right)
    reads: list[bool] = []

    def forbidden_read(*args: Any, **kwargs: Any) -> Any:
        reads.append(True)
        raise AssertionError("pending raw gradient reached checkpoint I/O")

    monkeypatch.setattr(trainer_adapter, "prepare_checkpoint_load", forbidden_read)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="fresh trainer with no pending gradients",
    ):
        trainer_adapter.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=model,
            trainer=trainer,
            restore_rng=False,
        )

    assert reads == []
    assert model.right.grad is not None
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
