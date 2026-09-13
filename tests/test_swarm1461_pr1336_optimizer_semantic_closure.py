"""Independent adversarial verifier for PR #1336 optimizer/scheduler semantics."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
    _require_actual_execution_matches_packet,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer

_MODEL_SHA = "1" * 64
_INIT_SHA = "2" * 64


class _Spec:
    def __init__(self, parameter_count: int) -> None:
        self._parameter_count = parameter_count

    def identity_sha256(self) -> str:
        return _MODEL_SHA

    def parameter_count(self) -> int:
        return self._parameter_count


class _InitSpec:
    def identity_sha256(self) -> str:
        return _INIT_SHA


class _TinyModel(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4, bias=False)
        self.spec = _Spec(sum(parameter.numel() for parameter in self.parameters()))
        self.init_spec = _InitSpec()


def _config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=0.00022,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=1,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=1337,
    )


def _packet() -> dict[str, object]:
    return {
        "identities": {
            "modelspec_sha256": _MODEL_SHA,
            "initspec_sha256": _INIT_SHA,
        },
        "recipe": {
            "optimizer_scheduler_precision": {
                "optimizer": "AdamW",
                "scheduler": "constant",
                "precision": "fp32",
            },
            "seed": 1337,
        },
    }


def _optimizer(model: nn.Module, *, maximize: bool = False) -> AdamW:
    config = _config()
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        maximize=maximize,
    )


def _snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _assert_unmutated(trainer: Trainer, before: list[torch.Tensor]) -> None:
    assert trainer.micro_step == 0
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0
    after = [parameter.detach() for parameter in trainer.model.parameters()]
    assert len(after) == len(before)
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_rejects_same_class_adamw_maximize_drift_before_mutation() -> None:
    model = _TinyModel()
    before = _snapshot(model)
    trainer = Trainer(model, _config(), optimizer=_optimizer(model, maximize=True))

    with pytest.raises(BoundedPilotAuthorizationError, match="optimizer"):
        _require_actual_execution_matches_packet(trainer, _packet())

    _assert_unmutated(trainer, before)


def test_rejects_injected_scheduler_under_constant_zero_warmup_before_mutation() -> None:
    model = _TinyModel()
    before = _snapshot(model)
    optimizer = _optimizer(model)
    injected_scheduler = LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    trainer = Trainer(
        model,
        _config(),
        optimizer=optimizer,
        scheduler=injected_scheduler,
    )

    assert trainer.config.scheduler == "constant"
    assert trainer.config.warmup_steps == 0
    assert trainer.scheduler is injected_scheduler
    with pytest.raises(BoundedPilotAuthorizationError, match="scheduler"):
        _require_actual_execution_matches_packet(trainer, _packet())

    _assert_unmutated(trainer, before)
