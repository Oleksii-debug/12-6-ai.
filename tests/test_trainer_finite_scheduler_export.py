"""Finite but detached scheduler exports must not alter exact-resume chronology.

Synthetic CPU-only regression; does not credit learned data, target exposure or
real training, final-test access or Windows qualification.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.training import (
    Trainer,
    TrainerConfig,
    TrainingStateInvalidError,
)


class _TinyLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.fixture
def preserve_state():
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])


@pytest.mark.parametrize(
    "field", ["last_epoch", "_step_count", "base_lrs", "_last_lr"],
)
def test_detached_finite_scheduler_export_poisoned(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any, field: str,
) -> None:
    model = _TinyLogits()
    trainer = Trainer(
        model, TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    scheduler = trainer.scheduler
    assert scheduler is not None
    original_export = scheduler.state_dict
    initial_epoch = scheduler.last_epoch
    initial_weights = model.weight.detach().clone()
    calls: list[int] = []

    def forged_export() -> dict[str, Any]:
        snapshot = copy.deepcopy(original_export())
        calls.append(1)
        if len(calls) == 2:
            if field in {"last_epoch", "_step_count"}:
                snapshot[field] += 2
            else:
                snapshot[field] = [float(snapshot[field][0]) + 0.01]
        return snapshot

    monkeypatch.setattr(scheduler, "state_dict", forged_export)
    with pytest.raises(TrainingStateInvalidError, match="scheduler export"):
        trainer.state_dict()
    assert len(calls) >= 2
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        1, 1, 2,
    )
    assert scheduler.last_epoch == initial_epoch
    torch.testing.assert_close(model.weight, initial_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_detached_finite_lambda_payload_poisoned(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    scheduler = trainer.scheduler
    assert scheduler is not None
    original_export = scheduler.state_dict
    calls: list[int] = []

    def forged_export() -> dict[str, Any]:
        snapshot = copy.deepcopy(original_export())
        calls.append(1)
        if len(calls) == 2:
            snapshot["lr_lambdas"] = [{"finite_but_wrong": 1}]
        return snapshot

    monkeypatch.setattr(scheduler, "state_dict", forged_export)
    with pytest.raises(TrainingStateInvalidError, match="scheduler export"):
        trainer.state_dict()
    assert trainer._failure_reason is not None


def test_normal_lambda_scheduler_export_keeps_resume_state(
    preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    saved = trainer.state_dict()
    assert saved.scheduler is not None
    assert trainer.scheduler is not None
    assert saved.scheduler["last_epoch"] == trainer.scheduler.last_epoch
    assert saved.scheduler["_last_lr"] == trainer.scheduler.get_last_lr()
    assert saved.scheduler["lr_lambdas"] == [None]
    assert trainer._failure_reason is None


@pytest.mark.parametrize("forged_epoch", [0, 3, False, 1.0])
def test_finite_live_scheduler_chronology_cannot_be_saved(
    preserve_state: Any, forged_epoch: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.optimizer_step == 1
    scheduler = trainer.scheduler
    assert scheduler is not None and scheduler.last_epoch == 1
    scheduler.last_epoch = forged_epoch
    with pytest.raises(
        TrainingStateInvalidError,
        match="scheduler chronology differs from committed optimizer step",
    ):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        1, 1, 2,
    )
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)

def test_missing_scheduler_export_is_not_a_valid_none_snapshot(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.scheduler is not None
    original_export = trainer.scheduler.state_dict
    calls: list[int] = []

    def missing_export() -> dict[str, Any] | None:
        calls.append(1)
        return None if len(calls) == 2 else original_export()

    monkeypatch.setattr(trainer.scheduler, "state_dict", missing_export)
    with pytest.raises(TrainingStateInvalidError, match="scheduler export"):
        trainer.state_dict()
    assert len(calls) >= 2
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        0, 0, 0,
    )
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_genuinely_absent_scheduler_still_exports_none(
    preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=2, scheduler="constant"),
        device="cpu",
    )
    assert trainer.scheduler is None
    assert trainer.state_dict().scheduler is None
    assert trainer._failure_reason is None
