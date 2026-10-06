"""Reject corrupt D02 optimizer ownership before checkpoint I/O or weight mutation.

The real D02 integration cases run when #2624's coverage validator is available.
Legacy D02 and generic trainer adapters remain compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    progress_trainer,
    trainer_adapter,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


class _CoverageTarget:
    _failure_reason: str | None = None
    _update_incomplete: bool = False

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.violation: str | None = "optimizer contains a foreign parameter"
        self.coverage_checks = 0

    def _require_optimizer_parameter_coverage(self) -> None:
        self.coverage_checks += 1
        if self.violation is not None:
            raise ValueError(self.violation)

    def load_state_dict(self, _state: Any) -> None:
        raise AssertionError("trainer state must not load before checkpoint preflight")


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_generic_target_does_not_dispatch_native_optimizer_ownership_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: Any,
) -> None:
    model = torch.nn.Linear(2, 2)
    trainer = _CoverageTarget(model)
    unchanged = [p.detach().clone() for p in model.parameters()]
    attempts: list[str] = []

    class CheckpointReadReached(Exception):
        pass

    def reached_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("open")
        raise CheckpointReadReached()

    monkeypatch.setattr(loader, "prepare_checkpoint_load", reached_read)
    with pytest.raises(CheckpointReadReached):
        loader.load_trainer_checkpoint(
            tmp_path / "nonexistent-checkpoint", model=model, trainer=trainer,
        )

    # Optimizer ownership is a native D02 authority. Generic adapters are
    # validated through their isolated state probe and must not dispatch an
    # arbitrary same-named callback before checkpoint I/O.
    assert trainer.coverage_checks == 0
    assert attempts == ["open"]
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    for parameter, before in zip(model.parameters(), unchanged, strict=True):
        torch.testing.assert_close(parameter.detach(), before, rtol=0, atol=0)


@pytest.mark.skipif(
    not hasattr(Trainer, "_require_optimizer_parameter_coverage"),
    reason="real D02 optimizer ownership contract is on separate PR #2624",
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("violation", ["foreign", "duplicate", "missing"])
def test_real_d02_mutated_optimizer_refused_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: Any, violation: str,
) -> None:
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, TrainerConfig(max_steps=2, seed=703), device="cpu")
    group = trainer.optimizer.param_groups[0]["params"]
    assert len(group) >= 2
    if violation == "foreign":
        group[0] = torch.nn.Parameter(torch.zeros_like(group[0]))
    elif violation == "duplicate":
        group[1] = group[0]
    else:
        group.pop()
    unchanged = [p.detach().clone() for p in model.parameters()]
    attempts: list[str] = []

    def forbidden_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("open")
        raise AssertionError("bad optimizer ownership must reject before checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbidden_read)
    with pytest.raises(CheckpointCompatibilityError, match="optimizer ownership"):
        loader.load_trainer_checkpoint(
            tmp_path / "nonexistent-checkpoint", model=model, trainer=trainer,
        )
    assert attempts == []
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    for parameter, before in zip(model.parameters(), unchanged, strict=True):
        torch.testing.assert_close(parameter.detach(), before, rtol=0, atol=0)
