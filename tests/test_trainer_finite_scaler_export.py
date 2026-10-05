"""Guard finite but detached GradScaler exports at D02 checkpoint publication.

CPU-only simulated scaler tests do not establish CUDA, physical checkpoint,
authorized data exposure or real learned-model training.
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


def _fresh_trainer() -> Trainer:
    return Trainer(_TinyLogits(), TrainerConfig(seed=703, max_steps=2), device="cpu")


def test_ordinary_disabled_scaler_export_remains_valid(preserve_state: Any) -> None:
    trainer = _fresh_trainer()
    assert not trainer.scaler.is_enabled()
    assert trainer.state_dict().scaler == {}
    assert trainer._failure_reason is None


def test_disabled_scaler_cannot_export_finite_enabled_payload(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any,
) -> None:
    trainer = _fresh_trainer()
    scaler = trainer.scaler
    original_export = scaler.state_dict
    calls: list[int] = []

    def forged_export() -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 2:
            return {"scale": 65536.0}
        return original_export()

    monkeypatch.setattr(scaler, "state_dict", forged_export)
    with pytest.raises(TrainingStateInvalidError, match="gradient scaler export"):
        trainer.state_dict()
    assert len(calls) >= 2
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        0, 0, 0,
    )


@pytest.mark.parametrize(
    "field", [
        "scale", "growth_factor", "backoff_factor",
        "growth_interval", "_growth_tracker",
    ],
)
def test_finite_enabled_scaler_export_cannot_forge_live_statistics(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any, field: str,
) -> None:
    trainer = _fresh_trainer()
    # CPU GradScaler exercises the exact same serializable schema as CUDA
    # without claiming to have run a CUDA/mixed-precision training step.
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=True)
    scaler = trainer.scaler
    original_export = scaler.state_dict
    calls: list[int] = []

    def forged_export() -> dict[str, Any]:
        exported = copy.deepcopy(original_export())
        calls.append(1)
        if len(calls) == 2:
            if field == "scale":
                exported[field] *= 2
            elif field == "backoff_factor":
                exported[field] *= 0.5
            elif field == "growth_factor":
                exported[field] += 0.125
            else:
                exported[field] += 2
        return exported

    monkeypatch.setattr(scaler, "state_dict", forged_export)
    with pytest.raises(TrainingStateInvalidError, match="gradient scaler export"):
        trainer.state_dict()
    assert len(calls) >= 2
    assert trainer._failure_reason is not None


def test_ordinary_enabled_scaler_export_still_valid(preserve_state: Any) -> None:
    trainer = _fresh_trainer()
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=True)
    saved = trainer.state_dict()
    assert saved.scaler == trainer.scaler.state_dict()
    assert trainer._failure_reason is None

@pytest.mark.parametrize("enabled", [False, True])
def test_missing_scaler_export_is_not_a_valid_none_snapshot(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any, enabled: bool,
) -> None:
    trainer = _fresh_trainer()
    if enabled:
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=True)
    original_export = trainer.scaler.state_dict
    calls: list[int] = []

    def missing_export() -> dict[str, Any] | None:
        calls.append(1)
        return None if len(calls) == 2 else original_export()

    monkeypatch.setattr(trainer.scaler, "state_dict", missing_export)
    with pytest.raises(TrainingStateInvalidError, match="gradient scaler export"):
        trainer.state_dict()
    assert len(calls) >= 2
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        0, 0, 0,
    )
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
