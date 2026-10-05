"""D05 rejects mistyped canonical D02 configuration before model mutation.

Synthetic pre-application checks; never evidence of real corpus training.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointCompatibilityError, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


class _CounterSubclass(int):
    """Numerically equal to int but not an exact durable D02 counter type."""


@pytest.fixture
def preserve_ambient_state():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize(
    ("field", "alias", "config_overrides"),
    [
        ("deterministic_algorithms", 1, {}),
        ("deterministic_warn_only", 0, {}),
        ("seed", 703.0, {}),
        ("weight_decay", 0, {}),
        ("gradient_clip_norm", 1, {}),
        ("betas", (0, 0.95), {"betas": (0.0, 0.95)}),
    ],
)
def test_canonical_alias_refused_before_optimizer_preflight_or_model_apply(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
    field: str,
    alias: Any,
    config_overrides: dict[str, Any],
) -> None:
    config = TrainerConfig(seed=703, max_steps=1, **config_overrides)
    source = Trainer(_TinyModel(), config, device="cpu")
    valid = asdict(source.state_dict())
    invalid = copy.deepcopy(valid)
    invalid["config"][field] = alias
    model = _TinyModel()
    target = Trainer(model, config, device="cpu")
    initial_weights = model.weight.detach().clone()
    optimizer_preflights: list[bool] = []

    def forbidden_preflight(*args: Any, **kwargs: Any) -> None:
        optimizer_preflights.append(True)
        raise AssertionError("typed config must fail before optimizer preflight")

    monkeypatch.setattr(trainer_adapter, "_preflight_optimizer_state", forbidden_preflight)
    with pytest.raises(CheckpointCompatibilityError, match="trainer config mismatch"):
        trainer_adapter._preflight_trainer_state_without_rng_guard(target, invalid)
    assert optimizer_preflights == []
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None and target._update_incomplete is False
    assert not target.optimizer.state
    torch.testing.assert_close(model.weight.detach(), initial_weights, rtol=0, atol=0)

    monkeypatch.undo()
    trainer_adapter._preflight_trainer_state_without_rng_guard(target, valid)
    assert target._failure_reason is None


@pytest.mark.parametrize("subclass_counter", [False, True])
def test_generic_adapter_retains_existing_config_equality(
    subclass_counter: bool,
) -> None:
    class GenericAdapter:
        def __init__(self) -> None:
            self.config = {"deterministic_algorithms": True}
            self.optimizer = None

        def load_state_dict(self, state: dict[str, Any]) -> None:
            self.restored = state

    state = {
        "micro_step": _CounterSubclass(0) if subclass_counter else 0,
        "optimizer_step": 0,
        "tokens_seen": 0,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "config": {"deterministic_algorithms": 1},
    }
    trainer_adapter._preflight_trainer_state_without_rng_guard(GenericAdapter(), state)


@pytest.mark.parametrize("field", ["micro_step", "optimizer_step", "tokens_seen"])
def test_canonical_counter_subclass_fails_before_optimizer_preflight(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
    field: str,
) -> None:
    """D05 must not accept a counter that direct typed D02 restore rejects."""
    config = TrainerConfig(seed=703, max_steps=1)
    saved = asdict(Trainer(_TinyModel(), config, device="cpu").state_dict())
    invalid = copy.deepcopy(saved)
    invalid[field] = _CounterSubclass(0)
    assert invalid[field] == saved[field] and type(invalid[field]) is not int

    model = _TinyModel()
    target = Trainer(model, config, device="cpu")
    weights_before = model.weight.detach().clone()
    calls: list[bool] = []

    def forbidden_preflight(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("canonical counter preflight must reject before optimizer")

    monkeypatch.setattr(trainer_adapter, "_preflight_optimizer_state", forbidden_preflight)
    with pytest.raises(
        CheckpointCompatibilityError, match=f"trainer {field} must be a non-negative integer",
    ):
        trainer_adapter._preflight_trainer_state_without_rng_guard(target, invalid)

    assert calls == []
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(model.weight.detach(), weights_before, rtol=0, atol=0)

    monkeypatch.undo()
    trainer_adapter._preflight_trainer_state_without_rng_guard(target, saved)
    assert target._failure_reason is None
