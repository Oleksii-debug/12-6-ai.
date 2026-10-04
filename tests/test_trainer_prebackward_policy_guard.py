"""Forward/loss torch-policy drift must never reach backward or earn step credit.

Synthetic CPU-only engineering tests; not authorized corpus/training evidence.
"""

from __future__ import annotations

import importlib
import random
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.training import Trainer, TrainerConfig, TrainingStateInvalidError


class _TinyLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.forward_calls = 0
        self.drift_in_forward = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if self.drift_in_forward:
            torch.use_deterministic_algorithms(False, warn_only=False)
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.fixture
def preserve_ambient_state():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    enabled_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(enabled_before, warn_only=warn_only_before)


@pytest.mark.parametrize("drift_stage", ["forward", "loss"])
def test_policy_drift_is_poisoned_before_backward(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state: Any,
    drift_stage: str,
) -> None:
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(
            max_steps=1,
            seed=703,
            deterministic_algorithms=True,
            deterministic_warn_only=True,
        ),
        device="cpu",
    )
    initial_weight = model.weight.detach().clone()
    backward_calls: list[torch.Tensor] = []
    model.weight.register_hook(lambda grad: backward_calls.append(grad) or grad)
    loss_calls: list[bool] = []

    train_module = importlib.import_module("twelve_six.training.trainer")
    original_loss = train_module.causal_pair_loss

    def observed_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
        loss_calls.append(True)
        result = original_loss(*args, **kwargs)
        if drift_stage == "loss":
            torch.use_deterministic_algorithms(False, warn_only=False)
        return result

    monkeypatch.setattr(train_module, "causal_pair_loss", observed_loss)
    model.drift_in_forward = drift_stage == "forward"

    with pytest.raises(TrainingStateInvalidError, match="deterministic"):
        trainer.train_microbatch(_BATCH)

    assert model.forward_calls == 1
    assert len(loss_calls) == (0 if drift_stage == "forward" else 1)
    assert backward_calls == [], "backward ran after deterministic policy drift"
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight.detach(), initial_weight, rtol=0, atol=0)
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)
    assert trainer._failure_reason is not None
    assert trainer._update_incomplete is True

    # Repairing the global policy cannot silently unpoison a mixed transition.
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    assert model.forward_calls == 1


def test_matching_policy_retains_normal_one_step(
    preserve_ambient_state: Any,
) -> None:
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(
            max_steps=1,
            seed=703,
            deterministic_algorithms=True,
            deterministic_warn_only=True,
        ),
        device="cpu",
    )
    metrics = trainer.train_microbatch(_BATCH)
    assert metrics.optimizer_stepped
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    trainer.assert_checkpoint_safe()
