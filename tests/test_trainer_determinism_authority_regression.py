"""Expected-red D02 reproducibility regressions; no project data or training credit.

Non-owning handoff from the pinned D02 #2624 head. These cases require
an explicit owner decision and production fix before adoption.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from twelve_six.training import Trainer, TrainerConfig, TrainingStateInvalidError


class _TinyLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.fixture
def preserve_ambient_state():
    """Do not leak global RNG or torch policy changes to other pytest cases."""
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(
            policy_before[0], warn_only=policy_before[1],
        )


def test_fresh_config_seed_defines_numpy_first_draw(preserve_ambient_state) -> None:
    """Same configuration must not inherit unrelated ambient NumPy seeds."""
    config = TrainerConfig(max_steps=1, seed=703)

    np.random.seed(111)
    Trainer(_TinyLogitModel(), config, device="cpu")
    first_draw = float(np.random.random())

    np.random.seed(222)
    Trainer(_TinyLogitModel(), config, device="cpu")
    second_draw = float(np.random.random())

    assert first_draw == second_draw, (
        "fresh D02 Trainer did not initialize the NumPy RNG from its seed; "
        "either seed it here or explicitly establish a different canonical "
        "initialization authority and update this regression"
    )


@pytest.mark.parametrize("operation", ["train", "checkpoint"])
def test_second_trainer_cannot_silently_change_first_trainer_policy(
    preserve_ambient_state, operation: str,
) -> None:
    """A trainer cannot continue or publish under another trainer's mode."""
    model = _TinyLogitModel()
    first = Trainer(
        model,
        TrainerConfig(
            max_steps=1, seed=703,
            deterministic_algorithms=True, deterministic_warn_only=True,
        ),
        device="cpu",
    )
    Trainer(
        _TinyLogitModel(),
        TrainerConfig(
            max_steps=1, seed=704,
            deterministic_algorithms=False, deterministic_warn_only=False,
        ),
        device="cpu",
    )

    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    before_weights = model.weight.detach().clone()

    with pytest.raises(TrainingStateInvalidError, match="deterministic"):
        if operation == "train":
            first.train_microbatch(_BATCH)
        else:
            first.state_dict()

    assert model.forward_calls == 0
    assert first.micro_step == 0
    assert first.optimizer_step == 0
    assert first.tokens_seen == 0
    torch.testing.assert_close(model.weight.detach(), before_weights, rtol=0, atol=0)


def test_matching_policy_still_allows_clean_checkpoint(
    preserve_ambient_state,
) -> None:
    trainer = Trainer(
        _TinyLogitModel(),
        TrainerConfig(
            max_steps=1, seed=703,
            deterministic_algorithms=True, deterministic_warn_only=True,
        ),
        device="cpu",
    )
    state = trainer.state_dict()
    assert state.optimizer_step == 0
    assert state.tokens_seen == 0
