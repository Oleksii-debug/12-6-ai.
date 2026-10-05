"""Reject finite first-party optimizer drift before the next AdamW update.

These are synthetic CPU tests; no real dataset/learned20M exposure is credited.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.training import Trainer, TrainerConfig, TrainingStateInvalidError


class _HookLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.fixture(autouse=True)
def restore_ambient_state():
    py_before = random.getstate()
    np_before = np.random.get_state()
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])


@pytest.mark.parametrize("schedule", ["constant", "cosine"])
@pytest.mark.parametrize(
    ("field", "forged"), [
        ("lr", 0.12),
        ("weight_decay", 0.5),
        ("betas", (0.5, 0.9)),
        ("eps", 0.1),
        ("amsgrad", True),
    ],
)
def test_forward_hook_cannot_change_first_party_next_update(
    schedule: str, field: str, forged: Any,
) -> None:
    model = _HookLogits()
    trainer = Trainer(
        model, TrainerConfig(
            seed=703, max_steps=3, scheduler=schedule, learning_rate=0.01,
        ),
        device="cpu",
    )
    original_weights = model.weight.detach().clone()
    calls: list[bool] = []

    def change_group(*args: Any) -> None:
        calls.append(True)
        trainer.optimizer.param_groups[0][field] = forged

    handle = model.register_forward_hook(change_group)
    try:
        with pytest.raises(TrainingStateInvalidError):
            trainer.train_microbatch(_BATCH)
    finally:
        handle.remove()

    assert len(calls) == 1
    assert trainer._failure_reason is not None
    assert trainer.optimizer_step == 0
    assert not trainer.optimizer.state
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, original_weights, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


@pytest.mark.parametrize("schedule", ["constant", "cosine"])
@pytest.mark.parametrize(
    ("field", "forged"), [
        ("lr", 0.12), ("weight_decay", 0.5), ("betas", (0.5, 0.9)),
    ],
)
def test_between_batches_finite_default_optimizer_drift_refuses_new_exposure(
    schedule: str, field: str, forged: Any,
) -> None:
    trainer = Trainer(
        _HookLogits(), TrainerConfig(
            seed=703, max_steps=3, scheduler=schedule, learning_rate=0.01,
        ),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    before_weights = trainer.model.weight.detach().clone()
    before = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
    trainer.optimizer.param_groups[0][field] = forged
    with pytest.raises(TrainingStateInvalidError):
        trainer.train_microbatch(_BATCH)
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == before
    torch.testing.assert_close(trainer.model.weight, before_weights, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["last_epoch", "_step_count"])
def test_live_default_lambda_chronology_drift_refused_before_next_batch(
    field: str,
) -> None:
    trainer = Trainer(
        _HookLogits(), TrainerConfig(
            seed=703, max_steps=3, scheduler="cosine", learning_rate=0.01,
        ),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.scheduler is not None
    before = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
    before_weights = trainer.model.weight.detach().clone()
    setattr(trainer.scheduler, field, 99)
    with pytest.raises(ValueError, match="checkpoint scheduler"):
        trainer.train_microbatch(_BATCH)
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == before
    torch.testing.assert_close(trainer.model.weight, before_weights, rtol=0, atol=0)


@pytest.mark.parametrize("schedule", ["constant", "cosine"])
def test_unchanged_first_party_adamw_still_completes_two_updates(schedule: str) -> None:
    trainer = Trainer(
        _HookLogits(), TrainerConfig(
            seed=703, max_steps=3, scheduler=schedule, learning_rate=0.01,
        ),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.optimizer_step == 2
    trainer.assert_checkpoint_safe()
    assert trainer._failure_reason is None


def test_injected_optimizer_is_not_bound_to_first_party_adamw_options() -> None:
    model = _HookLogits()
    config = TrainerConfig(
        seed=703, max_steps=3, scheduler="constant", learning_rate=0.01,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.02, weight_decay=0.2,
    )
    trainer = Trainer(model, config, optimizer=optimizer, device="cpu")
    assert trainer._canonical_default_optimizer_options is None
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.optimizer_step == 2

@pytest.mark.parametrize("schedule", ["constant", "cosine"])
def test_valid_gradient_accumulation_still_commits_one_update(schedule: str) -> None:
    trainer = Trainer(
        _HookLogits(), TrainerConfig(
            seed=703, max_steps=3, scheduler=schedule,
            learning_rate=0.01, gradient_accumulation_steps=2,
        ),
        device="cpu",
    )
    first = trainer.train_microbatch(_BATCH)
    assert not first.optimizer_stepped
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 0, 2)
    second = trainer.train_microbatch(_BATCH)
    assert second.optimizer_stepped
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (2, 1, 4)
    trainer.assert_checkpoint_safe()


@pytest.mark.parametrize("schedule", ["constant", "cosine"])
def test_second_accumulated_forward_hook_cannot_apply_forged_default_rate(
    schedule: str,
) -> None:
    model = _HookLogits()
    trainer = Trainer(
        model, TrainerConfig(
            seed=703, max_steps=3, scheduler=schedule,
            learning_rate=0.01, gradient_accumulation_steps=2,
        ),
        device="cpu",
    )
    first = trainer.train_microbatch(_BATCH)
    assert not first.optimizer_stepped
    original_weights = model.weight.detach().clone()

    def forge_rate(*args: Any) -> None:
        trainer.optimizer.param_groups[0]["lr"] = 0.12

    handle = model.register_forward_hook(forge_rate)
    try:
        with pytest.raises(TrainingStateInvalidError):
            trainer.train_microbatch(_BATCH)
    finally:
        handle.remove()
    assert trainer._failure_reason is not None
    assert trainer.optimizer_step == 0
    assert not trainer.optimizer.state
    torch.testing.assert_close(model.weight, original_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


@pytest.mark.parametrize("field", ["last_epoch", "_step_count"])
def test_forward_hook_forged_default_scheduler_never_reaches_optimizer(
    field: str,
) -> None:
    model = _HookLogits()
    trainer = Trainer(
        model, TrainerConfig(
            seed=703, max_steps=3, scheduler="cosine", learning_rate=0.01,
        ),
        device="cpu",
    )
    assert trainer.scheduler is not None
    original_weights = model.weight.detach().clone()

    def forge_scheduler(*args: Any) -> None:
        setattr(trainer.scheduler, field, 99)

    handle = model.register_forward_hook(forge_scheduler)
    try:
        with pytest.raises(ValueError, match="checkpoint scheduler"):
            trainer.train_microbatch(_BATCH)
    finally:
        handle.remove()
    assert trainer._failure_reason is not None
    assert trainer.optimizer_step == 0
    assert not trainer.optimizer.state
    torch.testing.assert_close(model.weight, original_weights, rtol=0, atol=0)
