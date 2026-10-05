"""Effectful optimizer serialization must not publish an unsafe D02 snapshot.

Synthetic CPU tests only; not real data exposure, learned weights or scale credit.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.training import (
    NonFiniteTrainingError,
    Trainer,
    TrainerConfig,
    TrainingStateInvalidError,
)


class _Logits(torch.nn.Module):
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
def preserve_process_state():
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(
            policy_before[0], warn_only=policy_before[1],
        )


@pytest.mark.parametrize(
    ("fault", "error_type", "error_pattern"),
    [
        ("weights", NonFiniteTrainingError, "non-finite model weights"),
        ("moment", NonFiniteTrainingError, "non-finite state"),
        ("gradient", RuntimeError, "residual model gradients"),
        ("policy", TrainingStateInvalidError, "deterministic policy"),
        ("interrupt", KeyboardInterrupt, "serialization interrupted"),
    ],
)
def test_effectful_optimizer_state_dict_never_publishes_unsafe_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    preserve_process_state: Any,
    fault: str,
    error_type: type[BaseException],
    error_pattern: str,
) -> None:
    model = _Logits()
    trainer = Trainer(
        model,
        TrainerConfig(seed=703, max_steps=2, deterministic_warn_only=True),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.optimizer_step == 1 and trainer.tokens_seen == 2
    original_state_dict = trainer.optimizer.state_dict
    initial_weights = model.weight.detach().clone()

    def mutate_during_export() -> dict[str, Any]:
        snapshot = original_state_dict()
        if fault == "weights":
            model.weight.data.fill_(float("nan"))
        elif fault == "moment":
            trainer.optimizer.state[model.weight]["exp_avg"].fill_(float("nan"))
        elif fault == "gradient":
            model.weight.grad = torch.ones_like(model.weight)
        elif fault == "policy":
            torch.use_deterministic_algorithms(False, warn_only=False)
        else:
            raise KeyboardInterrupt("serialization interrupted")
        return snapshot

    monkeypatch.setattr(trainer.optimizer, "state_dict", mutate_during_export)
    with pytest.raises(error_type, match=error_pattern):
        trainer.state_dict()

    # The optimizer step had already committed; export must not credit a new one.
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    assert trainer._failure_reason is not None
    assert model.weight.grad is None
    if fault != "weights":
        torch.testing.assert_close(model.weight.detach(), initial_weights, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_ordinary_state_export_still_preserves_named_adamw_resume(
    preserve_process_state: Any,
) -> None:
    trainer = Trainer(_Logits(), TrainerConfig(seed=703, max_steps=2), device="cpu")
    fresh = trainer.state_dict()
    assert (fresh.micro_step, fresh.optimizer_step, fresh.tokens_seen) == (0, 0, 0)
    assert fresh.optimizer["param_groups"][0]["param_names"] == ["weight"]
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    saved = trainer.state_dict()
    assert (saved.micro_step, saved.optimizer_step, saved.tokens_seen) == (1, 1, 2)
    assert saved.optimizer["param_groups"][0]["param_names"] == ["weight"]
    assert trainer._failure_reason is None and trainer._update_incomplete is False
    assert any(trainer.optimizer.state.values())


@pytest.mark.parametrize(
    ("fault", "expected_error"),
    [
        ("optimizer_moment", "checkpoint optimizer has non-finite state"),
        ("optimizer_lr", "checkpoint optimizer has non-finite state"),
        ("scheduler", "checkpoint scheduler has non-finite state"),
        ("scaler", "checkpoint gradient scaler has non-finite state"),
    ],
)
def test_detached_corrupt_snapshot_poisoned_even_when_live_state_stays_finite(
    monkeypatch: pytest.MonkeyPatch,
    preserve_process_state: Any,
    fault: str,
    expected_error: str,
) -> None:
    model = _Logits()
    config = TrainerConfig(
        seed=703, max_steps=2,
        scheduler="cosine" if fault == "scheduler" else "constant",
        warmup_steps=1 if fault == "scheduler" else 0,
    )
    trainer = Trainer(model, config, device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    weights_before = model.weight.detach().clone()
    original_moment = trainer.optimizer.state[model.weight]["exp_avg"].clone()
    if fault in {"optimizer_moment", "optimizer_lr"}:
        real_export = trainer.optimizer.state_dict

        def bad_optimizer_snapshot() -> dict[str, Any]:
            from copy import deepcopy

            detached = deepcopy(real_export())
            if fault == "optimizer_moment":
                next(iter(detached["state"].values()))["exp_avg"].fill_(float("nan"))
            else:
                detached["param_groups"][0]["lr"] = float("nan")
            return detached

        monkeypatch.setattr(trainer.optimizer, "state_dict", bad_optimizer_snapshot)
    else:
        component = trainer.scheduler if fault == "scheduler" else trainer.scaler
        assert component is not None
        original_export = component.state_dict
        calls: list[int] = []

        def bad_second_snapshot() -> dict[str, Any]:
            from copy import deepcopy

            calls.append(1)
            detached = deepcopy(original_export())
            if len(calls) == 2:
                if fault == "scheduler":
                    detached["base_lrs"] = [float("nan")]
                else:
                    detached["scale"] = float("nan")
            return detached

        monkeypatch.setattr(component, "state_dict", bad_second_snapshot)
    with pytest.raises(NonFiniteTrainingError, match=expected_error):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    torch.testing.assert_close(model.weight.detach(), weights_before, rtol=0, atol=0)
    torch.testing.assert_close(
        trainer.optimizer.state[model.weight]["exp_avg"], original_moment,
        rtol=0, atol=0,
    )
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()

@pytest.mark.parametrize(
    ("attack", "expected_message"),
    [
        ("finite-moment", "optimizer export"),
        ("finite-hyperparam", "optimizer export"),
        ("finite-live-lr", "default constant optimizer rate"),
        ("alias-state-id", "optimizer export"),
    ],
)
def test_finite_detached_optimizer_export_must_match_live_committed_state(
    monkeypatch: pytest.MonkeyPatch,
    preserve_process_state: Any,
    attack: str,
    expected_message: str,
) -> None:
    import copy

    model = _Logits()
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=2), device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    original_export = trainer.optimizer.state_dict
    weights = model.weight.detach().clone()
    original_moment = trainer.optimizer.state[model.weight]["exp_avg"].clone()

    def divergent_export() -> dict[str, Any]:
        data = copy.deepcopy(original_export())
        if attack == "finite-moment":
            next(iter(data["state"].values()))["exp_avg"].add_(0.125)
        elif attack == "finite-hyperparam":
            data["param_groups"][0]["lr"] += 0.00001
        elif attack == "finite-live-lr":
            trainer.optimizer.param_groups[0]["lr"] *= 2.0
        else:
            value = data["state"].pop(0)
            data["state"][False] = value
        return data

    monkeypatch.setattr(trainer.optimizer, "state_dict", divergent_export)
    with pytest.raises(TrainingStateInvalidError, match=expected_message):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    torch.testing.assert_close(model.weight, weights, rtol=0, atol=0)
    torch.testing.assert_close(
        trainer.optimizer.state[model.weight]["exp_avg"], original_moment,
        rtol=0, atol=0,
    )
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
