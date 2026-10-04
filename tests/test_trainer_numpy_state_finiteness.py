"""Reject non-finite NumPy leaves in direct D02 export and D05 preflight.

Synthetic CPU probes only; these are not optimizer exposure or real training evidence.
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
from twelve_six.training import (
    NonFiniteTrainingError,
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
def preserve_ambient_state():
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state()
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
        torch.set_rng_state(torch_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([float("nan")], dtype=np.float32),
        np.asarray([float("inf")], dtype=np.float64),
        np.asarray([complex(float("nan"), 0)], dtype=np.complex64),
        np.asarray([1.0, float("nan"), 3.0], dtype=np.float64)[::-1],
        np.float32(float("nan")),
        np.complex64(complex(1, float("inf"))),
        complex(float("nan"), 1.0),
    ],
    ids=[
        "numpy-f32-nan", "numpy-f64-inf", "numpy-complex-nan",
        "numpy-noncontiguous", "numpy-scalar-f32", "numpy-scalar-complex",
        "python-complex",
    ],
)
def test_recursive_numeric_tree_rejects_nonfinite_numpy_or_complex(bad: Any) -> None:
    with pytest.raises(NonFiniteTrainingError, match="checkpoint scheduler has non-finite"):
        Trainer._require_finite_state_tree(
            {"list": [{"numpy_leaf": bad}]}, "checkpoint scheduler",
        )


@pytest.mark.parametrize(
    "valid",
    [
        np.asarray([0.0, 1.5], dtype=np.float32),
        np.asarray([complex(1.0, -1.0)], dtype=np.complex128),
        np.asarray([0, 42], dtype=np.int64),
        np.float32(0.5),
        np.complex64(1 + 2j),
        complex(1, 2),
    ],
)
def test_finite_numpy_numeric_types_still_allowed(valid: Any) -> None:
    Trainer._require_finite_state_tree({"leaf": [valid]}, "checkpoint scheduler")


@pytest.mark.parametrize(
    "bad", [np.asarray([float("nan")], dtype=np.float32), np.float32(float("inf"))],
    ids=["numpy-array", "numpy-scalar"],
)
def test_direct_export_does_not_publish_detached_nonfinite_scheduler_state(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
    bad: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=2, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    scheduler = trainer.scheduler
    assert scheduler is not None
    normal_export = scheduler.state_dict
    calls: list[int] = []

    def detached_scheduler_state() -> dict[str, Any]:
        state = copy.deepcopy(normal_export())
        calls.append(1)
        if len(calls) == 2:
            state["numpy_diagnostics"] = bad
        return state

    monkeypatch.setattr(scheduler, "state_dict", detached_scheduler_state)
    with pytest.raises(NonFiniteTrainingError, match="checkpoint scheduler has non-finite"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


@pytest.mark.parametrize(
    "bad", [np.asarray([float("nan")], dtype=np.float32), np.float32(float("inf"))],
    ids=["numpy-array", "numpy-scalar"],
)
def test_d05_preflight_rejects_numpy_scheduler_state_before_optimizer_probe(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
    bad: Any,
) -> None:
    config = TrainerConfig(seed=703, max_steps=2, scheduler="cosine")
    source = Trainer(_TinyLogits(), config, device="cpu")
    clean = asdict(source.state_dict())
    invalid = copy.deepcopy(clean)
    invalid["scheduler"]["numpy_diagnostics"] = bad
    target = Trainer(_TinyLogits(), config, device="cpu")
    calls: list[bool] = []

    def forbidden_preflight(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("numpy corruption must fail before optimizer preflight")

    monkeypatch.setattr(trainer_adapter, "_preflight_optimizer_state", forbidden_preflight)
    with pytest.raises(
        CheckpointCompatibilityError, match="checkpoint trainer scheduler has non-finite",
    ):
        trainer_adapter._preflight_trainer_state_without_rng_guard(target, invalid)
    assert not calls and not target.optimizer.state
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    monkeypatch.undo()
    trainer_adapter._preflight_trainer_state_without_rng_guard(target, clean)
