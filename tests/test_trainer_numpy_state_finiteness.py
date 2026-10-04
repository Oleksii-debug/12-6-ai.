"""Reject non-finite NumPy leaves in direct D02 export and D05 preflight.

Synthetic CPU probes only; these are not optimizer exposure or real training evidence.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIdentity
from twelve_six.checkpoint import core, progress_trainer, trainer_adapter
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


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-replay"])
def test_sealed_numpy_scheduler_nan_rejected_by_both_public_d05_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
    loader: Any,
    restore_rng: bool,
) -> None:
    """Valid SHA proves integrity, not finite metadata or safe resumed weights."""
    config = TrainerConfig(seed=703, max_steps=2, scheduler="cosine")
    source_model = _TinyLogits()
    source = Trainer(source_model, config, device="cpu")
    valid = asdict(source.state_dict())
    invalid = copy.deepcopy(valid)
    invalid["scheduler"]["numpy_diagnostics"] = np.asarray(
        [float("nan")], dtype=np.float32,
    )
    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "numpy-scheduler-preapply", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler={"name": "cosine"},
        environment_lock_hash="f" * 64,
    )
    invalid_path = tmp_path / "sealed-numpy-invalid-дані"
    valid_path = tmp_path / "sealed-numpy-valid-дані"
    for path, payload in ((invalid_path, invalid), (valid_path, valid)):
        core.save_checkpoint(
            path, model=source_model, trainer_state=payload, identity=identity,
        )
        core.verify_checkpoint(path)

    target_model = _TinyLogits()
    target = Trainer(target_model, config, device="cpu")
    weights_before = target_model.weight.detach().clone()
    reached_apply: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def reject_model_apply(*args: Any, **kwargs: Any) -> None:
        reached_apply.append(True)
        raise AssertionError("non-finite NumPy scheduler reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", reject_model_apply)
    extra = {"expected_step": 0, "expected_tokens_seen": 0} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(
        CheckpointCompatibilityError, match="checkpoint trainer scheduler has non-finite",
    ):
        loader.load_trainer_checkpoint(
            invalid_path, model=target_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert reached_apply == []
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert not target.optimizer.state and target_model.weight.grad is None
    torch.testing.assert_close(target_model.weight, weights_before, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target_model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([float("nan")], dtype=np.float32),
        np.float32(float("inf")),
        {"nested_moment": np.asarray([float("nan")], dtype=np.float64)},
        float("nan"),
    ],
    ids=["numpy-array", "numpy-scalar", "nested-numpy", "python-float"],
)
def test_effectful_optimizer_numpy_corruption_never_earns_step_credit(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
    bad: Any,
) -> None:
    trainer = Trainer(_TinyLogits(), TrainerConfig(seed=703, max_steps=2), device="cpu")
    original_step = trainer.optimizer.step
    weight = trainer.model.weight

    def inject_invalid_numeric_state(*args: Any, **kwargs: Any):
        result = original_step(*args, **kwargs)
        trainer.optimizer.state[weight]["numpy_diagnostics"] = copy.deepcopy(bad)
        return result

    monkeypatch.setattr(trainer.optimizer, "step", inject_invalid_numeric_state)
    with pytest.raises(NonFiniteTrainingError, match="optimizer produced non-finite state"):
        trainer.train_microbatch(_BATCH)
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 0, 2)
    assert trainer._failure_reason is not None and trainer._update_incomplete
    assert trainer.model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_effectful_optimizer_finite_numpy_state_preserves_step_credit(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
) -> None:
    trainer = Trainer(_TinyLogits(), TrainerConfig(seed=703, max_steps=2), device="cpu")
    original_step = trainer.optimizer.step
    weight = trainer.model.weight

    def add_valid_numeric_state(*args: Any, **kwargs: Any):
        result = original_step(*args, **kwargs)
        trainer.optimizer.state[weight]["numpy_diagnostics"] = np.asarray(
            [0.5, 1.0], dtype=np.float32,
        )
        return result

    monkeypatch.setattr(trainer.optimizer, "step", add_valid_numeric_state)
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    assert trainer._failure_reason is None and not trainer._update_incomplete
    state = trainer.state_dict()
    moment = state.optimizer["state"][0]["numpy_diagnostics"]
    np.testing.assert_array_equal(moment, np.asarray([0.5, 1.0], dtype=np.float32))
