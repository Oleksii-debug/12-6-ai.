"""NumPy optimizer leaves must be finite before training or physical resume.

Tiny synthetic CPU tests only; no authorized data, trained model or scale credit.
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

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    core,
    progress_trainer,
    trainer_adapter,
)
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


@pytest.fixture(autouse=True)
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
    "bad_leaf",
    [
        np.float32(float("nan")),
        np.float64(float("inf")),
        np.complex64(complex(1.0, float("inf"))),
        np.array([0.0, float("nan"), 1.0], dtype=np.float32),
        np.array([[0.0, float("inf")], [1.0, 2.0]], dtype=np.float64)[:, ::-1],
        np.array([1.0 + 0j, 1.0 + complex(0.0, float("nan"))]),
    ],
)
def test_numpy_nonfinite_leaves_refused_without_unbounded_copy(bad_leaf: Any) -> None:
    with pytest.raises(NonFiniteTrainingError, match="non-finite state"):
        Trainer._require_finite_state_tree({"nested": [bad_leaf]}, "checkpoint optimizer")


@pytest.mark.parametrize(
    "good_leaf",
    [
        np.float32(0.0),
        np.complex64(1.0 + 2.0j),
        np.int64(3),
        np.array([1.0, 2.0, 3.0], dtype=np.float32)[::-1],
        np.array([], dtype=np.float64),
        np.array([0, 1, 2], dtype=np.int64),
    ],
)
def test_numpy_finite_leaves_remain_accepted(good_leaf: Any) -> None:
    Trainer._require_finite_state_tree({"nested": (good_leaf,)}, "checkpoint optimizer")


@pytest.mark.parametrize(
    "bad_leaf",
    [
        np.float32(float("nan")),
        np.array([0.0, float("nan"), 1.0], dtype=np.float32),
    ],
    ids=["numpy-scalar", "numpy-array"],
)
def test_detached_numpy_export_poisoned_without_live_weight_mutation(
    monkeypatch: pytest.MonkeyPatch, bad_leaf: Any,
) -> None:
    model = _TinyLogits()
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=2), device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    weights_before = model.weight.detach().clone()
    original_moment = trainer.optimizer.state[model.weight]["exp_avg"].clone()
    real_export = trainer.optimizer.state_dict

    def corrupted_detached_export() -> dict[str, Any]:
        snapshot = copy.deepcopy(real_export())
        next(iter(snapshot["state"].values()))["exp_avg"] = copy.deepcopy(bad_leaf)
        return snapshot

    monkeypatch.setattr(trainer.optimizer, "state_dict", corrupted_detached_export)
    with pytest.raises(NonFiniteTrainingError, match="checkpoint optimizer has non-finite"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    torch.testing.assert_close(model.weight, weights_before, rtol=0, atol=0)
    torch.testing.assert_close(
        trainer.optimizer.state[model.weight]["exp_avg"], original_moment, rtol=0, atol=0,
    )
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_live_numpy_moment_cannot_earn_checkpoint_credit() -> None:
    model = _TinyLogits()
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=2), device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    trainer.optimizer.state[model.weight]["exp_avg"] = np.array(
        [0.0, float("nan"), 1.0], dtype=np.float32,
    )
    with pytest.raises(NonFiniteTrainingError, match="optimizer produced non-finite state"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert trainer.optimizer_step == 1 and model.weight.grad is None


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "numpy-nonfinite-preapply", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=703,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_checksum_valid_numpy_nan_moment_refused_before_model_apply_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
) -> None:
    source_model = _TinyLogits()
    config = TrainerConfig(seed=703, max_steps=2)
    source = Trainer(source_model, config, device="cpu")
    assert source.train_microbatch(_BATCH).optimizer_stepped
    good = asdict(source.state_dict())
    bad = copy.deepcopy(good)
    next(iter(bad["optimizer"]["state"].values()))["exp_avg"] = np.array(
        [0.0, float("nan"), 1.0], dtype=np.float32,
    )
    invalid = tmp_path / "bad-numpy-дані з пробілами"
    valid = tmp_path / "valid-numpy-дані з пробілами"
    for path, state in ((invalid, bad), (valid, good)):
        core.save_checkpoint(path, model=source_model, trainer_state=state, identity=_identity())
        core.verify_checkpoint(path)

    target_model = _TinyLogits()
    target = Trainer(target_model, config, device="cpu")
    initial_weights = target_model.weight.detach().clone()
    touched: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        touched.append(True)
        raise AssertionError("invalid optimizer moments reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = {"expected_step": 1, "expected_tokens_seen": 2} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(CheckpointCompatibilityError, match="non-finite"):
        loader.load_trainer_checkpoint(
            invalid, model=target_model, trainer=target,
            strict_model=False, restore_rng=False, **extra,
        )
    assert touched == []
    assert not target.optimizer.state
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target_model.weight, initial_weights, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid, model=target_model, trainer=target,
        strict_model=False, restore_rng=False, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
    for source_state, target_state in zip(
        source.optimizer.state.values(), target.optimizer.state.values(), strict=True,
    ):
        for name in source_state:
            torch.testing.assert_close(source_state[name], target_state[name], rtol=0, atol=0)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
