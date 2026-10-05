"""Reject checksum-valid scheduler epoch substitutions before D05 applies model state.

All fixtures are synthetic CPU; no corpus or real learned-weight evidence.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIdentity
from twelve_six.checkpoint import core, progress_trainer, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


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
def preserve_process_state():
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
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


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "scheduler-epoch-preapply", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=703,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler={"name": "cosine"},
        environment_lock_hash="f" * 64,
    )


def _source() -> Trainer:
    source = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=703, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert source.scheduler is not None and source.scheduler.last_epoch == 1
    return source


@pytest.mark.parametrize("forged_epoch", [3, False, 1.0])
def test_direct_d02_restore_rejects_finite_epoch_before_optimizer_mutation(
    forged_epoch: Any,
) -> None:
    source = _source()
    valid = asdict(source.state_dict())
    tampered = copy.deepcopy(valid)
    tampered["scheduler"]["last_epoch"] = forged_epoch
    target = Trainer(_TinyLogits(), source.config, device="cpu")
    with pytest.raises(ValueError, match="checkpoint scheduler chronology"):
        target.load_state_dict(tampered)
    assert not target.optimizer.state
    assert target._failure_reason is None
    assert not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(valid)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.scheduler is not None and target.scheduler.last_epoch == 1


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
@pytest.mark.parametrize("forged_epoch", [3, False, 1.0], ids=["future", "bool", "float"])
def test_resealed_invalid_scheduler_epoch_fails_before_both_public_model_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    loader: Any, restore_rng: bool, forged_epoch: Any,
) -> None:
    source = _source()
    valid = asdict(source.state_dict())
    tampered = copy.deepcopy(valid)
    tampered["scheduler"]["last_epoch"] = forged_epoch
    invalid_path = tmp_path / "bad-epoch-дані з пробілами"
    valid_path = tmp_path / "valid-epoch-дані з пробілами"
    for path, payload in ((invalid_path, tampered), (valid_path, valid)):
        core.save_checkpoint(
            path, model=source.model, trainer_state=payload, identity=_identity(),
        )
        core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    calls: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("invalid chronology reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(CheckpointCompatibilityError, match="scheduler chronology"):
        loader.load_trainer_checkpoint(
            invalid_path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert not calls and not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target.model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.scheduler is not None and target.scheduler.last_epoch == 1
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)


def test_live_finite_optimizer_lr_drift_refused_before_checkpoint_credit() -> None:
    source = _source()
    assert source.scheduler is not None
    original_lr = source.scheduler.get_last_lr()[0]
    source.optimizer.param_groups[0]["lr"] = original_lr + 0.01
    with pytest.raises(
        RuntimeError, match="default scheduler rate differs",
    ):
        source.state_dict()
    assert source._failure_reason is not None
    assert (source.micro_step, source.optimizer_step, source.tokens_seen) == (1, 1, 2)


@pytest.mark.parametrize("field", ["optimizer_lr", "scheduler_last_lr"])
@pytest.mark.parametrize("loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"])
def test_resealed_finite_lr_inconsistency_rejected_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, loader: Any,
) -> None:
    source = _source()
    payload = asdict(source.state_dict())
    if field == "optimizer_lr":
        payload["optimizer"]["param_groups"][0]["lr"] += 0.01
    else:
        payload["scheduler"]["_last_lr"][0] += 0.01
    path = tmp_path / "inconsistent-rate-дані з пробілами"
    core.save_checkpoint(path, model=source.model, trainer_state=payload, identity=_identity())
    core.verify_checkpoint(path)
    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    model_applied: list[bool] = []

    def fail_if_model_applied(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("invalid rate reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", fail_if_model_applied)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(CheckpointCompatibilityError, match="scheduler chronology"):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=False, **extra,
        )
    assert not model_applied and not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)


@pytest.mark.parametrize("attack", ["paired-rate", "paired-rate-and-base"])
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_resealed_paired_finite_rate_forgery_rejected_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    attack: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    valid = asdict(source.state_dict())
    corrupted = copy.deepcopy(valid)
    group = corrupted["optimizer"]["param_groups"][0]
    scheduler = corrupted["scheduler"]
    group["lr"] += 0.001
    scheduler["_last_lr"][0] += 0.001
    if attack == "paired-rate-and-base":
        group["initial_lr"] *= 2
        scheduler["base_lrs"][0] *= 2
    # These cross-field equalities were previously sufficient despite
    # disagreeing with the configured schedule and actual committed step.
    assert group["lr"] == scheduler["_last_lr"][0]
    invalid_path = tmp_path / "paired-forgery-дані з пробілами"
    valid_path = tmp_path / "valid-retry-дані з пробілами"
    for path, payload in ((invalid_path, corrupted), (valid_path, valid)):
        core.save_checkpoint(
            path, model=source.model, trainer_state=payload, identity=_identity(),
        )
        core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def reject_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("paired finite-rate forgery reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", reject_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(CheckpointCompatibilityError, match="scheduler chronology"):
        loader.load_trainer_checkpoint(
            invalid_path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target.model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert target.scheduler is not None and source.scheduler is not None
    assert target.scheduler.last_epoch == source.scheduler.last_epoch == 2
    assert target.scheduler.get_last_lr() == source.scheduler.get_last_lr()
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)

@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_resealed_unscheduled_default_rate_forgery_fails_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: Any, restore_rng: bool,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=3, scheduler="constant", learning_rate=0.01,
    )
    source = Trainer(_TinyLogits(), config, device="cpu")
    assert source.scheduler is None
    assert source.train_microbatch(_BATCH).optimizer_stepped
    valid = asdict(source.state_dict())
    tampered = copy.deepcopy(valid)
    tampered["optimizer"]["param_groups"][0]["lr"] = 0.12
    invalid_path = tmp_path / "bad-constant-rate-дані з пробілами"
    valid_path = tmp_path / "good-constant-rate-дані з пробілами"
    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "constant-rate-preapply", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3, "scheduler": "constant", "learning_rate": 0.01},
        seed=703,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler={"name": "constant"},
        environment_lock_hash="f" * 64,
    )
    for path, payload in ((invalid_path, tampered), (valid_path, valid)):
        core.save_checkpoint(
            path, model=source.model, trainer_state=payload, identity=identity,
        )
        core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("invalid constant LR reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(
        CheckpointCompatibilityError, match="scheduler chronology mismatch",
    ):
        loader.load_trainer_checkpoint(
            invalid_path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert not model_applied and not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target.model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.scheduler is None and source.scheduler is None
    assert target.optimizer.param_groups[0]["lr"] == source.optimizer.param_groups[0]["lr"]
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)

@pytest.mark.parametrize("schedule", ["constant", "cosine"])
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_resealed_default_adamw_finite_decay_forgery_rejected_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    schedule: str, loader: Any, restore_rng: bool,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=3, scheduler=schedule, learning_rate=0.01,
    )
    source = Trainer(_TinyLogits(), config, device="cpu")
    assert source.train_microbatch(_BATCH).optimizer_stepped
    valid = asdict(source.state_dict())
    invalid = copy.deepcopy(valid)
    invalid["optimizer"]["param_groups"][0]["weight_decay"] = 0.5
    bad_path = tmp_path / "bad-adamw-option-дані з пробілами"
    good_path = tmp_path / "good-adamw-option-дані з пробілами"
    identity = replace(
        _identity(),
        scheduler={"name": schedule},
        training_config={"steps": 3, "scheduler": schedule, "learning_rate": 0.01},
    )
    for path, payload in ((bad_path, invalid), (good_path, valid)):
        core.save_checkpoint(
            path, model=source.model, trainer_state=payload, identity=identity,
        )
        core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        applied.append(True)
        raise AssertionError("invalid AdamW option reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(
        CheckpointCompatibilityError, match="scheduler chronology mismatch",
    ):
        loader.load_trainer_checkpoint(
            bad_path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert not applied and not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        good_path, model=target.model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.optimizer.param_groups[0]["weight_decay"] == 0.0
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_noncallable_trainer_loader_fails_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "noncallable-loader-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.load_state_dict = None  # type: ignore[method-assign]
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("non-callable trainer loader reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(TypeError, match="trainer must provide load_state_dict"):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
@pytest.mark.parametrize(
    "authority",
    [
        "_require_finite_auxiliary_state",
        "_require_safe_optimizer_hyperparameters",
        "_require_finite_committed_update",
        "_require_no_residual_model_gradients",
        "_require_deterministic_policy",
        "_require_optimizer_parameter_coverage",
    ],
)
def test_late_missing_d02_authority_fails_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    loader: Any, restore_rng: bool, authority: str,
) -> None:
    source = _source()
    path = tmp_path / "late-authority-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    actual_prepare = loader._prepare_model_weights
    model_applied: list[bool] = []

    def prepare_then_shadow(*args: Any, **kwargs: Any) -> Any:
        materialized = actual_prepare(*args, **kwargs)
        setattr(target, authority, None)
        return materialized

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("late missing D02 authority reached model application")

    monkeypatch.setattr(loader, "_prepare_model_weights", prepare_then_shadow)
    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(core.CheckpointCompatibilityError, match="authority unavailable"):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_late_trainer_model_rebind_fails_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "late-model-rebind-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_model = target.model
    initial_weights = original_model.weight.detach().clone()
    actual_prepare = loader._prepare_model_weights
    model_applied: list[bool] = []
    replacement: list[Any] = []

    def prepare_then_rebind(*args: Any, **kwargs: Any) -> Any:
        materialized = actual_prepare(*args, **kwargs)
        rebound = _TinyLogits()
        replacement.append(rebound)
        target.model = rebound
        return materialized

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("late trainer model rebind reached model application")

    monkeypatch.setattr(loader, "_prepare_model_weights", prepare_then_rebind)
    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="owns a different model",
    ):
        loader.load_trainer_checkpoint(
            path, model=original_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert replacement and target.model is replacement[0]
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(original_model.weight, initial_weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("config", "trainer config mismatch"),
        ("optimizer-loader", "optimizer must provide state_dict/load_state_dict"),
        ("scheduler", "scheduler state/config mismatch"),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_late_trainer_state_drift_fails_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    drift: str, error: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "late-state-drift-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    actual_prepare = loader._prepare_model_weights
    model_applied: list[bool] = []

    def prepare_then_drift(*args: Any, **kwargs: Any) -> Any:
        materialized = actual_prepare(*args, **kwargs)
        if drift == "config":
            target.config = replace(target.config, max_steps=target.config.max_steps + 1)
        elif drift == "optimizer-loader":
            target.optimizer.load_state_dict = None  # type: ignore[method-assign]
        elif drift == "scheduler":
            target.scheduler = None
        else:
            raise AssertionError(f"unknown drift fixture: {drift}")
        return materialized

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("late trainer-state drift reached model application")

    monkeypatch.setattr(loader, "_prepare_model_weights", prepare_then_drift)
    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(core.CheckpointCompatibilityError, match=error):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("micro-step", "fresh trainer with no consumed exposure"),
        ("pending-tokens", "fresh trainer with no consumed exposure"),
        ("gradient", "fresh trainer with no pending gradients"),
        ("policy", "live torch deterministic policy disagrees"),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_late_target_freshness_drift_fails_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    drift: str, error: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "late-target-drift-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    actual_prepare = loader._prepare_model_weights
    model_applied: list[bool] = []

    def prepare_then_drift(*args: Any, **kwargs: Any) -> Any:
        materialized = actual_prepare(*args, **kwargs)
        if drift == "micro-step":
            target.micro_step = 1
        elif drift == "pending-tokens":
            target._pending_tokens = 1
        elif drift == "gradient":
            target.model.weight.grad = torch.ones_like(target.model.weight)
        elif drift == "policy":
            torch.use_deterministic_algorithms(
                not target.config.deterministic_algorithms,
                warn_only=target.config.deterministic_warn_only,
            )
        else:
            raise AssertionError(f"unknown target drift fixture: {drift}")
        return materialized

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("late target freshness drift reached model application")

    monkeypatch.setattr(loader, "_prepare_model_weights", prepare_then_drift)
    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(core.CheckpointCompatibilityError, match=error):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("hook_effect", "error"),
    [
        ("model-rebind", "owns a different model"),
        ("micro-step", "fresh trainer with no consumed exposure"),
        ("scheduler-rebind", "scheduler binding changed"),
        ("config-rebind", "config binding changed"),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_late_stateful_preflight_hook_drift_is_rechecked_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    hook_effect: str, error: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "late-preflight-hook-drift-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_model = target.model
    initial_weights = original_model.weight.detach().clone()
    actual_prepare = loader._prepare_model_weights
    model_applied: list[bool] = []

    def prepare_then_arm_hook(*args: Any, **kwargs: Any) -> Any:
        materialized = actual_prepare(*args, **kwargs)
        assert target.scheduler is not None
        actual_state_dict = target.scheduler.state_dict

        def effectful_state_dict() -> Any:
            state = actual_state_dict()
            if hook_effect == "model-rebind":
                target.model = _TinyLogits()
            elif hook_effect == "micro-step":
                target.micro_step = 1
            elif hook_effect == "scheduler-rebind":
                target.scheduler = None
            elif hook_effect == "config-rebind":
                target.config = replace(target.config)
            else:
                raise AssertionError(f"unknown hook effect: {hook_effect}")
            return state

        target.scheduler.state_dict = effectful_state_dict  # type: ignore[method-assign]
        return materialized

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("stateful preflight hook drift reached model application")

    monkeypatch.setattr(loader, "_prepare_model_weights", prepare_then_arm_hook)
    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(core.CheckpointCompatibilityError, match=error):
        loader.load_trainer_checkpoint(
            path, model=original_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    torch.testing.assert_close(original_model.weight, initial_weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("hook_effect", "error"),
    [
        ("micro-step", "fresh trainer with no consumed exposure"),
        ("scheduler-rebind", "scheduler binding changed"),
        ("drop-failure-marker", "safety classification changed"),
        ("drop-incomplete-marker", "safety classification changed"),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_final_trainer_loader_bind_drift_fails_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    hook_effect: str, error: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "late-loader-bind-drift-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_model = target.model
    initial_weights = original_model.weight.detach().clone()
    actual_bind = loader._bind_trainer_state_loader
    bind_calls = 0
    model_applied: list[bool] = []

    def bind_then_drift(trainer: Any) -> Any:
        nonlocal bind_calls
        bound = actual_bind(trainer)
        bind_calls += 1
        if bind_calls == 2:
            if hook_effect == "micro-step":
                target.micro_step = 1
            elif hook_effect == "scheduler-rebind":
                target.scheduler = None
            elif hook_effect == "drop-failure-marker":
                del target._failure_reason
            elif hook_effect == "drop-incomplete-marker":
                del target._update_incomplete
            else:
                raise AssertionError(f"unknown bind hook effect: {hook_effect}")
        return bound

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("final trainer-loader bind drift reached model application")

    monkeypatch.setattr(loader, "_bind_trainer_state_loader", bind_then_drift)
    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(core.CheckpointCompatibilityError, match=error):
        loader.load_trainer_checkpoint(
            path, model=original_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert bind_calls == 2
    assert model_applied == []
    assert not target.optimizer.state
    torch.testing.assert_close(original_model.weight, initial_weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("attack", "error"),
    [
        ("model", "model binding changed"),
        ("optimizer", "optimizer binding changed"),
        ("scheduler", "scheduler binding changed"),
        ("scaler", "scaler binding changed"),
        ("config", "config binding changed"),
        ("failure-marker", "safety classification changed"),
        ("incomplete-marker", "safety classification changed"),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_model_apply_hook_drift_is_detected_before_trainer_state_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    attack: str, error: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "model-apply-hook-drift-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_model = target.model
    original_optimizer = target.optimizer
    original_scheduler = target.scheduler
    original_scaler = target.scaler
    original_config = target.config
    initial_weights = original_model.weight.detach().clone()
    actual_model_loader = original_model.load_state_dict

    def model_loader_with_drift(
        state: Any, *, strict: bool = True,
    ) -> Any:
        result = actual_model_loader(state, strict=strict)
        if attack == "model":
            target.model = _TinyLogits()
        elif attack == "optimizer":
            target.optimizer = object()  # type: ignore[assignment]
        elif attack == "scheduler":
            target.scheduler = object()  # type: ignore[assignment]
        elif attack == "scaler":
            target.scaler = object()  # type: ignore[assignment]
        elif attack == "config":
            target.config = replace(target.config)
        elif attack == "failure-marker":
            del target._failure_reason
        elif attack == "incomplete-marker":
            del target._update_incomplete
        else:
            raise AssertionError(f"unknown model-loader hook attack: {attack}")
        return result

    original_model.load_state_dict = model_loader_with_drift  # type: ignore[method-assign]
    trainer_state_applied: list[bool] = []
    actual_trainer_loader = target.load_state_dict

    def track_trainer_load(state: Any) -> None:
        trainer_state_applied.append(True)
        actual_trainer_loader(state)

    target.load_state_dict = track_trainer_load  # type: ignore[method-assign]
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(core.CheckpointCompatibilityError, match=error):
        loader.load_trainer_checkpoint(
            path, model=original_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert trainer_state_applied == []
    assert target._failure_reason is not None
    assert target._update_incomplete
    assert not original_optimizer.state
    assert not torch.equal(original_model.weight, initial_weights)
    torch.testing.assert_close(original_model.weight, source.model.weight, rtol=0, atol=0)

    target.model = original_model
    target.optimizer = original_optimizer
    target.scheduler = original_scheduler
    target.scaler = original_scaler
    target.config = original_config
    target.load_state_dict = actual_trainer_loader  # type: ignore[method-assign]
    original_model.load_state_dict = actual_model_loader  # type: ignore[method-assign]
    retry_model_applied: list[bool] = []

    def forbid_retry_model_application(*args: Any, **kwargs: Any) -> None:
        retry_model_applied.append(True)
        raise AssertionError("poisoned model-hook retry reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_retry_model_application)
    with pytest.raises(core.CheckpointCompatibilityError, match="target trainer is poisoned"):
        loader.load_trainer_checkpoint(
            path, model=original_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert retry_model_applied == []


@pytest.mark.parametrize(
    "marker", ["_failure_reason", "_update_incomplete"],
    ids=["failure-marker", "incomplete-marker"],
)
@pytest.mark.parametrize("outcome", ["raise", "return"], ids=["raises", "returns"])
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_apply_stage_marker_loss_is_repaired_and_same_instance_retry_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    marker: str, outcome: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "apply-marker-loss-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    actual_loader = target.load_state_dict

    def loader_with_marker_loss(state: Any) -> None:
        delattr(target, marker)
        if outcome == "raise":
            raise RuntimeError("injected trainer apply failure")

    target.load_state_dict = loader_with_marker_loss  # type: ignore[method-assign]
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    error: type[BaseException]
    message: str
    if outcome == "raise":
        error, message = RuntimeError, "injected trainer apply failure"
    else:
        error, message = core.CheckpointCompatibilityError, "safety classification changed"

    with pytest.raises(error, match=message):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert hasattr(target, "_failure_reason")
    assert hasattr(target, "_update_incomplete")
    assert target._failure_reason is not None
    assert target._update_incomplete
    assert not target.optimizer.state
    assert not torch.equal(target.model.weight, initial_weights)
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)

    target.load_state_dict = actual_loader  # type: ignore[method-assign]
    retry_model_applied: list[bool] = []

    def forbid_retry_model_application(*args: Any, **kwargs: Any) -> None:
        retry_model_applied.append(True)
        raise AssertionError("poisoned same-instance retry reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_retry_model_application)
    with pytest.raises(core.CheckpointCompatibilityError, match="target trainer is poisoned"):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert retry_model_applied == []


@pytest.mark.parametrize(
    "marker", ["_failure_reason", "_update_incomplete"],
    ids=["failure-marker", "incomplete-marker"],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_preflight_rng_rollback_marker_loss_still_poisons_canonical_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    marker: str, loader: Any,
) -> None:
    source = _source()
    path = tmp_path / "preflight-rng-marker-loss-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    actual_restore = core.restore_rng_state
    restore_calls = 0

    def fail_first_restore(state: Any) -> None:
        nonlocal restore_calls
        restore_calls += 1
        if restore_calls == 1:
            delattr(target, marker)
            raise RuntimeError("injected preflight RNG rollback failure")
        actual_restore(state)

    monkeypatch.setattr(core, "restore_rng_state", fail_first_restore)
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("preflight RNG rollback failure reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(RuntimeError, match="injected preflight RNG rollback failure"):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=False, **extra,
        )

    assert restore_calls >= 1
    assert model_applied == []
    assert hasattr(target, "_failure_reason")
    assert hasattr(target, "_update_incomplete")
    assert target._failure_reason == "checkpoint_preflight_rng_rollback_failed"
    assert target._update_incomplete
    assert not target.optimizer.state
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_late_noncallable_trainer_loader_fails_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "late-noncallable-loader-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    actual_prepare = loader._prepare_model_weights
    model_applied: list[bool] = []

    def prepare_then_disable(*args: Any, **kwargs: Any) -> Any:
        materialized = actual_prepare(*args, **kwargs)
        target.load_state_dict = None  # type: ignore[method-assign]
        return materialized

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("late non-callable trainer loader reached model application")

    monkeypatch.setattr(loader, "_prepare_model_weights", prepare_then_disable)
    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(TypeError, match="trainer must provide load_state_dict"):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)


def test_noncallable_trainer_state_dict_refuses_save_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = Trainer(_TinyLogits(), TrainerConfig(max_steps=1, seed=703), device="cpu")
    trainer.state_dict = None  # type: ignore[method-assign]
    published: list[bool] = []

    def forbid_publication(*args: Any, **kwargs: Any) -> None:
        published.append(True)
        raise AssertionError("non-callable trainer state_dict reached checkpoint publication")

    monkeypatch.setattr(trainer_adapter, "save_checkpoint", forbid_publication)
    with pytest.raises(TypeError, match="trainer must provide state_dict"):
        trainer_adapter.save_trainer_checkpoint(
            tmp_path / "noncallable-save-дані з пробілами",
            model=trainer.model,
            trainer=trainer,
            identity=_identity(),
        )

    assert published == []


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("lr", -0.01),
        ("lr", "0.01"),
        ("weight_decay", -0.1),
        ("weight_decay", "0.1"),
        ("eps", 0.0),
        ("eps", "1e-8"),
        ("betas", (1.0, 0.999)),
        ("betas", ("0.9", "0.999")),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_resealed_invalid_optimizer_hyperparameters_fail_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, bad_value: Any, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    invalid = asdict(source.state_dict())
    invalid["optimizer"]["param_groups"][0][field] = bad_value
    path = tmp_path / "bad-optimizer-hyperparameters-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=invalid, identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("invalid optimizer hyperparameters reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(
        CheckpointCompatibilityError,
        match="optimizer hyperparameters invalid",
    ):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_noncallable_optimizer_zero_grad_fails_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "noncallable-zero-grad-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.optimizer.zero_grad = None  # type: ignore[method-assign]
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("non-callable optimizer zero_grad reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(
        CheckpointCompatibilityError,
        match="optimizer zero_grad unavailable",
    ):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize("component_name", ["scheduler", "scaler"])
@pytest.mark.parametrize("component_method", ["state_dict", "load_state_dict"])
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_noncallable_stateful_component_interface_fails_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    component_name: str, component_method: str,
    loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "noncallable-stateful-interface-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    component = getattr(target, component_name)
    assert component is not None
    setattr(component, component_method, None)
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("non-callable stateful interface reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(
        CheckpointCompatibilityError,
        match=rf"{component_name} must provide state_dict/load_state_dict",
    ):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_method", ["load_state_dict", "state_dict"])
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_noncallable_optimizer_checkpoint_interface_fails_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    optimizer_method: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "noncallable-optimizer-interface-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    setattr(target.optimizer, optimizer_method, None)
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("non-callable optimizer interface reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(
        CheckpointCompatibilityError,
        match="optimizer must provide state_dict/load_state_dict",
    ):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("authority", "message"),
    [
        ("_require_finite_state_tree", "numeric-state authority unavailable"),
        ("_require_checkpoint_scheduler_chronology", "scheduler authority unavailable"),
        ("_require_optimizer_state_parameter_order", "optimizer-order authority unavailable"),
        ("_require_finite_auxiliary_state", "auxiliary-state authority unavailable"),
        (
            "_require_safe_optimizer_hyperparameters",
            "optimizer-hyperparameter authority unavailable",
        ),
        ("_require_finite_committed_update", "committed-update authority unavailable"),
        ("_require_no_residual_model_gradients", "gradient-cleanliness authority unavailable"),
        ("_require_deterministic_policy", "deterministic-policy authority unavailable"),
        (
            "_require_optimizer_parameter_coverage",
            "optimizer-coverage authority unavailable",
        ),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_canonical_d02_missing_mandatory_authority_fails_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    authority: str, message: str, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    path = tmp_path / "missing-canonical-authority-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    setattr(target, authority, None)
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("missing canonical authority reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(CheckpointCompatibilityError, match=message):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_canonical_d02_missing_scaler_authority_fails_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    source.scaler = torch.amp.GradScaler("cpu", enabled=True)
    invalid = asdict(source.state_dict())
    invalid["scaler"]["growth_factor"] = 1.0
    path = tmp_path / "missing-scaler-authority-дані з пробілами"
    core.save_checkpoint(
        path, model=source.model, trainer_state=invalid, identity=_identity(),
    )
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.scaler = torch.amp.GradScaler("cpu", enabled=True)
    target._require_checkpoint_scaler_state = None  # type: ignore[method-assign]
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("missing scaler authority reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(
        CheckpointCompatibilityError, match="scaler authority unavailable"
    ):
        loader.load_trainer_checkpoint(
            path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("scale", -1.0),
        ("growth_factor", 1.0),
        ("backoff_factor", 0.0),
        ("growth_interval", 0),
        ("_growth_tracker", -1),
        ("_growth_tracker", 2000),
        ("scale", 1e-300),
        ("scale", 1e-40),  # float32-positive scale with overflowing reciprocal
        ("scale", 1e300),
        ("growth_factor", 1.000000000000001),
        ("growth_factor", 1e300),
        ("backoff_factor", 1e-300),
        ("backoff_factor", 0.999999999999999),
    ],
)
@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_resealed_invalid_scaler_statistics_fail_before_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, bad_value: Any, loader: Any, restore_rng: bool,
) -> None:
    # Enabled CPU scaler exposes GradScaler's real serialized schema without
    # claiming CUDA fp16 or any real learned-model training.
    source = _source()
    source.scaler = torch.amp.GradScaler("cpu", enabled=True)
    valid = asdict(source.state_dict())
    invalid = copy.deepcopy(valid)
    invalid["scaler"][field] = bad_value
    bad_path = tmp_path / "bad-scaler-дані з пробілами"
    good_path = tmp_path / "good-scaler-дані з пробілами"
    for path, payload in ((bad_path, invalid), (good_path, valid)):
        core.save_checkpoint(
            path, model=source.model, trainer_state=payload, identity=_identity(),
        )
        core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.scaler = torch.amp.GradScaler("cpu", enabled=True)
    initial_weights = target.model.weight.detach().clone()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    model_applied: list[bool] = []

    def forbid_model_application(*args: Any, **kwargs: Any) -> None:
        model_applied.append(True)
        raise AssertionError("invalid scaler reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_application)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    with pytest.raises(CheckpointCompatibilityError, match="scaler statistics invalid"):
        loader.load_trainer_checkpoint(
            bad_path, model=target.model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )
    assert model_applied == []
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        good_path, model=target.model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.scaler.state_dict() == source.scaler.state_dict()
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)
    # The verified retry must also preserve the next real synthetic AdamW
    # update and enabled CPU scaler growth chronology after restoration.
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert source.optimizer_step == target.optimizer_step == 2
    assert source.scaler.state_dict() == target.scaler.state_dict()
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)

@pytest.mark.parametrize(
    ("field", "bad_value"), [
        ("scale", 1e-300),
        ("scale", 1e-40),  # float32-positive scale with overflowing reciprocal
        ("scale", 1e300),
        ("growth_factor", 1.000000000000001),
        ("growth_factor", 1e300),
        ("backoff_factor", 1e-300),
        ("backoff_factor", 0.999999999999999),
    ],
)
def test_direct_d02_rejects_float32_invalid_scaler_before_optimizer_mutation(
    field: str, bad_value: float,
) -> None:
    source = _source()
    source.scaler = torch.amp.GradScaler("cpu", enabled=True)
    valid = asdict(source.state_dict())
    bad = copy.deepcopy(valid)
    bad["scaler"][field] = bad_value

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.scaler = torch.amp.GradScaler("cpu", enabled=True)
    before_weights = target.model.weight.detach().clone()
    with pytest.raises(ValueError, match="scaler checkpoint statistics invalid in float32"):
        target.load_state_dict(bad)
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, before_weights, rtol=0, atol=0)

    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(valid)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.scaler.state_dict() == source.scaler.state_dict()
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert source.optimizer_step == target.optimizer_step == 2
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)

def test_direct_d02_accepts_small_scaler_with_finite_float32_inverse() -> None:
    # The new bound is on the usable inverse, not an arbitrary large minimum.
    # 1e-38 is float32-representable and its reciprocal is still finite.
    source = _source()
    source.scaler = torch.amp.GradScaler("cpu", enabled=True)
    state = asdict(source.state_dict())
    state["scaler"]["scale"] = 1e-38

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.scaler = torch.amp.GradScaler("cpu", enabled=True)
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(state)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert bool(torch.isfinite(target.model.weight).all())

def test_live_scaler_with_overflowing_inverse_cannot_publish_checkpoint() -> None:
    source = _source()
    source.scaler = torch.amp.GradScaler("cpu", enabled=True)
    invalid = source.scaler.state_dict()
    invalid["scale"] = 1e-40
    source.scaler.load_state_dict(invalid)

    with pytest.raises(ValueError, match="scaler checkpoint statistics invalid in float32"):
        source.assert_checkpoint_safe()
    assert source._failure_reason is not None
    with pytest.raises(RuntimeError, match="restore a verified checkpoint"):
        source.train_microbatch(_BATCH)

@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "exact-rng"])
def test_d05_accepts_small_scaler_with_finite_float32_inverse(
    tmp_path: Path, loader: Any, restore_rng: bool,
) -> None:
    source = _source()
    source.scaler = torch.amp.GradScaler("cpu", enabled=True)
    state = asdict(source.state_dict())
    state["scaler"]["scale"] = 1e-38
    path = tmp_path / "valid-low-scale-дані з пробілами"
    core.save_checkpoint(path, model=source.model, trainer_state=state, identity=_identity())
    core.verify_checkpoint(path)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.scaler = torch.amp.GradScaler("cpu", enabled=True)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    loader.load_trainer_checkpoint(
        path, model=target.model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert bool(torch.isfinite(target.model.weight).all())
