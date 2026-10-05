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
