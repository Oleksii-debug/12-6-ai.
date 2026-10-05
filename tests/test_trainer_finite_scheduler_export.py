"""Finite but detached scheduler exports must not alter exact-resume chronology.

Synthetic CPU-only regression; does not credit learned data, target exposure or
real training, final-test access or Windows qualification.
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
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
def preserve_state():
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])


@pytest.mark.parametrize(
    "field", ["last_epoch", "_step_count", "base_lrs", "_last_lr"],
)
def test_detached_finite_scheduler_export_poisoned(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any, field: str,
) -> None:
    model = _TinyLogits()
    trainer = Trainer(
        model, TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    scheduler = trainer.scheduler
    assert scheduler is not None
    original_export = scheduler.state_dict
    initial_epoch = scheduler.last_epoch
    initial_weights = model.weight.detach().clone()
    calls: list[int] = []

    def forged_export() -> dict[str, Any]:
        snapshot = copy.deepcopy(original_export())
        calls.append(1)
        if len(calls) == 2:
            if field in {"last_epoch", "_step_count"}:
                snapshot[field] += 2
            else:
                snapshot[field] = [float(snapshot[field][0]) + 0.01]
        return snapshot

    monkeypatch.setattr(scheduler, "state_dict", forged_export)
    with pytest.raises(TrainingStateInvalidError, match="scheduler export"):
        trainer.state_dict()
    assert len(calls) >= 2
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        1, 1, 2,
    )
    assert scheduler.last_epoch == initial_epoch
    torch.testing.assert_close(model.weight, initial_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_detached_finite_lambda_payload_poisoned(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    scheduler = trainer.scheduler
    assert scheduler is not None
    original_export = scheduler.state_dict
    calls: list[int] = []

    def forged_export() -> dict[str, Any]:
        snapshot = copy.deepcopy(original_export())
        calls.append(1)
        if len(calls) == 2:
            snapshot["lr_lambdas"] = [{"finite_but_wrong": 1}]
        return snapshot

    monkeypatch.setattr(scheduler, "state_dict", forged_export)
    with pytest.raises(TrainingStateInvalidError, match="scheduler export"):
        trainer.state_dict()
    assert trainer._failure_reason is not None


def test_normal_lambda_scheduler_export_keeps_resume_state(
    preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    saved = trainer.state_dict()
    assert saved.scheduler is not None
    assert trainer.scheduler is not None
    assert saved.scheduler["last_epoch"] == trainer.scheduler.last_epoch
    assert saved.scheduler["_last_lr"] == trainer.scheduler.get_last_lr()
    assert saved.scheduler["lr_lambdas"] == [None]
    assert trainer._failure_reason is None


@pytest.mark.parametrize("forged_epoch", [0, 3, False, 1.0])
def test_finite_live_scheduler_chronology_cannot_be_saved(
    preserve_state: Any, forged_epoch: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.optimizer_step == 1
    scheduler = trainer.scheduler
    assert scheduler is not None and scheduler.last_epoch == 1
    scheduler.last_epoch = forged_epoch
    with pytest.raises(
        TrainingStateInvalidError,
        match="scheduler chronology differs from committed optimizer step",
    ):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        1, 1, 2,
    )
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)

def test_missing_scheduler_export_is_not_a_valid_none_snapshot(
    monkeypatch: pytest.MonkeyPatch, preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.scheduler is not None
    original_export = trainer.scheduler.state_dict
    calls: list[int] = []

    def missing_export() -> dict[str, Any] | None:
        calls.append(1)
        return None if len(calls) == 2 else original_export()

    monkeypatch.setattr(trainer.scheduler, "state_dict", missing_export)
    with pytest.raises(TrainingStateInvalidError, match="scheduler export"):
        trainer.state_dict()
    assert len(calls) >= 2
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        0, 0, 0,
    )
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_genuinely_absent_scheduler_still_exports_none(
    preserve_state: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=2, scheduler="constant"),
        device="cpu",
    )
    assert trainer.scheduler is None
    assert trainer.state_dict().scheduler is None
    assert trainer._failure_reason is None


@pytest.mark.parametrize("forged_epoch", [0, 2, False, 1.0])
def test_resume_scheduler_chronology_preflight_is_retryable(
    preserve_state: Any, forged_epoch: Any,
) -> None:
    source = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert source.optimizer_step == 1
    saved = source.state_dict()
    assert saved.scheduler is not None and saved.scheduler["last_epoch"] == 1
    forged_scheduler = copy.deepcopy(saved.scheduler)
    forged_scheduler["last_epoch"] = forged_epoch
    corrupted = replace(saved, scheduler=forged_scheduler)

    target = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert not target.optimizer.state
    with pytest.raises(
        ValueError, match="checkpoint scheduler chronology differs from committed optimizer step",
    ):
        target.load_state_dict(corrupted)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (
        0, 0, 0,
    )
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete

    # After preflight rejection, restore corresponding model weights and the
    # unmodified committed trainer state into this same still-fresh target.
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (
        1, 1, 2,
    )
    assert target.scheduler is not None
    assert target.scheduler.last_epoch == 1
    assert target._failure_reason is None and not target._update_incomplete
    assert target.optimizer.param_groups[0]["lr"] == source.optimizer.param_groups[0]["lr"]
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert target.optimizer_step == source.optimizer_step == 2
    assert target.scheduler is not None and source.scheduler is not None
    assert target.scheduler.last_epoch == source.scheduler.last_epoch == 2
    assert target.scheduler._step_count == source.scheduler._step_count == 3
    assert target.scheduler.get_last_lr() == source.scheduler.get_last_lr()
    torch.testing.assert_close(
        target.model.weight, source.model.weight, rtol=0, atol=0,
    )


@pytest.mark.parametrize("attack", ["finite-rate", "wrong-length", "non-list"])
def test_resume_rejects_incoherent_scheduler_last_lr_before_optimizer_apply(
    preserve_state: Any, attack: str,
) -> None:
    source = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    assert saved.scheduler is not None
    assert saved.scheduler["_last_lr"] == [
        group["lr"] for group in saved.optimizer["param_groups"]
    ]
    bad_scheduler = copy.deepcopy(saved.scheduler)
    if attack == "finite-rate":
        bad_scheduler["_last_lr"][0] += 0.01
    elif attack == "wrong-length":
        bad_scheduler["_last_lr"] = []
    else:
        bad_scheduler["_last_lr"] = 0.01
    corrupt = replace(saved, scheduler=bad_scheduler)

    target = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    with pytest.raises(
        ValueError, match="checkpoint scheduler last LR differs from checkpoint optimizer",
    ):
        target.load_state_dict(corrupt)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (
        0, 0, 0,
    )
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete

    # A failed preflight must leave the fresh target reusable.
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == 1
    assert target.scheduler is not None
    assert target.scheduler.get_last_lr() == source.scheduler.get_last_lr()
    assert target._failure_reason is None and not target._update_incomplete
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert target.optimizer_step == source.optimizer_step == 2
    assert target.scheduler is not None and source.scheduler is not None
    assert target.scheduler.last_epoch == source.scheduler.last_epoch == 2
    assert target.scheduler._step_count == source.scheduler._step_count == 3
    assert target.scheduler.get_last_lr() == source.scheduler.get_last_lr()
    torch.testing.assert_close(
        target.model.weight, source.model.weight, rtol=0, atol=0,
    )

@pytest.mark.parametrize("forged_count", [0, 1, 3, False, 2.0])
def test_live_lambda_internal_step_count_cannot_be_saved(
    preserve_state: Any, forged_count: Any,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.scheduler is not None and trainer.scheduler._step_count == 2
    trainer.scheduler._step_count = forged_count
    with pytest.raises(
        TrainingStateInvalidError,
        match="scheduler chronology differs from committed optimizer step",
    ):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (
        1, 1, 2,
    )
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


@pytest.mark.parametrize("forged_count", [0, 1, 3, False, 2.0])
def test_resume_lambda_internal_step_count_preflight_is_retryable(
    preserve_state: Any, forged_count: Any,
) -> None:
    source = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    assert saved.scheduler is not None and saved.scheduler["_step_count"] == 2
    forged_scheduler = copy.deepcopy(saved.scheduler)
    forged_scheduler["_step_count"] = forged_count
    target = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    with pytest.raises(
        ValueError, match="checkpoint scheduler step count differs from committed optimizer step",
    ):
        target.load_state_dict(replace(saved, scheduler=forged_scheduler))
    assert not target.optimizer.state
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (
        0, 0, 0,
    )
    assert target._failure_reason is None and not target._update_incomplete
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == 1
    assert target.scheduler is not None and target.scheduler._step_count == 2
    assert target._failure_reason is None and not target._update_incomplete
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert target.optimizer_step == source.optimizer_step == 2
    assert target.scheduler is not None and source.scheduler is not None
    assert target.scheduler.last_epoch == source.scheduler.last_epoch == 2
    assert target.scheduler._step_count == source.scheduler._step_count == 3
    assert target.scheduler.get_last_lr() == source.scheduler.get_last_lr()
    torch.testing.assert_close(
        target.model.weight, source.model.weight, rtol=0, atol=0,
    )


@pytest.mark.parametrize("mode", ["paired-rate", "all-rate-fields"])
def test_default_schedule_cannot_restore_paired_finite_lr_forgery(
    preserve_state: Any, mode: str,
) -> None:
    source = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=703, max_steps=4, scheduler="cosine", warmup_steps=2),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    corrupt = copy.deepcopy(saved)
    optimizer_state = copy.deepcopy(corrupt.optimizer)
    scheduler_state = copy.deepcopy(corrupt.scheduler)
    assert scheduler_state is not None
    optimizer_state["param_groups"][0]["lr"] += 0.01
    scheduler_state["_last_lr"][0] += 0.01
    if mode == "all-rate-fields":
        optimizer_state["param_groups"][0]["initial_lr"] *= 2
        scheduler_state["base_lrs"][0] *= 2
    corrupt = replace(corrupt, optimizer=optimizer_state, scheduler=scheduler_state)
    # The simpler cross-field equality test accepts this internally
    # consistent but unfaithful rate: config/step are authoritative.
    assert optimizer_state["param_groups"][0]["lr"] == scheduler_state["_last_lr"][0]

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    with pytest.raises(
        TrainingStateInvalidError, match="default scheduler rate differs",
    ):
        target.load_state_dict(corrupt)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (
        0, 0, 0,
    )
    assert not target.optimizer.state
    assert target._failure_reason is None and not target._update_incomplete

    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == 1
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert target.optimizer_step == source.optimizer_step == 2
    assert target.scheduler is not None and source.scheduler is not None
    assert target.scheduler.get_last_lr() == source.scheduler.get_last_lr()
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["paired-rate", "all-rate-fields"])
def test_default_schedule_cannot_publish_paired_finite_live_lr_drift(
    preserve_state: Any, mode: str,
) -> None:
    trainer = Trainer(
        _TinyLogits(), TrainerConfig(seed=703, max_steps=4, scheduler="cosine"),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.scheduler is not None
    trainer.optimizer.param_groups[0]["lr"] += 0.01
    trainer.scheduler._last_lr[0] += 0.01
    if mode == "all-rate-fields":
        trainer.optimizer.param_groups[0]["initial_lr"] *= 2
        trainer.scheduler.base_lrs[0] *= 2
    assert trainer.optimizer.param_groups[0]["lr"] == trainer.scheduler.get_last_lr()[0]
    with pytest.raises(
        TrainingStateInvalidError, match="default scheduler rate differs",
    ):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert trainer.optimizer_step == 1
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


@pytest.mark.parametrize(
    ("kind", "warmup"),
    [("cosine", 0), ("cosine", 2), ("linear_warmup", 2), ("constant", 2)],
)
@pytest.mark.parametrize("completed", [0, 1, 2])
def test_default_schedule_authority_accepts_valid_replay_and_next_step(
    preserve_state: Any, kind: str, warmup: int, completed: int,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=4, scheduler=kind, warmup_steps=warmup,
    )
    source = Trainer(_TinyLogits(), config, device="cpu")
    for _ in range(completed):
        assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    target = Trainer(_TinyLogits(), config, device="cpu")
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == completed
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert target.scheduler is not None and source.scheduler is not None
    assert target.scheduler.last_epoch == source.scheduler.last_epoch
    assert target.scheduler.get_last_lr() == source.scheduler.get_last_lr()
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)


def test_injected_optimizer_does_not_inherit_default_schedule_rate_oracle(
    preserve_state: Any,
) -> None:
    config = TrainerConfig(seed=703, max_steps=4, scheduler="cosine")
    model = _TinyLogits()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    source = Trainer(model, config, optimizer=optimizer, device="cpu")
    assert not source._canonical_default_schedule
    assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    target_model = _TinyLogits()
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.002)
    target = Trainer(target_model, config, optimizer=target_optimizer, device="cpu")
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == 1
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)

@pytest.mark.parametrize("completed", [0, 1, 2])
def test_unscheduled_default_constant_rate_valid_replay(
    preserve_state: Any, completed: int,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=4, scheduler="constant", warmup_steps=0,
        learning_rate=0.01,
    )
    source = Trainer(_TinyLogits(), config, device="cpu")
    assert source.scheduler is None
    for _ in range(completed):
        assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    assert saved.scheduler is None
    target = Trainer(_TinyLogits(), config, device="cpu")
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == completed
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert source.optimizer.param_groups[0]["lr"] == 0.01
    assert target.optimizer.param_groups[0]["lr"] == 0.01
    torch.testing.assert_close(source.model.weight, target.model.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("forged_rate", "expected_error", "expected_message"),
    [
        (0.12, TrainingStateInvalidError, "default constant optimizer rate"),
        (
            False,
            NonFiniteTrainingError,
            "optimizer learning rate type differs from live optimizer",
        ),
        (0.0, TrainingStateInvalidError, "default constant optimizer rate"),
        (
            10 ** 400,
            NonFiniteTrainingError,
            "optimizer learning rate type differs from live optimizer",
        ),
    ],
)
def test_unscheduled_default_constant_rate_direct_resume_rejects_before_apply(
    preserve_state: Any,
    forged_rate: Any,
    expected_error: type[BaseException],
    expected_message: str,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=4, scheduler="constant", learning_rate=0.01,
    )
    source = Trainer(_TinyLogits(), config, device="cpu")
    assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    corrupt_optimizer = copy.deepcopy(saved.optimizer)
    corrupt_optimizer["param_groups"][0]["lr"] = forged_rate
    corrupt = replace(saved, optimizer=corrupt_optimizer)
    target = Trainer(_TinyLogits(), config, device="cpu")
    with pytest.raises(expected_error, match=expected_message):
        target.load_state_dict(corrupt)
    assert not target.optimizer.state
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None and not target._update_incomplete
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == 1
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(source.model.weight, target.model.weight, rtol=0, atol=0)


def test_unscheduled_default_constant_live_rate_forgery_poisoned(
    preserve_state: Any,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=4, scheduler="constant", learning_rate=0.01,
    )
    trainer = Trainer(_TinyLogits(), config, device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    trainer.optimizer.param_groups[0]["lr"] = 0.12
    with pytest.raises(TrainingStateInvalidError, match="default constant optimizer rate"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert trainer.optimizer_step == 1
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_injected_unscheduled_optimizer_retains_custom_constant_rate_policy(
    preserve_state: Any,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=4, scheduler="constant", learning_rate=0.01,
    )
    model = _TinyLogits()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    trainer = Trainer(model, config, optimizer=optimizer, device="cpu")
    assert trainer.scheduler is None
    assert not trainer._canonical_unscheduled_default_optimizer
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    saved = trainer.state_dict()
    assert saved.optimizer["param_groups"][0]["lr"] == 0.02
    target_model = _TinyLogits()
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.02)
    target = Trainer(target_model, config, optimizer=target_optimizer, device="cpu")
    target.model.load_state_dict(model.state_dict())
    target.load_state_dict(saved)
    assert target.optimizer_step == 1
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(trainer.model.weight, target.model.weight, rtol=0, atol=0)

@pytest.mark.parametrize(
    ("option", "wrong"), [
        ("weight_decay", 0.5),
        ("eps", 0.1),
        ("betas", (0.5, 0.9)),
        ("amsgrad", True),
    ],
)
@pytest.mark.parametrize("schedule", ["constant", "cosine"])
def test_default_adamw_finite_option_forgery_refused_before_direct_resume(
    preserve_state: Any, option: str, wrong: Any, schedule: str,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=4, scheduler=schedule, learning_rate=0.01,
    )
    source = Trainer(_TinyLogits(), config, device="cpu")
    assert source.train_microbatch(_BATCH).optimizer_stepped
    saved = source.state_dict()
    bad_optimizer = copy.deepcopy(saved.optimizer)
    bad_optimizer["param_groups"][0][option] = wrong
    target = Trainer(_TinyLogits(), config, device="cpu")
    with pytest.raises(TrainingStateInvalidError, match="default AdamW options differ"):
        target.load_state_dict(replace(saved, optimizer=bad_optimizer))
    assert not target.optimizer.state
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None and not target._update_incomplete
    target.model.load_state_dict(source.model.state_dict())
    target.load_state_dict(saved)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("option", "wrong"), [
        ("weight_decay", 0.5),
        ("eps", 0.1),
        ("betas", (0.5, 0.9)),
        ("amsgrad", True),
    ],
)
@pytest.mark.parametrize("schedule", ["constant", "cosine"])
def test_default_adamw_finite_live_option_forgery_poisoned(
    preserve_state: Any, option: str, wrong: Any, schedule: str,
) -> None:
    config = TrainerConfig(
        seed=703, max_steps=4, scheduler=schedule, learning_rate=0.01,
    )
    trainer = Trainer(_TinyLogits(), config, device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    trainer.optimizer.param_groups[0][option] = wrong
    with pytest.raises(TrainingStateInvalidError, match="default AdamW options differ"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert trainer.optimizer_step == 1
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
