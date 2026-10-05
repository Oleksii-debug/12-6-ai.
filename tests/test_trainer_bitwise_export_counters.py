"""Exact optimizer export bits and counter claims; tiny synthetic CPU tests only."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.training import Trainer, TrainerConfig, TrainingStateInvalidError


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
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state()
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
        torch.set_rng_state(torch_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(
            policy_before[0], warn_only=policy_before[1],
        )


@pytest.mark.parametrize(
    ("saved", "live"),
    [
        (torch.tensor([0.0]), torch.tensor([-0.0])),
        (torch.tensor(0.0), torch.tensor(-0.0)),
        (torch.tensor([0.0], dtype=torch.bfloat16),
         torch.tensor([-0.0], dtype=torch.bfloat16)),
        (torch.tensor([complex(0.0, 0.0)]), torch.tensor([complex(-0.0, 0.0)])),
        (np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32)),
        (np.array([1.0, 0.0, 2.0])[::-1],
         np.array([1.0, -0.0, 2.0])[::-1]),
        (np.float32(0.0), np.float32(-0.0)),
        (np.complex64(complex(0.0, 1.0)), np.complex64(complex(-0.0, 1.0))),
        (0.0, -0.0),
        (complex(0.0, 1.0), complex(-0.0, 1.0)),
    ],
    ids=[
        "torch-fp32", "torch-scalar", "torch-bfloat16", "torch-complex",
        "numpy-fp32", "numpy-strided", "numpy-scalar", "numpy-complex",
        "python-float", "python-complex",
    ],
)
def test_optimizer_export_rejects_numerically_equal_but_bitwise_different(
    saved: Any, live: Any,
) -> None:
    assert not Trainer._exact_export_leaf_equal(saved, live)
    assert Trainer._exact_export_leaf_equal(live, live)


@pytest.mark.parametrize(
    "leaf",
    [
        torch.tensor(0.0),
        torch.zeros((2, 3), dtype=torch.float32),
        np.float32(0.0),
        np.array([1.0, -0.0, 2.0])[::-1],
        0.0,
        -0.0,
        1 + 2j,
        {"state": [torch.tensor(1.0), np.array([], dtype=np.float32)]},
    ],
)
def test_exact_optimizer_export_still_accepts_identical_leaves(leaf: Any) -> None:
    assert Trainer._exact_export_leaf_equal(leaf, leaf)


def test_export_hook_cannot_forge_consistent_completed_steps_and_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyLogits()
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=3), device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    original_export = trainer.optimizer.state_dict
    real_weights = model.weight.detach().clone()
    real_moment = trainer.optimizer.state[model.weight]["exp_avg"].detach().clone()

    def forged_export() -> dict[str, Any]:
        saved = original_export()
        trainer.micro_step += 1
        trainer.optimizer_step += 1
        trainer.tokens_seen += 2
        return saved

    monkeypatch.setattr(trainer.optimizer, "state_dict", forged_export)
    with pytest.raises(TrainingStateInvalidError, match="changed committed counters"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    # No second AdamW step happened, and the forged exposure cannot be published.
    assert trainer.optimizer.state[model.weight]["step"].item() == 1
    torch.testing.assert_close(model.weight, real_weights, rtol=0, atol=0)
    torch.testing.assert_close(
        trainer.optimizer.state[model.weight]["exp_avg"], real_moment, rtol=0, atol=0,
    )
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


@pytest.mark.parametrize("target", ["parameter", "buffer"])
def test_finite_live_model_export_mutation_cannot_publish(
    monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    model = _TinyLogits()
    model.register_buffer("running", torch.tensor([0.0, 1.0]))
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=3), device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    saved_weight = model.weight.detach().clone()
    saved_buffer = model.running.detach().clone()
    saved_moment = trainer.optimizer.state[model.weight]["exp_avg"].detach().clone()
    original_export = trainer.optimizer.state_dict

    def mutate_after_serializer() -> dict[str, Any]:
        snapshot = original_export()
        with torch.no_grad():
            if target == "parameter":
                model.weight.add_(0.125)
            else:
                model.running.add_(0.125)
        return snapshot

    monkeypatch.setattr(trainer.optimizer, "state_dict", mutate_after_serializer)
    with pytest.raises(TrainingStateInvalidError, match="model weights or buffers"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    assert trainer.optimizer.state[model.weight]["step"].item() == 1
    torch.testing.assert_close(
        trainer.optimizer.state[model.weight]["exp_avg"], saved_moment, rtol=0, atol=0,
    )
    if target == "parameter":
        assert not torch.equal(model.weight, saved_weight)
        torch.testing.assert_close(model.running, saved_buffer, rtol=0, atol=0)
    else:
        torch.testing.assert_close(model.weight, saved_weight, rtol=0, atol=0)
        assert not torch.equal(model.running, saved_buffer)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_model_fingerprint_hashes_strided_buffers_and_signed_zero() -> None:
    model = _TinyLogits()
    model.register_buffer("view", torch.tensor([[0.0, 2.0], [3.0, 4.0]]).t())
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=2), device="cpu")
    assert not model.view.is_contiguous()
    original = trainer._model_export_fingerprint()
    assert original == trainer._model_export_fingerprint()
    with torch.no_grad():
        model.view[0, 0] = -0.0
    assert original != trainer._model_export_fingerprint()


@pytest.mark.parametrize("attack", ["moment", "learning-rate"])
def test_preflight_auxiliary_export_cannot_redefine_committed_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    model = _TinyLogits()
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=3), device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    before_moment = trainer.optimizer.state[model.weight]["exp_avg"].detach().clone()
    before_rate = trainer.optimizer.param_groups[0]["lr"]
    before_fingerprint = trainer._optimizer_live_fingerprint()
    assert before_fingerprint is not None
    native_scaler_export = trainer.scaler.state_dict
    calls: list[int] = []

    def mutate_optimizer_during_preflight() -> dict[str, Any]:
        calls.append(1)
        snapshot = native_scaler_export()
        if len(calls) == 1:
            if attack == "moment":
                trainer.optimizer.state[model.weight]["exp_avg"].add_(0.125)
            else:
                trainer.optimizer.param_groups[0]["lr"] *= 2.0
        return snapshot

    monkeypatch.setattr(trainer.scaler, "state_dict", mutate_optimizer_during_preflight)
    with pytest.raises(
        TrainingStateInvalidError,
        match="checkpoint preflight changed optimizer state",
    ):
        trainer.state_dict()
    assert calls
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    assert trainer._optimizer_live_fingerprint() != before_fingerprint
    if attack == "moment":
        assert not torch.equal(
            trainer.optimizer.state[model.weight]["exp_avg"], before_moment
        )
        assert trainer.optimizer.param_groups[0]["lr"] == before_rate
    else:
        torch.testing.assert_close(
            trainer.optimizer.state[model.weight]["exp_avg"],
            before_moment,
            rtol=0,
            atol=0,
        )
        assert trainer.optimizer.param_groups[0]["lr"] != before_rate
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


@pytest.mark.parametrize("attack", ["moment-bytes", "equal-byte-replacement"])
def test_optimizer_export_hook_cannot_redefine_committed_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    model = _TinyLogits()
    trainer = Trainer(model, TrainerConfig(seed=703, max_steps=3), device="cpu")
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    moment = trainer.optimizer.state[model.weight]["exp_avg"]
    before_moment = moment.detach().clone()
    before_ptr = moment.data_ptr()
    before_fingerprint = trainer._optimizer_live_fingerprint()
    assert before_fingerprint is not None
    native_export = trainer.optimizer.state_dict
    calls: list[int] = []

    def mutate_during_optimizer_export() -> dict[str, Any]:
        calls.append(1)
        slot = trainer.optimizer.state[model.weight]
        if attack == "moment-bytes":
            slot["exp_avg"].add_(0.125)
        else:
            slot["exp_avg"] = slot["exp_avg"].clone()
        return native_export()

    monkeypatch.setattr(trainer.optimizer, "state_dict", mutate_during_optimizer_export)
    with pytest.raises(
        TrainingStateInvalidError,
        match="checkpoint export changed optimizer state",
    ):
        trainer.state_dict()
    assert calls == [1]
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    assert trainer._optimizer_live_fingerprint() != before_fingerprint
    current = trainer.optimizer.state[model.weight]["exp_avg"]
    if attack == "moment-bytes":
        assert not torch.equal(current, before_moment)
    else:
        assert torch.equal(current, before_moment)
        assert current.data_ptr() != before_ptr
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_canonical_lambda_lr_positive_export_matches_live_state() -> None:
    model = _TinyLogits()
    trainer = Trainer(
        model, TrainerConfig(seed=703, max_steps=3, scheduler="cosine", warmup_steps=1),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    snapshot = trainer.state_dict()
    assert trainer.scheduler is not None and snapshot.scheduler is not None
    assert snapshot.scheduler["last_epoch"] == trainer.scheduler.last_epoch
    assert snapshot.scheduler["_last_lr"] == trainer.scheduler.get_last_lr()
    assert trainer._failure_reason is None


@pytest.mark.parametrize(
    "attack", ["detached-epoch", "detached-last-lr", "live-epoch"],
)
def test_finite_scheduler_snapshot_or_live_epoch_forgery_refused(
    monkeypatch: pytest.MonkeyPatch, attack: str,
) -> None:
    import copy

    model = _TinyLogits()
    trainer = Trainer(
        model, TrainerConfig(seed=703, max_steps=3, scheduler="cosine", warmup_steps=1),
        device="cpu",
    )
    assert trainer.train_microbatch(_BATCH).optimizer_stepped
    assert trainer.scheduler is not None
    scheduler = trainer.scheduler
    committed_epoch = scheduler.last_epoch
    weights = model.weight.detach().clone()
    original = scheduler.state_dict
    calls: list[int] = []

    def untrusted_scheduler_snapshot() -> dict[str, Any]:
        calls.append(1)
        saved = copy.deepcopy(original())
        if attack == "live-epoch" and len(calls) == 1:
            scheduler.last_epoch += 1
        elif attack == "detached-epoch":
            saved["last_epoch"] += 1
        elif attack == "detached-last-lr":
            saved["_last_lr"][0] *= 0.5
        return saved

    monkeypatch.setattr(scheduler, "state_dict", untrusted_scheduler_snapshot)
    with pytest.raises(TrainingStateInvalidError, match="scheduler"):
        trainer.state_dict()
    assert trainer._failure_reason is not None
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
    if attack != "live-epoch":
        assert scheduler.last_epoch == committed_epoch
    torch.testing.assert_close(model.weight, weights, rtol=0, atol=0)
    assert model.weight.grad is None
    assert len(calls) >= 1
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
