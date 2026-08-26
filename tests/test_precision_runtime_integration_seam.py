from __future__ import annotations

import copy

import torch
from torch import nn

from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.observability import TrainingObserver


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _config(*, max_steps: int = 2) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=163,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
    }


def test_fp32_precision_seam_does_not_change_model_math_at_construction() -> None:
    torch.manual_seed(163)
    model = TinyLM()
    inputs = _batch()["input_ids"]
    state_before = copy.deepcopy(model.state_dict())
    logits_before = model(inputs).detach().clone()

    trainer = Trainer(model, _config(), device="cpu")

    assert trainer.precision_runtime.to_dict() == {
        "requested": "fp32",
        "device_type": "cpu",
        "parameter_dtype": "float32",
        "optimizer_master_dtype": "float32",
        "autocast_enabled": False,
        "autocast_dtype": None,
        "grad_scaler_enabled": False,
        "grad_scaler_device": None,
    }
    for name, tensor in state_before.items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), logits_before, rtol=0, atol=0)


def test_precision_runtime_is_reconstructed_around_legacy_trainer_state() -> None:
    torch.manual_seed(163)
    model = TinyLM()
    config = _config()
    trainer = Trainer(model, config, device="cpu")
    first = trainer.train_microbatch(_batch())
    assert first.optimizer_step == 1
    state = trainer.state_dict()
    assert not hasattr(state, "precision_runtime")

    restored_model = TinyLM()
    restored_model.load_state_dict(copy.deepcopy(model.state_dict()))
    restored = Trainer(restored_model, config, device="cpu")
    restored.load_state_dict(state)

    assert restored.optimizer_step == trainer.optimizer_step
    assert restored.micro_step == trainer.micro_step
    assert restored.tokens_seen == trainer.tokens_seen
    assert restored.precision_runtime.to_dict() == trainer.precision_runtime.to_dict()
    second = restored.train_microbatch(_batch())
    assert second.optimizer_step == 2


def test_training_observer_remains_compatible_with_precision_runtime_trainer() -> None:
    torch.manual_seed(163)
    trainer = Trainer(TinyLM(), _config(max_steps=1), device="cpu")
    observer = TrainingObserver(
        {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": "a" * 40,
            "stage": "INTEGRATE-163",
            "modelspec_sha256": "b" * 64,
            "training": {"seed": 163, "max_steps": 1, "precision": "fp32"},
        },
        device="cpu",
        max_step_samples=4,
    )

    measured = observer.train_microbatch(trainer, _batch(), data_wait_seconds=0.0)
    summary = observer.summary()

    assert measured.optimizer_step == 1
    assert trainer.precision_runtime.requested == "fp32"
    assert summary["counters"]["optimized_tokens"] == trainer.tokens_seen
    assert summary["counters"]["observed_microbatches"] == 1
