from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from twelve_six.training import (
    NonFiniteTrainingError,
    Trainer,
    TrainerConfig,
    TrainingStateInvalidError,
)


class SmallDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 4)
        self.head = nn.Linear(4, 4)
        self.last_logits_dtype = None

    def forward(self, input_ids):
        logits = self.head(self.embedding(input_ids))
        self.last_logits_dtype = logits.dtype
        return logits


def batch():
    return {"input_ids": torch.tensor([[0, 1, 2]])}


def test_cosine_warmup_and_accumulation_replay_do_not_pull_extra_input() -> None:
    config = TrainerConfig(
        max_steps=4, gradient_accumulation_steps=2, scheduler="cosine",
        warmup_steps=2, learning_rate=0.01,
    )
    original = SmallDecoder()
    first = Trainer(copy.deepcopy(original), config)
    second = Trainer(copy.deepcopy(original), config)
    sentinel = object()
    stream = iter([batch()] * 8 + [sentinel])
    metrics = []
    first.run(stream, on_metrics=metrics.append)
    second.run(iter([batch()] * 8))
    assert next(stream) is sentinel
    assert [item.optimizer_step for item in metrics] == [0, 1, 1, 2, 2, 3, 3, 4]
    assert [item.learning_rate for item in metrics if item.optimizer_stepped] == pytest.approx(
        [0.005, 0.01, 0.01, 0.005]
    )
    for left, right in zip(first.model.parameters(), second.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_finite_elements_with_overflowing_norm_do_not_commit() -> None:
    model = SmallDecoder()
    model.head.weight.register_hook(lambda grad: torch.full_like(grad, 1e30))
    trainer = Trainer(model, TrainerConfig(max_steps=1, gradient_clip_norm=None))
    before = model.head.weight.detach().clone()
    with pytest.raises(NonFiniteTrainingError, match="gradient norm"):
        trainer.train_microbatch(batch())
    torch.testing.assert_close(model.head.weight, before, atol=0, rtol=0)
    assert trainer.optimizer_step == 0
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.assert_checkpoint_safe()


def test_nonfinite_optimizer_result_requires_fresh_verified_state() -> None:
    class CorruptingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            super().step(closure)
            self.state[self.param_groups[0]["params"][0]]["exp_avg_sq"].fill_(float("inf"))

    model = SmallDecoder()
    config = TrainerConfig(max_steps=1)
    clean_weights = copy.deepcopy(model.state_dict())
    initial = Trainer(model, config, optimizer=CorruptingAdamW(model.parameters()))
    verified_state = initial.state_dict()
    with pytest.raises(NonFiniteTrainingError, match="optimizer state"):
        initial.train_microbatch(batch())
    assert initial.optimizer_step == 0
    with pytest.raises(TrainingStateInvalidError, match="verified"):
        initial.load_state_dict(verified_state)
    replacement = SmallDecoder()
    replacement.load_state_dict(clean_weights)
    restored = Trainer(replacement, config)
    restored.load_state_dict(verified_state)
    assert restored.train_microbatch(batch()).optimizer_step == 1
    restored.assert_checkpoint_safe()


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [
        ("cpu", "bf16", torch.bfloat16),
        pytest.param(
            "cuda", "fp16", torch.float16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_supported_autocast_commits_one_finite_step(device, precision, dtype) -> None:
    model = SmallDecoder()
    trainer = Trainer(model, TrainerConfig(max_steps=1, precision=precision), device=device)
    metrics = trainer.train_microbatch(batch())
    assert model.last_logits_dtype == dtype
    assert metrics.optimizer_stepped
    assert metrics.optimizer_step == 1
    assert trainer.scaler.is_enabled() is (precision == "fp16")
    trainer.assert_checkpoint_safe()
