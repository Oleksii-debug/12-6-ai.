from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from twelve_six.training import (
    NonFiniteTrainingError,
    Trainer,
    TrainerConfig,
    causal_lm_loss,
    causal_pair_loss,
)


class ToyBigramLM(nn.Module):
    """Test-only learnable stub; it is not the canonical D01 architecture."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table(input_ids)


def repeating_batch() -> dict[str, torch.Tensor]:
    tokens = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3],
            [1, 2, 3, 0, 1, 2, 3, 0],
        ],
        dtype=torch.long,
    )
    return {"input_ids": tokens}


def test_causal_loss_uses_next_token_shift() -> None:
    logits = torch.full((1, 3, 4), -10.0)
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0
    labels = torch.tensor([[1, 2, 3]])
    assert causal_lm_loss(logits, labels).item() < 1e-6


def test_causal_pair_loss_uses_aligned_targets_and_masks_padding() -> None:
    logits = torch.full((1, 3, 4), -10.0)
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0
    logits[0, 2, 0] = 10.0
    target_ids = torch.tensor([[2, 3, 1]])
    loss_mask = torch.tensor([[1, 1, 0]])
    assert causal_pair_loss(logits, target_ids, loss_mask=loss_mask).item() < 1e-6


def test_gradient_accumulation_steps_only_on_boundary() -> None:
    model = ToyBigramLM(vocab_size=4)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.1,
            max_steps=4,
            gradient_accumulation_steps=2,
            gradient_clip_norm=1.0,
        ),
    )

    first = trainer.train_microbatch(repeating_batch())
    second = trainer.train_microbatch(repeating_batch())

    assert first.optimizer_stepped is False
    assert first.optimizer_step == 0
    assert second.optimizer_stepped is True
    assert second.optimizer_step == 1
    assert second.grad_norm is not None and math.isfinite(second.grad_norm)
    assert trainer.tokens_seen == 28


def test_optimizer_step_changes_weights_with_finite_metrics() -> None:
    torch.manual_seed(5)
    model = ToyBigramLM(vocab_size=4)
    before = model.table.weight.detach().clone()
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.1,
            max_steps=1,
            gradient_clip_norm=1.0,
            seed=5,
        ),
    )

    metrics = trainer.train_microbatch(repeating_batch())
    after = model.table.weight.detach()

    assert metrics.optimizer_stepped is True
    assert metrics.optimizer_step == 1
    assert math.isfinite(metrics.loss)
    assert metrics.grad_norm is not None and math.isfinite(metrics.grad_norm)
    assert not torch.equal(before, after)


def test_loss_decreases_on_deterministic_tiny_corpus() -> None:
    torch.manual_seed(7)
    model = ToyBigramLM(vocab_size=4)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.15,
            weight_decay=0.0,
            max_steps=40,
            scheduler="constant",
            gradient_clip_norm=1.0,
            seed=7,
        ),
    )

    losses = [trainer.train_microbatch(repeating_batch()).loss for _ in range(40)]
    assert losses[-1] < losses[0] * 0.2
    assert trainer.optimizer_step == 40
    assert trainer.tokens_seen == 40 * 14


def test_trainer_state_resume_preserves_optimizer_scheduler_and_counters() -> None:
    torch.manual_seed(11)
    config = TrainerConfig(
        learning_rate=0.1,
        max_steps=10,
        warmup_steps=1,
        scheduler="cosine",
        seed=11,
    )
    source_model = ToyBigramLM(vocab_size=4)
    source = Trainer(source_model, config)
    for _ in range(3):
        source.train_microbatch(repeating_batch())

    target_model = ToyBigramLM(vocab_size=4)
    target_model.load_state_dict(source_model.state_dict())
    target = Trainer(target_model, config)
    target.load_state_dict(source.state_dict())

    assert target.micro_step == 3
    assert target.optimizer_step == 3
    assert target.tokens_seen == 42
    assert target.state_dict().optimizer["state"]
    assert target.state_dict().scheduler is not None

    source.train_microbatch(repeating_batch())
    target.train_microbatch(repeating_batch())
    for source_parameter, target_parameter in zip(
        source_model.parameters(), target_model.parameters(), strict=True
    ):
        torch.testing.assert_close(source_parameter, target_parameter)


def test_trainer_consumes_d04_aligned_target_contract_without_double_shift() -> None:
    torch.manual_seed(13)
    trainer = Trainer(
        ToyBigramLM(vocab_size=4),
        TrainerConfig(learning_rate=0.1, max_steps=1, seed=13),
    )
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
        "target_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        "loss_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
    }

    metrics = trainer.train_microbatch(batch)

    assert metrics.optimizer_stepped is True
    assert metrics.tokens == 3
    assert math.isfinite(metrics.loss)


def test_nonfinite_loss_fails_before_optimizer_step() -> None:
    class NaNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            batch, time = input_ids.shape
            logits = torch.zeros(batch, time, 4) * self.scale
            return logits / torch.tensor(0.0)

    trainer = Trainer(NaNModel(), TrainerConfig(max_steps=1))
    with pytest.raises(NonFiniteTrainingError):
        trainer.train_microbatch(repeating_batch())
    assert trainer.optimizer_step == 0


def test_nonfinite_gradient_fails_before_optimizer_step() -> None:
    model = ToyBigramLM(vocab_size=4)
    model.table.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    trainer = Trainer(model, TrainerConfig(max_steps=1))
    with pytest.raises(NonFiniteTrainingError, match="gradient"):
        trainer.train_microbatch(repeating_batch())
    assert trainer.optimizer_step == 0


def test_refuses_partial_accumulation_boundary() -> None:
    trainer = Trainer(
        ToyBigramLM(vocab_size=4),
        TrainerConfig(max_steps=4, gradient_accumulation_steps=2),
    )
    trainer.train_microbatch(repeating_batch())
    with pytest.raises(RuntimeError, match="mid-accumulation"):
        trainer.assert_accumulation_boundary()
