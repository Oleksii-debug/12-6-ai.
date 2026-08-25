from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from twelve_six.training import Trainer, TrainerConfig


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 12)
        self.proj = nn.Linear(12, 17, bias=False)
        self.proj.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(input_ids))


class _RedundantNormReferenceTrainer(Trainer):
    """Reinsert the removed reduction without changing the optimized update math."""

    legacy_grad_norm: float | None = None

    def _normalize_gradients_for_clipping(self, token_count: int) -> None:
        super()._normalize_gradients_for_clipping(token_count)
        squared_norm = torch.zeros((), device=self.device)
        found = False
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            found = True
            grad = parameter.grad.detach()
            squared_norm += torch.sum(grad.float() * grad.float())
        self.legacy_grad_norm = float(torch.sqrt(squared_norm).item()) if found else 0.0


def _config(*, clip: float | None = 1.0) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=1,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=clip,
        precision="fp32",
        seed=148,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _assert_tensor_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_tensor_tree_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tensor_tree_equal(left_item, right_item)
        return
    assert left == right


def test_clipped_update_matches_redundant_norm_reference_bitwise() -> None:
    torch.manual_seed(148)
    reference_model = _TinyLM()
    optimized_model = copy.deepcopy(reference_model)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
    }

    reference = _RedundantNormReferenceTrainer(reference_model, _config())
    optimized = Trainer(optimized_model, _config())
    reference_metrics = reference.train_microbatch(batch)
    optimized_metrics = optimized.train_microbatch(batch)

    assert reference_metrics == optimized_metrics
    assert reference.legacy_grad_norm is not None
    assert optimized_metrics.grad_norm is not None
    assert optimized_metrics.grad_norm == pytest.approx(reference.legacy_grad_norm, rel=1e-6, abs=1e-7)

    for reference_parameter, optimized_parameter in zip(
        reference_model.parameters(), optimized_model.parameters(), strict=True
    ):
        assert torch.equal(reference_parameter, optimized_parameter)
    _assert_tensor_tree_equal(reference.optimizer.state_dict(), optimized.optimizer.state_dict())


def test_unclipped_path_retains_original_norm_contract() -> None:
    torch.manual_seed(149)
    model = _TinyLM()
    trainer = Trainer(model, _config(clip=None))
    metrics = trainer.train_microbatch(
        {
            "input_ids": torch.tensor([[2, 4, 6, 8]], dtype=torch.long),
            "labels": torch.tensor([[2, 4, 6, 8]], dtype=torch.long),
        }
    )
    assert metrics.optimizer_stepped is True
    assert metrics.grad_norm is not None
    assert metrics.grad_norm > 0.0
