from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from twelve_six.training.batch_noise import diagnose_gradient_noise, gradient_statistics
from twelve_six.training.batch_noise_probe import BASE_LOSS_TOKENS, PARAMETER_COUNT, fixed_268k_model_spec
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


def test_gradient_statistics_zero_variance_for_identical_samples() -> None:
    gradient = torch.tensor([1.0, -2.0, 3.0])
    result = gradient_statistics([gradient.clone() for _ in range(8)], effective_microbatch_counts=(1, 2, 4))

    assert result["signal_squared"] == pytest.approx(14.0)
    assert result["trace_covariance_unbiased"] == pytest.approx(0.0)
    assert result["noise_scale_microbatches_proxy"] == pytest.approx(0.0)
    assert result["exact_critical_batch_size_claim"] is False
    for point in result["effective_batch_proxies"]:
        assert point["mean_relative_deviation_from_all_sample_mean"] == pytest.approx(0.0)
        assert point["mean_cosine_to_all_sample_mean"] == pytest.approx(1.0)


def test_accumulated_gradient_groups_reduce_known_alternating_noise() -> None:
    gradients = [
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 2.0]),
    ]
    result = gradient_statistics(gradients, effective_microbatch_counts=(1, 2))

    assert result["signal_squared"] == pytest.approx(2.0)
    assert result["trace_covariance_unbiased"] == pytest.approx(8.0 / 3.0)
    by_size = {point["effective_microbatches"]: point for point in result["effective_batch_proxies"]}
    assert by_size[2]["empirical_group_trace_covariance"] == pytest.approx(0.0)
    assert by_size[2]["mean_relative_deviation_from_all_sample_mean"] == pytest.approx(0.0)
    assert by_size[2]["mean_cosine_to_all_sample_mean"] == pytest.approx(1.0)


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 6)
        self.projection = nn.Linear(6, 8, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def _config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=2,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=41,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _batches() -> list[dict[str, torch.Tensor]]:
    return [
        {"input_ids": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)},
        {"input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)},
        {"input_ids": torch.tensor([[2, 3, 4, 5], [6, 7, 0, 1]], dtype=torch.long)},
        {"input_ids": torch.tensor([[3, 4, 5, 6], [7, 0, 1, 2]], dtype=torch.long)},
    ]


def test_diagnostic_probe_preserves_optimizer_model_grad_and_rng_state() -> None:
    torch.manual_seed(99)
    model = TinyLM()
    trainer = Trainer(model, _config(), device="cpu")
    batches = _batches()
    trainer.train_microbatch(batches[0])

    result = diagnose_gradient_noise(
        model,
        trainer,
        batches,
        effective_microbatch_counts=(1, 2, 4),
    )

    preservation = result["state_preservation"]
    assert preservation["model_state_unchanged"] is True
    assert preservation["optimizer_scheduler_counters_unchanged"] is True
    assert preservation["parameter_gradients_restored"] is True
    assert preservation["python_torch_cuda_rng_restored"] is True
    assert preservation["model_train_eval_mode_restored"] is True
    assert result["statistics"]["sample_count"] == 4
    assert result["statistics"]["trace_covariance_unbiased"] >= 0.0
    assert result["memory_overhead"]["full_model_duplicate_retained"] is False


def test_diagnostic_probe_does_not_change_next_optimizer_update() -> None:
    batches = _batches()
    torch.manual_seed(123)
    left_model = TinyLM()
    torch.manual_seed(123)
    right_model = TinyLM()
    left = Trainer(left_model, _config(), device="cpu")
    right = Trainer(right_model, _config(), device="cpu")

    left.train_microbatch(batches[0])
    right.train_microbatch(batches[0])
    diagnose_gradient_noise(left_model, left, batches, effective_microbatch_counts=(1, 2))
    left_metrics = left.train_microbatch(batches[1])
    right_metrics = right.train_microbatch(batches[1])

    assert left_metrics.loss == pytest.approx(right_metrics.loss, abs=0.0)
    assert left_metrics.grad_norm == pytest.approx(right_metrics.grad_norm, abs=0.0)
    for left_parameter, right_parameter in zip(left_model.parameters(), right_model.parameters(), strict=True):
        assert torch.equal(left_parameter, right_parameter)


def test_fixed_268k_geometry_and_base_batch_contract() -> None:
    spec = fixed_268k_model_spec()

    assert spec.parameter_count() == PARAMETER_COUNT == 267_912
    assert spec.d_model == 72
    assert spec.n_layers == 4
    assert spec.n_heads == 6
    assert spec.d_ff == 192
    assert spec.vocab_size == 256
    assert BASE_LOSS_TOKENS == 252
