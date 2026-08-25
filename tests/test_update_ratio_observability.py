from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from twelve_six.training import Trainer, TrainerConfig, TrainingObserver
from twelve_six.training.observability import summarize_parameter_update


def _config(*, max_steps: int) -> TrainerConfig:
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
        seed=1337,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _identity(max_steps: int) -> dict[str, object]:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "source_sha": "5" * 40,
        "stage": "TRAIN-52-TEST",
        "training": {"seed": 1337, "max_steps": max_steps},
    }


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(16, 8)
        self.blocks = nn.ModuleList(
            [nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)]
        )
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = torch.tanh(block(x))
        return {"logits": self.lm_head(x)}


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]],
            dtype=torch.long,
        )
    }


def test_exact_known_global_and_per_block_update_magnitude() -> None:
    before = {
        "blocks.0.weight": torch.tensor([3.0, 4.0]),
        "blocks.1.weight": torch.tensor([0.0, 12.0]),
    }
    after = {
        "blocks.0.weight": torch.tensor([3.3, 4.4]),
        "blocks.1.weight": torch.tensor([0.0, 13.2]),
    }

    global_metrics, per_block = summarize_parameter_update(before, after)

    assert global_metrics.parameter_norm == pytest.approx(13.0)
    assert global_metrics.update_norm == pytest.approx(1.3)
    assert global_metrics.update_weight_ratio == pytest.approx(0.1)
    assert global_metrics.max_update_magnitude == pytest.approx(1.2)
    assert global_metrics.parameter_elements == 4
    assert per_block["blocks.0"].parameter_norm == pytest.approx(5.0)
    assert per_block["blocks.0"].update_norm == pytest.approx(0.5)
    assert per_block["blocks.0"].update_weight_ratio == pytest.approx(0.1)
    assert per_block["blocks.1"].parameter_norm == pytest.approx(12.0)
    assert per_block["blocks.1"].update_norm == pytest.approx(1.2)
    assert per_block["blocks.1"].update_weight_ratio == pytest.approx(0.1)


def test_update_probe_does_not_change_deterministic_training_or_fingerprint() -> None:
    torch.manual_seed(1337)
    original = TinyLM()
    baseline_model = copy.deepcopy(original)
    observed_model = copy.deepcopy(original)
    config = _config(max_steps=3)
    baseline = Trainer(baseline_model, config, device="cpu")
    observed = Trainer(observed_model, config, device="cpu")
    observer = TrainingObserver(
        _identity(3),
        max_step_samples=8,
        max_update_samples=8,
        gpu_sample_every_steps=100,
    )
    identity_hash_before = observer.run_identity_sha256

    for _ in range(3):
        baseline_metrics = baseline.train_microbatch(_batch())
        observed_metrics = observer.train_microbatch(
            observed,
            _batch(),
            data_wait_seconds=0.0,
        )
        assert observed_metrics == baseline_metrics
        assert observer._pending_update_snapshot is None

    for baseline_parameter, observed_parameter in zip(
        baseline_model.parameters(), observed_model.parameters(), strict=True
    ):
        assert torch.equal(baseline_parameter, observed_parameter)
    assert observer.run_identity_sha256 == identity_hash_before
    assert len(observer.update_samples) == 3
    assert observer.update_samples[-1].global_metrics.update_norm > 0.0
    assert set(observer.update_samples[-1].per_block) == {"blocks.0", "blocks.1"}
    summary = observer.summary()["update_magnitude"]
    assert summary["semantics"]["telemetry_is_training_or_checkpoint_state"] is False
    assert summary["overhead"]["temporary_snapshot_peak_bytes"] == summary["overhead"][
        "model_trainable_parameter_storage_bytes"
    ]
    assert "update_magnitude" not in baseline.state_dict().config
    assert "update_magnitude" not in observed.state_dict().config


def test_tied_parameters_are_snapshotted_once() -> None:
    class TiedLM(TinyLM):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head.weight = self.token_embedding.weight

    torch.manual_seed(7)
    model = TiedLM()
    trainer = Trainer(model, _config(max_steps=1), device="cpu")
    observer = TrainingObserver(_identity(1), max_update_samples=4)
    observer.train_microbatch(trainer, _batch(), data_wait_seconds=0.0)

    unique_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    update_summary = observer.summary()["update_magnitude"]
    assert update_summary["overhead"]["model_trainable_parameter_storage_bytes"] == unique_bytes
    assert observer.update_samples[0].global_metrics.parameter_tensors == len(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )


def test_update_telemetry_retention_and_future_probe_cadence_are_bounded() -> None:
    torch.manual_seed(9)
    model = TinyLM()
    trainer = Trainer(model, _config(max_steps=24), device="cpu")
    observer = TrainingObserver(
        _identity(24),
        max_step_samples=4,
        max_update_samples=4,
        update_sample_every_steps=1,
    )
    for _ in range(24):
        observer.train_microbatch(trainer, _batch(), data_wait_seconds=0.0)

    update_summary = observer.summary()["update_magnitude"]
    assert len(observer.update_samples) <= 4
    assert update_summary["retention_stride"] > 1
    assert update_summary["effective_future_sample_every_optimizer_steps"] > 1
    assert update_summary["probe_samples_total"] < trainer.optimizer_step
    assert update_summary["pathology_candidates"]["record_limit"] == 32


def test_manual_record_step_remains_valid_without_attaching_a_model() -> None:
    from twelve_six.training.trainer import StepMetrics

    observer = TrainingObserver(_identity(1))
    observer.record_step(
        StepMetrics(
            micro_step=1,
            optimizer_step=1,
            loss=1.0,
            update_loss=1.0,
            learning_rate=3e-4,
            grad_norm=0.5,
            tokens=10,
            optimizer_stepped=True,
        ),
        data_wait_seconds=0.0,
        step_seconds=0.1,
    )
    assert observer.summary()["update_magnitude"]["status"] == "ENABLED_NOT_YET_SAMPLED"
