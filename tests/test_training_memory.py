from __future__ import annotations

from pathlib import Path

import torch

from twelve_six.distributed import ModelScaleSpec, ParallelPlan, estimate_training_memory
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training import (
    Trainer,
    TrainerConfig,
    measure_training_tensor_memory,
    parameter_tensor_bytes,
    scaler_state_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_100k_live_trainer_state_matches_measured_per_parameter_formula() -> None:
    stage = load_stage_config(REPO_ROOT / "configs/stages/s1_100k.json")
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, TrainerConfig(max_steps=1), device="cpu")
    parameter_count = stage.expected_parameters

    initialized = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)
    assert initialized.parameter_bytes == 4 * parameter_count
    assert initialized.gradient_bytes == 0
    assert initialized.adam_moment_bytes == 0
    assert initialized.optimizer_other_tensor_bytes == 0
    assert scaler_state_metadata(trainer.scaler) == {
        "present": True,
        "enabled": False,
        "state_keys": [],
        "tensor_bytes": 0,
    }

    captured = {}
    original_step = trainer.optimizer.step

    def measured_step(*args, **kwargs):
        captured["pre"] = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)
        result = original_step(*args, **kwargs)
        captured["post"] = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)
        return result

    trainer.optimizer.step = measured_step  # type: ignore[method-assign]
    input_ids = torch.arange(8, dtype=torch.long).view(1, -1) % stage.model.vocab_size
    trainer.train_microbatch({"input_ids": input_ids, "labels": input_ids.clone()})

    assert captured["pre"].gradient_bytes == 4 * parameter_count
    assert captured["post"].adam_moment_bytes == 8 * parameter_count
    assert captured["post"].optimizer_other_tensor_bytes > 0
    final = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)
    assert final.gradient_bytes == 0
    assert final.adam_moment_bytes == 8 * parameter_count

    planning_model = ModelScaleSpec(
        total_parameters=parameter_count,
        hidden_size=stage.model.d_model,
        num_layers=stage.model.n_layers,
        num_attention_heads=stage.model.n_heads,
        sequence_length=8,
        micro_batch_size=1,
    )
    estimate = estimate_training_memory(planning_model, ParallelPlan())
    assert estimate.parameter_bytes_per_rank == captured["pre"].parameter_bytes
    assert estimate.gradient_bytes_per_rank == captured["pre"].gradient_bytes
    assert estimate.optimizer_bytes_per_rank == captured["post"].adam_moment_bytes
    assert estimate.master_weight_bytes_per_rank == 0


def test_tied_embedding_is_not_double_counted() -> None:
    stage = load_stage_config(REPO_ROOT / "configs/stages/s1_100k.json")
    model = TwelveSixDecoder(stage.model, stage.init)
    assert model.lm_head.weight is model.token_embedding.weight
    assert parameter_tensor_bytes(model) == 4 * stage.expected_parameters
