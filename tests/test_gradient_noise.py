from __future__ import annotations

import copy
import random
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from twelve_six.gradient_noise import (
    estimate_gradient_stochasticity,
    gradient_state_fingerprint,
    load_experiment_config,
    model_state_fingerprint,
    optimizer_state_fingerprint,
    state_fingerprint,
)
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.scaling_experiment import controlled_specs
from twelve_six.training import Trainer, TrainerConfig


def _trainer() -> Trainer:
    torch.manual_seed(7)
    model = TwelveSixDecoder(controlled_specs()[0], InitSpec())
    config = TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=4,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=7,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    return Trainer(model, config, device="cpu")


def _batch(offset: int = 0) -> dict[str, torch.Tensor]:
    values = (torch.arange(4 * 32, dtype=torch.long) + offset).remainder(256)
    return {"input_ids": values.view(4, 32)}


def test_controlled_family_matches_requested_scale_bands() -> None:
    assert [spec.parameter_count() for spec in controlled_specs()] == [
        95_568,
        267_912,
        467_808,
        1_037_696,
    ]


def test_probe_is_exactly_non_mutating_after_real_optimizer_state_exists() -> None:
    trainer = _trainer()
    trainer.train_microbatch(_batch())
    for parameter in trainer.model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)

    random.seed(101)
    torch.manual_seed(101)
    before = {
        "model": model_state_fingerprint(trainer.model),
        "optimizer": optimizer_state_fingerprint(trainer),
        "trainer": state_fingerprint(asdict(trainer.state_dict())),
        "grads": gradient_state_fingerprint(trainer.model),
        "python_rng": copy.deepcopy(random.getstate()),
        "torch_rng": torch.get_rng_state().clone(),
        "mode": trainer.model.training,
        "counters": (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen),
    }

    result = estimate_gradient_stochasticity(
        trainer,
        [_batch(11), _batch(29), _batch(47), _batch(83)],
        clip_norm=1.0,
        virtual_batch_multipliers=(1, 2, 4),
    )

    assert result["non_mutation"]["verified"] is True
    assert model_state_fingerprint(trainer.model) == before["model"]
    assert optimizer_state_fingerprint(trainer) == before["optimizer"]
    assert state_fingerprint(asdict(trainer.state_dict())) == before["trainer"]
    assert gradient_state_fingerprint(trainer.model) == before["grads"]
    assert random.getstate() == before["python_rng"]
    assert torch.equal(torch.get_rng_state(), before["torch_rng"])
    assert trainer.model.training is before["mode"]
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == before["counters"]


def test_identical_repeated_microbatches_have_negligible_variance_proxy() -> None:
    trainer = _trainer()
    batch = _batch(19)
    result = estimate_gradient_stochasticity(
        trainer,
        [batch, batch, batch, batch],
        clip_norm=1.0,
        virtual_batch_multipliers=(1, 2, 4),
    )
    assert result["global"]["noise_to_signal_trace_ratio"] < 1e-5
    assert len(result["per_block"]) == controlled_specs()[0].n_layers
    for block in result["per_block"]:
        assert block["noise_to_signal_trace_ratio"] < 1e-5


def test_virtual_batch_proxy_records_expected_group_counts() -> None:
    trainer = _trainer()
    batches = [_batch(offset) for offset in range(8)]
    result = estimate_gradient_stochasticity(
        trainer,
        batches,
        clip_norm=1.0,
        virtual_batch_multipliers=(1, 2, 4),
    )
    assert [row["independent_groups_observed"] for row in result["virtual_batch_noise_proxy"]] == [
        8,
        4,
        2,
    ]
    assert result["noise_proxy"]["universal_gradient_noise_scale_claimed"] is False


def test_experiment_config_is_checkpointed_and_local_free() -> None:
    path = Path("configs/experiments/gradient_stochasticity_matrix_v1.json")
    config = load_experiment_config(path)
    assert config["training"]["checkpoints"] == [0, 4, 16, 48]
    assert config["runtime"]["device"] == "cpu"
    assert config["runtime"]["paid_compute"] is False
    assert config["probe"]["repeat_microbatches"] == 8


def test_invalid_virtual_batch_partition_fails_closed() -> None:
    trainer = _trainer()
    with pytest.raises(ValueError, match="divide repeat count"):
        estimate_gradient_stochasticity(
            trainer,
            [_batch(1), _batch(2), _batch(3), _batch(4)],
            clip_norm=1.0,
            virtual_batch_multipliers=(1, 3),
        )
