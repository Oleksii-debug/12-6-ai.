from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

import twelve_six.checkpoint as checkpoint
from twelve_six.checkpoint import core
from twelve_six.checkpoint.hardening import (
    _preflight_optimizer_state,
    _validate_manifest_scalar_invariants,
    preflight_trainer_state,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


def test_package_import_installs_strict_numpy_dtype_preflight() -> None:
    target = np.zeros((2, 3), dtype=np.float32)
    corrupted = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(checkpoint.CheckpointCompatibilityError, match="dtype mismatch"):
        core._materialize_for_target(corrupted, target)


def test_strict_materialization_preserves_explicit_bfloat16_uint16_representation() -> None:
    target = torch.zeros((2, 3), dtype=torch.bfloat16)
    payload = target.view(torch.uint16).numpy().copy()

    restored = core._materialize_for_target(payload, target)

    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored, target)


def test_optimizer_preflight_rejects_wrong_shaped_sgd_momentum() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    state = optimizer.state_dict()
    parameter_id = state["param_groups"][0]["params"][0]
    state["state"][parameter_id] = {"momentum_buffer": torch.zeros(1)}

    with pytest.raises(
        checkpoint.CheckpointCompatibilityError,
        match="optimizer tensor shape mismatch",
    ):
        _preflight_optimizer_state(optimizer, state)


def test_manifest_preflight_rejects_negative_checkpoint_counters() -> None:
    manifest = {
        "identity": {
            "parameter_count": 1,
            "seed": 0,
            "precision": "fp32",
            "step": -1,
            "tokens_seen": 0,
            "model_spec": {"name": "probe"},
            "training_config": {"name": "probe"},
            "optimizer": {"name": "sgd"},
            "scheduler": None,
        }
    }

    with pytest.raises(checkpoint.CheckpointIntegrityError, match="identity.step"):
        _validate_manifest_scalar_invariants(manifest)


def test_trainer_preflight_rejects_corrupt_optimizer_before_restore() -> None:
    model = nn.Linear(4, 4)
    config = TrainerConfig(max_steps=2, learning_rate=1e-3)
    trainer = Trainer(model, config, device="cpu")
    state = asdict(trainer.state_dict())
    parameter_id = state["optimizer"]["param_groups"][0]["params"][0]
    state["optimizer"]["state"][parameter_id] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros(1),
        "exp_avg_sq": torch.zeros(1),
    }

    with pytest.raises(
        checkpoint.CheckpointCompatibilityError,
        match="optimizer tensor shape mismatch",
    ):
        preflight_trainer_state(trainer, state)
