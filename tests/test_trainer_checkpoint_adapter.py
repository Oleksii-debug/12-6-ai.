from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twelve_six.checkpoint import (
    CheckpointIdentity,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)


class Model:
    def __init__(self):
        self.weights = np.array([0.25, -0.5], dtype=np.float64)

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        assert not strict or set(state) == {"weights"}
        self.weights = state["weights"].copy()


@dataclass
class TrainerState:
    micro_step: int
    optimizer_step: int
    tokens_seen: int
    optimizer: dict[str, object]
    scheduler: dict[str, object] | None
    scaler: dict[str, object] | None
    config: dict[str, object]


class Trainer:
    def __init__(self):
        self.micro_step = 4
        self.optimizer_step = 2
        self.tokens_seen = 64
        self.velocity = np.array([1.0, 2.0], dtype=np.float64)
        self.scheduler_step = 2
        self.config = {"seed": 7, "precision": "fp32"}

    def state_dict(self):
        return TrainerState(
            micro_step=self.micro_step,
            optimizer_step=self.optimizer_step,
            tokens_seen=self.tokens_seen,
            optimizer={"state": {0: {"momentum": self.velocity.copy()}}},
            scheduler={"last_epoch": self.scheduler_step},
            scaler={},
            config=dict(self.config),
        )

    def load_state_dict(self, state):
        assert state["config"] == self.config
        self.micro_step = state["micro_step"]
        self.optimizer_step = state["optimizer_step"]
        self.tokens_seen = state["tokens_seen"]
        self.velocity = state["optimizer"]["state"][0]["momentum"].copy()
        self.scheduler_step = state["scheduler"]["last_epoch"]


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="c" * 40,
        model_spec={"kind": "trainer-adapter-test"},
        parameter_count=2,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"seed": 7, "precision": "fp32"},
        seed=7,
        precision="fp32",
        step=2,
        tokens_seen=64,
        optimizer={"name": "trainer-owned"},
        scheduler={"name": "trainer-owned"},
        environment_lock_hash="5" * 64,
    )


def test_dataclass_trainer_state_roundtrip(tmp_path: Path):
    model = Model()
    trainer = Trainer()
    expected_weights = model.weights.copy()
    expected_velocity = trainer.velocity.copy()

    checkpoint = tmp_path / "trainer-checkpoint"
    save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity(),
    )

    model.weights[:] = 99.0
    trainer.micro_step = 999
    trainer.optimizer_step = 999
    trainer.tokens_seen = 999
    trainer.velocity[:] = -99.0
    trainer.scheduler_step = 999

    result = load_trainer_checkpoint(checkpoint, model=model, trainer=trainer)

    np.testing.assert_array_equal(model.weights, expected_weights)
    np.testing.assert_array_equal(trainer.velocity, expected_velocity)
    assert trainer.micro_step == 4
    assert trainer.optimizer_step == 2
    assert trainer.tokens_seen == 64
    assert trainer.scheduler_step == 2
    assert result.trainer_state["config"] == trainer.config
