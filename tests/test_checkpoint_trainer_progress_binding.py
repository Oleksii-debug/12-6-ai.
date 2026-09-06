from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)


class NumpyModel:
    def __init__(self, values: list[float]) -> None:
        self.weights = np.asarray(values, dtype=np.float64).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


class GenericTrainer:
    def __init__(self) -> None:
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 10}
        self.loads = 0
        self.state = {
            "micro_step": 7,
            "optimizer_step": 7,
            "tokens_seen": 128,
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "config": dict(self.config),
        }

    def state_dict(self) -> dict[str, object]:
        return dict(self.state)

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.loads += 1
        self.state = dict(state)


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-trainer-progress", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 10},
        seed=7,
        precision="float64",
        step=7,
        tokens_seen=128,
        optimizer={"name": "trainer-owned"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("expected_step", 8), ("expected_tokens_seen", 127)],
)
def test_trainer_wrong_positive_progress_rejected_before_mutation(
    tmp_path: Path, field: str, value: int
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = GenericTrainer()
    before = model.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="progress mismatch"):
        load_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=trainer,
            restore_rng=False,
            **{field: value},
        )

    np.testing.assert_array_equal(model.weights, before)
    assert model.loads == 0
    assert trainer.loads == 0


def test_trainer_exact_positive_progress_restores_once(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = GenericTrainer()

    result = load_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        restore_rng=False,
        expected_step=7,
        expected_tokens_seen=128,
    )

    np.testing.assert_array_equal(model.weights, np.asarray([1.0, 2.0, 3.0]))
    assert model.loads == 1
    assert trainer.loads == 1
    assert result.manifest["identity"]["step"] == 7
    assert result.manifest["identity"]["tokens_seen"] == 128
