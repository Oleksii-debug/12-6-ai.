from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twelve_six.checkpoint.core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    load_checkpoint,
    save_checkpoint,
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


class SimpleOptimizer:
    def __init__(self, value: int) -> None:
        self.value = value
        self.loads = 0

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.loads += 1
        self.value = state["value"]


class SourceScheduler:
    def state_dict(self) -> dict[str, Any]:
        return {"counter": 1, "history": "corrupt-history-shape"}


class TargetScheduler:
    def __init__(self) -> None:
        self.counter = 0
        self.history = [0.25]
        self.loads = 0

    def state_dict(self) -> dict[str, Any]:
        return {"counter": self.counter, "history": list(self.history)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loads += 1
        if not isinstance(state.get("history"), list):
            raise ValueError("scheduler history must be a list")
        self.counter = state["counter"]
        self.history = list(state["history"])


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "direct-scheduler-preflight", "parameters": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": True},
        seed=17,
        precision="float64",
        step=1,
        tokens_seen=3,
        optimizer={"name": "SimpleOptimizer"},
        scheduler={"name": "TargetScheduler"},
        environment_lock_hash="f" * 64,
    )


def test_direct_scheduler_mismatch_fails_before_model_or_optimizer_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "direct-scheduler-mismatch"
    source_model = NumpyModel([1.0, 2.0, 3.0])
    source_optimizer = SimpleOptimizer(7)
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=SourceScheduler(),
        identity=_identity(),
    )

    target_model = NumpyModel([9.0, 9.0, 9.0])
    target_optimizer = SimpleOptimizer(99)
    target_scheduler = TargetScheduler()
    before_model = target_model.weights.copy()
    before_optimizer = target_optimizer.value
    before_scheduler = target_scheduler.state_dict()

    with pytest.raises(CheckpointCompatibilityError, match="scheduler state"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            restore_rng=False,
        )

    np.testing.assert_array_equal(target_model.weights, before_model)
    assert target_model.loads == 0
    assert target_optimizer.value == before_optimizer
    assert target_optimizer.loads == 0
    assert target_scheduler.state_dict() == before_scheduler
    assert target_scheduler.loads == 0
