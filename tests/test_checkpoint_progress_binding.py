from __future__ import annotations

from pathlib import Path

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


def identity(*, step: int = 7, tokens_seen: int = 128) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-progress-binding", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 10},
        seed=7,
        precision="float64",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "test"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def test_expected_step_mismatch_fails_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1.0, 2.0, 3.0]), identity=identity())

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="progress mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            expected_step=8,
        )

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_expected_tokens_seen_mismatch_fails_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1.0, 2.0, 3.0]), identity=identity())

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="progress mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            expected_tokens_seen=127,
        )

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_exact_progress_binding_allows_intended_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1.0, 2.0, 3.0]), identity=identity())

    target = NumpyModel([9.0, 9.0, 9.0])
    result = load_checkpoint(
        checkpoint,
        model=target,
        restore_rng=False,
        expected_step=7,
        expected_tokens_seen=128,
    )

    np.testing.assert_array_equal(target.weights, np.asarray([1.0, 2.0, 3.0]))
    assert target.loads == 1
    assert result.manifest["identity"]["step"] == 7
    assert result.manifest["identity"]["tokens_seen"] == 128


@pytest.mark.parametrize("field", ["expected_step", "expected_tokens_seen"])
def test_invalid_expected_progress_value_fails_closed(tmp_path: Path, field: str) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1.0, 2.0, 3.0]), identity=identity())

    target = NumpyModel([9.0, 9.0, 9.0])
    kwargs = {field: -1}
    with pytest.raises(CheckpointCompatibilityError, match=field):
        load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            **kwargs,
        )

    assert target.loads == 0
