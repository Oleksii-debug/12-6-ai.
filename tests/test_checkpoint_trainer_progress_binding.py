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


def identity(
    *,
    seed: int = 7,
    training_config: dict[str, object] | None = None,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-trainer-progress", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config=training_config or {"steps": 10},
        seed=seed,
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


@pytest.mark.parametrize(
    ("expectation", "value", "message"),
    [
        ("expected_init_spec_hash", "not-a-sha", "expected_init_spec_hash"),
        ("expected_packing_hash", "not-a-sha", "expected_packing_hash"),
        (
            "expected_training_config_hash",
            "z" * 64,
            "expected_training_config_hash",
        ),
        (
            "expected_environment_lock_hash",
            "z" * 64,
            "expected_environment_lock_hash",
        ),
        ("expected_split_identity", "", "expected_split_identity"),
        ("expected_packing_version", "", "expected_packing_version"),
        ("expected_previous_run_id", "", "expected_previous_run_id"),
    ],
)
def test_malformed_canonical_expectation_rejected_before_snapshot_or_mutation(
    tmp_path: Path,
    expectation: str,
    value: str,
    message: str,
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

    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=trainer,
            restore_rng=False,
            **{expectation: value},
        )

    np.testing.assert_array_equal(model.weights, before)
    assert model.loads == 0
    assert trainer.loads == 0


def test_trainer_previous_run_id_is_bound_before_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(training_config={"steps": 10, "run_id": "run-parent-a"}),
    )
    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = GenericTrainer()
    before = model.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="previous run id"):
        load_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=trainer,
            restore_rng=False,
            expected_previous_run_id="run-parent-b",
        )

    np.testing.assert_array_equal(model.weights, before)
    assert model.loads == 0
    assert trainer.loads == 0

    result = load_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        restore_rng=False,
        expected_previous_run_id="run-parent-a",
    )
    assert result.manifest["identity"]["training_config"]["run_id"] == "run-parent-a"
    assert model.loads == 1
    assert trainer.loads == 1


def test_equal_malformed_nested_provenance_cannot_self_confirm(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    malformed = {
        "steps": 10,
        "init_spec_sha256": "not-a-sha",
        "data": {
            "split_identity": "",
            "packing_sha256": "not-a-sha",
            "packing_version": "",
        },
    }
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(training_config=malformed),
    )
    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = GenericTrainer()
    before = model.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="expected_init_spec_hash"):
        load_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=trainer,
            restore_rng=False,
            expected_init_spec_hash="not-a-sha",
        )

    np.testing.assert_array_equal(model.weights, before)
    assert model.loads == 0
    assert trainer.loads == 0


def test_boolean_seed_cannot_equal_integer_seed_binding(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(seed=1),
    )
    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = GenericTrainer()
    before = model.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="expected_seed"):
        load_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=trainer,
            restore_rng=False,
            expected_seed=True,
        )

    np.testing.assert_array_equal(model.weights, before)
    assert model.loads == 0
    assert trainer.loads == 0


@pytest.mark.parametrize("failed_stage", ["model", "trainer", "rng"])
def test_partial_restore_poison_prevents_in_place_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    from twelve_six.checkpoint import progress_trainer

    class CanonicalTarget(GenericTrainer):
        def __init__(self) -> None:
            super().__init__()
            self._failure_reason: str | None = None
            self._update_incomplete = False

    checkpoint = tmp_path / "apply-fault"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = CanonicalTarget()
    if failed_stage == "model":
        original_apply = progress_trainer._apply_model_weights

        def broken_apply(*args: object, **kwargs: object) -> None:
            original_apply(*args, **kwargs)
            raise RuntimeError("model apply failed after mutation")

        monkeypatch.setattr(progress_trainer, "_apply_model_weights", broken_apply)
    elif failed_stage == "trainer":
        original_load = CanonicalTarget.load_state_dict

        def broken_load(self: CanonicalTarget, state: dict[str, object]) -> None:
            original_load(self, state)
            if self is trainer:
                raise RuntimeError("trainer apply failed after mutation")

        monkeypatch.setattr(CanonicalTarget, "load_state_dict", broken_load)
    else:
        def broken_rng(_state: object) -> None:
            raise RuntimeError("rng apply failed after model mutation")

        monkeypatch.setattr(progress_trainer, "restore_rng_state", broken_rng)

    with pytest.raises(RuntimeError, match="apply failed"):
        load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer,
            restore_rng=(failed_stage == "rng"),
        )
    assert trainer._failure_reason == "checkpoint_restore_apply_failed"
    assert trainer._update_incomplete is True
    # The real D02 runtime guard refuses optimizer work on this poisoned state.
    from twelve_six.training.trainer import Trainer, TrainingStateInvalidError

    with pytest.raises(TrainingStateInvalidError, match="failed training transition"):
        Trainer._assert_trainable(trainer)
    # The checkpoint was preflighted, but an application-time error may have
    # already changed the model; an in-place restore retry must fail closed.
    with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
        load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=False,
        )


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupted_restore_poison_keeps_original_interruption(
    interrupt: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.checkpoint import progress_trainer

    class CanonicalTarget(GenericTrainer):
        def __init__(self) -> None:
            super().__init__()
            self._failure_reason: str | None = None
            self._update_incomplete = False

    checkpoint = tmp_path / "interrupted-restore"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = CanonicalTarget()

    def interrupted(_state: object) -> None:
        raise interrupt("interrupted during RNG restore")

    monkeypatch.setattr(progress_trainer, "restore_rng_state", interrupted)
    with pytest.raises(interrupt, match="interrupted during RNG restore"):
        load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )
    assert trainer._failure_reason == "checkpoint_restore_apply_failed"
    assert trainer._update_incomplete is True


def test_partial_d02_restore_diagnostic_survives_d05_failure_wrapper(
    tmp_path: Path,
) -> None:
    """D05 must not overwrite D02's original error and cleanup-fault detail."""

    class DiagnosticTarget(GenericTrainer):
        def __init__(self) -> None:
            super().__init__()
            self._failure_reason: str | None = None
            self._update_incomplete = False

        def load_state_dict(self, state: dict[str, object]) -> None:
            super().load_state_dict(state)
            # The isolated preflight uses a copy and must remain non-mutating.
            if self is trainer:
                self._failure_reason = (
                    "trainer state restore failed after possible partial apply; "
                    "gradient cleanup failed: RuntimeError"
                )
                self._update_incomplete = True
                raise RuntimeError("injected optimizer restore failure")

    checkpoint = tmp_path / "d02-diagnostic"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    trainer = DiagnosticTarget()
    model = NumpyModel([9.0, 9.0, 9.0])

    with pytest.raises(RuntimeError, match="injected optimizer restore failure"):
        load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=False,
        )

    assert trainer._failure_reason == (
        "trainer state restore failed after possible partial apply; "
        "gradient cleanup failed: RuntimeError"
    )
    assert trainer._update_incomplete is True
    assert trainer.loads == 1
    from twelve_six.training.trainer import Trainer, TrainingStateInvalidError

    with pytest.raises(TrainingStateInvalidError, match="gradient cleanup failed"):
        Trainer._assert_trainable(trainer)
    with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
        load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=False,
        )
