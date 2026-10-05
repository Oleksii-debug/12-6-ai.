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


class _CanonicalAuthorityProtocol:
    def _require_finite_auxiliary_state(self) -> None:
        return None

    def _require_safe_optimizer_hyperparameters(self, _state: object = None) -> None:
        return None

    def _require_finite_committed_update(self) -> None:
        return None

    def _require_no_residual_model_gradients(self) -> None:
        return None

    def _require_deterministic_policy(self) -> None:
        return None

    def _require_optimizer_parameter_coverage(self) -> None:
        return None

    def _require_finite_state_tree(self, _state: object, _label: str) -> None:
        return None

    def _require_checkpoint_scheduler_chronology(
        self, _scheduler: object, _step: int, _optimizer: object,
    ) -> None:
        return None

    def _require_checkpoint_scaler_state(self, _state: object) -> None:
        return None


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

    class CanonicalTarget(_CanonicalAuthorityProtocol, GenericTrainer):
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
        original_bind = progress_trainer._bind_model_state_loader

        def bind_broken_apply(model: object, strict: bool):
            apply = original_bind(model, strict)

            def broken_apply(materialized: object) -> None:
                apply(materialized)
                raise RuntimeError("model apply failed after mutation")

            return broken_apply

        monkeypatch.setattr(
            progress_trainer,
            "_bind_model_state_loader",
            bind_broken_apply,
        )
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
    if failed_stage == "rng":
        # RNG must be restored last, after trainer state is applied.
        assert trainer.loads == 1
    # The real D02 runtime guard refuses optimizer work on this poisoned state.
    from twelve_six.training.trainer import Trainer, TrainingStateInvalidError

    with pytest.raises(TrainingStateInvalidError, match="failed training transition"):
        Trainer._assert_trainable(trainer)
    # A poisoned retry must reject before any model-scale checkpoint I/O.
    def forbidden_checkpoint_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("poisoned retry must not read checkpoint")

    monkeypatch.setattr(progress_trainer, "prepare_checkpoint_load", forbidden_checkpoint_read)
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

    class CanonicalTarget(_CanonicalAuthorityProtocol, GenericTrainer):
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

    class DiagnosticTarget(_CanonicalAuthorityProtocol, GenericTrainer):
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


def test_fresh_target_recovers_after_partial_model_apply_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.checkpoint import progress_trainer

    class CanonicalTarget(_CanonicalAuthorityProtocol, GenericTrainer):
        def __init__(self) -> None:
            super().__init__()
            self._failure_reason: str | None = None
            self._update_incomplete = False

    checkpoint = tmp_path / "fresh-recovery"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    original_bind = progress_trainer._bind_model_state_loader
    call_count = 0

    def bind_fail_only_first(model: object, strict: bool):
        apply = original_bind(model, strict)

        def fail_only_first_application(materialized: object) -> None:
            nonlocal call_count
            call_count += 1
            apply(materialized)
            if call_count == 1:
                raise RuntimeError("first model apply failed after mutation")

        return fail_only_first_application

    monkeypatch.setattr(
        progress_trainer,
        "_bind_model_state_loader",
        bind_fail_only_first,
    )
    poisoned = CanonicalTarget()
    with pytest.raises(RuntimeError, match="first model apply failed"):
        load_trainer_checkpoint(
            checkpoint,
            model=NumpyModel([9.0, 9.0, 9.0]),
            trainer=poisoned,
            restore_rng=False,
        )
    assert poisoned._failure_reason == "checkpoint_restore_apply_failed"
    assert poisoned._update_incomplete is True

    fresh_model = NumpyModel([8.0, 8.0, 8.0])
    fresh_trainer = CanonicalTarget()
    restored = load_trainer_checkpoint(
        checkpoint, model=fresh_model, trainer=fresh_trainer, restore_rng=False,
    )
    assert call_count == 2
    np.testing.assert_array_equal(fresh_model.weights, np.asarray([1.0, 2.0, 3.0]))
    assert fresh_trainer.loads == 1
    assert fresh_trainer._failure_reason is None
    assert fresh_trainer._update_incomplete is False
    assert restored.manifest["identity"]["step"] == 7
    from twelve_six.training.trainer import Trainer

    Trainer._assert_trainable(fresh_trainer)


def test_rejected_preflight_does_not_poison_fresh_canonical_target(
    tmp_path: Path,
) -> None:
    class CanonicalTarget(_CanonicalAuthorityProtocol, GenericTrainer):
        def __init__(self) -> None:
            super().__init__()
            self._failure_reason: str | None = None
            self._update_incomplete = False

    checkpoint = tmp_path / "preflight-remains-retryable"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    model = NumpyModel([8.0, 8.0, 8.0])
    trainer = CanonicalTarget()
    with pytest.raises(CheckpointCompatibilityError, match="progress mismatch"):
        load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer,
            restore_rng=False, expected_step=8,
        )

    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    assert model.loads == 0
    assert trainer.loads == 0

    result = load_trainer_checkpoint(
        checkpoint, model=model, trainer=trainer,
        restore_rng=False, expected_step=7,
    )
    assert result.manifest["identity"]["step"] == 7
    np.testing.assert_array_equal(model.weights, np.asarray([1.0, 2.0, 3.0]))
    assert model.loads == 1
    assert trainer.loads == 1


@pytest.mark.parametrize("restore_rng", [True, False])
def test_loader_rng_draws_are_rewound_only_when_requested(
    tmp_path: Path,
    restore_rng: bool,
) -> None:
    """The first resumed draw must use checkpoint, not state-loader, RNG."""

    import random
    import torch

    from twelve_six.checkpoint import core

    class DrawingTrainer(GenericTrainer):
        def load_state_dict(self, state: dict[str, object]) -> None:
            super().load_state_dict(state)
            if self is target:
                random.random()
                np.random.random_sample()
                torch.rand(())

    ambient_rng = core.capture_rng_state()
    try:
        random.seed(703)
        np.random.seed(703)
        torch.manual_seed(703)
        checkpoint = tmp_path / "rng-last"
        save_trainer_checkpoint(
            checkpoint,
            model=NumpyModel([1.0, 2.0, 3.0]),
            trainer=GenericTrainer(),
            identity=identity(),
        )

        from twelve_six.checkpoint import progress_trainer

        verified = progress_trainer.prepare_checkpoint_load(checkpoint)
        _, combined = progress_trainer._decode_verified_state(verified)
        saved_rng = combined["rng"]
        expected_python = random.Random()
        expected_python.setstate(saved_rng["python"])
        expected_np = np.random.RandomState()
        expected_np.set_state(saved_rng["numpy"])
        expected_torch = torch.Generator(device="cpu")
        expected_torch.set_state(saved_rng["torch"]["cpu"])
        next_python = expected_python.random()
        next_numpy = expected_np.random_sample()
        next_torch = torch.rand((), generator=expected_torch).item()

        # Ensure that restoring the checkpoint (or opting out) is observable.
        random.random()
        np.random.random_sample()
        torch.rand(())
        target = DrawingTrainer()
        load_trainer_checkpoint(
            checkpoint,
            model=NumpyModel([9.0, 9.0, 9.0]),
            trainer=target,
            restore_rng=restore_rng,
        )
        assert target.loads == 1
        actual = (random.random(), np.random.random_sample(), torch.rand(()).item())
        expected = (next_python, next_numpy, next_torch)
        if restore_rng:
            assert actual == expected
        else:
            assert actual != expected
    finally:
        core.restore_rng_state(ambient_rng)


def test_canonical_progress_restore_rejects_unowned_model_before_checkpoint_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.checkpoint import progress_trainer

    checkpoint = tmp_path / "canonical-model-mismatch"
    save_trainer_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        trainer=GenericTrainer(),
        identity=identity(),
    )
    owned_model = NumpyModel([3.0, 4.0, 5.0])
    wrong_model = NumpyModel([9.0, 9.0, 9.0])

    class CanonicalTarget(_CanonicalAuthorityProtocol, GenericTrainer):
        def __init__(self) -> None:
            super().__init__()
            self.model = owned_model
            self._failure_reason: str | None = None
            self._update_incomplete = False

    trainer = CanonicalTarget()

    def forbidden_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cross-model restore must reject before checkpoint read")

    monkeypatch.setattr(progress_trainer, "prepare_checkpoint_load", forbidden_read)
    with pytest.raises(CheckpointCompatibilityError, match="different model"):
        load_trainer_checkpoint(
            checkpoint, model=wrong_model, trainer=trainer, restore_rng=False,
        )
    np.testing.assert_array_equal(owned_model.weights, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(wrong_model.weights, [9.0, 9.0, 9.0])
    assert owned_model.loads == 0
    assert wrong_model.loads == 0
    assert trainer.loads == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False

@pytest.mark.parametrize("unfresh", ["micro_step", "_pending_tokens"])
def test_progress_refuses_nonfresh_canonical_target_before_checkpoint_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unfresh: str,
) -> None:
    from twelve_six.checkpoint import progress_trainer

    class CanonicalTarget(_CanonicalAuthorityProtocol, GenericTrainer):
        def __init__(self, model: NumpyModel) -> None:
            super().__init__()
            self.model = model
            self._failure_reason: str | None = None
            self._update_incomplete = False
            self.micro_step = 0
            self.optimizer_step = 0
            self.tokens_seen = 0
            self._pending_tokens = 0
            self._pending_loss_sum = 0.0

    model = NumpyModel([9.0, 9.0, 9.0])
    trainer = CanonicalTarget(model)
    setattr(trainer, unfresh, 1)

    def forbidden_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("progress preflight must not read checkpoint")

    monkeypatch.setattr(progress_trainer, "prepare_checkpoint_load", forbidden_read)
    with pytest.raises(CheckpointCompatibilityError, match="fresh trainer"):
        load_trainer_checkpoint(
            tmp_path / "not-opened", model=model, trainer=trainer, restore_rng=False,
        )
    np.testing.assert_array_equal(model.weights, [9.0, 9.0, 9.0])
    assert model.loads == 0
    assert trainer.loads == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
