"""Regression contracts for the canonical D05 trainer adapter restore path."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIdentity
from twelve_six.checkpoint import core, progress_trainer, trainer_adapter


class Model:
    def __init__(self, values: list[float]) -> None:
        self.weights = np.asarray(values, dtype=np.float64).copy()
        self.loads = 0
        self.params = [SimpleNamespace(grad=None)]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()

    def parameters(self) -> list[SimpleNamespace]:
        return self.params


class PlainTrainer:
    def __init__(self) -> None:
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 10}
        self.loads = 0
        self.state: dict[str, object] = {
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


class CanonicalTarget(_CanonicalAuthorityProtocol, PlainTrainer):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self._failure_reason: str | None = None
        self._update_incomplete = False
        self.micro_step = 0
        self.optimizer_step = 0
        self.tokens_seen = 0
        self._pending_tokens = 0
        self._pending_loss_sum = 0.0

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.micro_step = int(state["micro_step"])
        self.optimizer_step = int(state["optimizer_step"])
        self.tokens_seen = int(state["tokens_seen"])


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "adapter-recovery", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 10},
        seed=703,
        precision="float64",
        step=7,
        tokens_seen=128,
        optimizer={"name": "trainer-owned"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def checkpoint_at(path: Path) -> None:
    trainer_adapter.save_trainer_checkpoint(
        path,
        model=Model([1.0, 2.0, 3.0]),
        trainer=PlainTrainer(),
        identity=identity(),
    )


def test_fresh_adapter_target_restores_exact_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fresh"
    checkpoint_at(checkpoint)
    model = Model([9.0, 9.0, 9.0])
    trainer = CanonicalTarget(model)
    result = trainer_adapter.load_trainer_checkpoint(
        checkpoint, model=model, trainer=trainer, restore_rng=False,
    )
    np.testing.assert_array_equal(model.weights, [1.0, 2.0, 3.0])
    assert model.loads == 1
    assert trainer.loads == 1
    assert trainer.micro_step == result.manifest["identity"]["step"]
    assert trainer.optimizer_step == 7
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize(
    "unfresh",
    ["micro_step", "optimizer_step", "tokens_seen", "_pending_tokens",
     "_pending_loss_sum", "gradient"],
)
def test_nonfresh_adapter_target_refused_before_snapshot_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unfresh: str,
) -> None:
    model = Model([9.0, 9.0, 9.0])
    trainer = CanonicalTarget(model)
    if unfresh == "gradient":
        model.params[0].grad = object()
    else:
        setattr(trainer, unfresh, 1)
    def forbidden_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("nonfresh target must not open the checkpoint")

    monkeypatch.setattr(trainer_adapter, "prepare_checkpoint_load", forbidden_read)
    with pytest.raises(CheckpointCompatibilityError, match="fresh trainer"):
        trainer_adapter.load_trainer_checkpoint(
            tmp_path / "not-opened", model=model, trainer=trainer,
        )
    np.testing.assert_array_equal(model.weights, [9.0, 9.0, 9.0])
    assert model.loads == 0
    assert trainer.loads == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize("stage", ["model", "trainer", "rng"])
def test_adapter_partial_apply_poison_and_zero_read_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    checkpoint = tmp_path / "partial"
    checkpoint_at(checkpoint)
    model = Model([9.0, 9.0, 9.0])

    class FailingTarget(CanonicalTarget):
        def load_state_dict(self, state: dict[str, object]) -> None:
            super().load_state_dict(state)
            if self is target and stage == "trainer":
                raise RuntimeError("injected trainer failure after apply")

    target = FailingTarget(model)
    if stage == "model":
        original_bind = trainer_adapter._bind_model_state_loader

        def bind_then_fail(model: object, strict: bool):
            apply = original_bind(model, strict)

            def fail_after_model(materialized: object) -> None:
                apply(materialized)
                raise RuntimeError("injected model failure after apply")

            return fail_after_model

        monkeypatch.setattr(trainer_adapter, "_bind_model_state_loader", bind_then_fail)
    elif stage == "rng":
        def fail_rng(_state: object) -> None:
            raise RuntimeError("injected rng failure after apply")

        monkeypatch.setattr(trainer_adapter, "restore_rng_state", fail_rng)

    with pytest.raises(RuntimeError, match="injected .* failure after apply"):
        trainer_adapter.load_trainer_checkpoint(
            checkpoint, model=model, trainer=target, restore_rng=(stage == "rng"),
        )
    assert target._failure_reason == "checkpoint_restore_apply_failed"
    assert target._update_incomplete is True
    if stage == "rng":
        assert target.loads == 1

    def forbidden_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("poisoned retry must not read checkpoint")

    monkeypatch.setattr(trainer_adapter, "prepare_checkpoint_load", forbidden_read)
    with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
        trainer_adapter.load_trainer_checkpoint(
            checkpoint, model=model, trainer=target, restore_rng=False,
        )


def test_adapter_preserves_d02_diagnostic_on_trainer_apply_failure(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "d02-diagnostic"
    checkpoint_at(checkpoint)
    model = Model([9.0, 9.0, 9.0])

    class DiagnosticTarget(CanonicalTarget):
        def load_state_dict(self, state: dict[str, object]) -> None:
            super().load_state_dict(state)
            if self is target:
                self._failure_reason = "optimizer restore failed; gradient cleanup failed"
                raise RuntimeError("original optimizer failure")

    target = DiagnosticTarget(model)
    with pytest.raises(RuntimeError, match="original optimizer failure"):
        trainer_adapter.load_trainer_checkpoint(
            checkpoint, model=model, trainer=target, restore_rng=False,
        )
    assert target._failure_reason == "optimizer restore failed; gradient cleanup failed"
    assert target._update_incomplete is True


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_adapter_interruption_preserves_original_and_poison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    checkpoint = tmp_path / "interruption"
    checkpoint_at(checkpoint)
    model = Model([9.0, 9.0, 9.0])
    trainer = CanonicalTarget(model)

    def interrupt(_state: object) -> None:
        raise interruption("injected final RNG interruption")

    monkeypatch.setattr(trainer_adapter, "restore_rng_state", interrupt)
    with pytest.raises(interruption, match="injected final RNG interruption"):
        trainer_adapter.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )
    assert trainer.loads == 1
    assert trainer._failure_reason == "checkpoint_restore_apply_failed"
    assert trainer._update_incomplete is True


@pytest.mark.parametrize("restore_rng", [True, False])
def test_adapter_loader_rng_draws_restored_only_when_requested(
    tmp_path: Path,
    restore_rng: bool,
) -> None:
    import torch

    ambient = core.capture_rng_state()
    try:
        random.seed(703)
        np.random.seed(703)
        torch.manual_seed(703)
        checkpoint = tmp_path / "rng-last"
        checkpoint_at(checkpoint)
        verified = trainer_adapter.prepare_checkpoint_load(checkpoint)
        _, decoded = trainer_adapter._decode_verified_state(verified)
        saved = decoded["rng"]
        expected_python = random.Random()
        expected_python.setstate(saved["python"])
        expected_np = np.random.RandomState()
        expected_np.set_state(saved["numpy"])
        expected_torch = torch.Generator(device="cpu")
        expected_torch.set_state(saved["torch"]["cpu"])
        expected = (
            expected_python.random(),
            expected_np.random_sample(),
            torch.rand((), generator=expected_torch).item(),
        )

        random.random()
        np.random.random_sample()
        torch.rand(())
        model = Model([9.0, 9.0, 9.0])

        class DrawingTarget(CanonicalTarget):
            def load_state_dict(self, state: dict[str, object]) -> None:
                super().load_state_dict(state)
                if self is target:
                    random.random()
                    np.random.random_sample()
                    torch.rand(())

        target = DrawingTarget(model)
        trainer_adapter.load_trainer_checkpoint(
            checkpoint, model=model, trainer=target, restore_rng=restore_rng,
        )
        assert target.loads == 1
        actual = (random.random(), np.random.random_sample(), torch.rand(()).item())
        if restore_rng:
            assert actual == expected
        else:
            assert actual != expected
    finally:
        core.restore_rng_state(ambient)


@pytest.mark.parametrize(
    "unfresh",
    ["micro_step", "tokens_seen", "_pending_tokens", "gradient"],
)
def test_actual_d02_target_refused_without_checkpoint_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unfresh: str,
) -> None:
    """Exercise the actual Trainer target, not only a duck-typed test adapter."""

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    try:
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, TrainerConfig(max_steps=10, seed=703))
        before = [parameter.detach().clone() for parameter in model.parameters()]
        if unfresh == "gradient":
            parameter = next(model.parameters())
            parameter.grad = torch.ones_like(parameter)
        else:
            setattr(trainer, unfresh, 1)

        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("D02 nonfresh preflight must precede checkpoint I/O")

        monkeypatch.setattr(trainer_adapter, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="fresh trainer"):
            trainer_adapter.load_trainer_checkpoint(
                tmp_path / "not-opened", model=model, trainer=trainer,
            )
        for parameter, saved in zip(model.parameters(), before, strict=True):
            torch.testing.assert_close(parameter.detach(), saved)
        assert trainer._failure_reason is None
        assert trainer._update_incomplete is False
    finally:
        core.restore_rng_state(ambient)


@pytest.mark.parametrize("operation", ["save", "load"])
def test_canonical_model_owner_mismatch_refuses_before_io_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    owned_model = Model([3.0, 4.0, 5.0])
    wrong_model = Model([9.0, 9.0, 9.0])
    trainer = CanonicalTarget(owned_model)
    destination = tmp_path / "mismatched-model"
    if operation == "save":
        with pytest.raises(CheckpointCompatibilityError, match="different model"):
            trainer_adapter.save_trainer_checkpoint(
                destination, model=wrong_model, trainer=trainer, identity=identity(),
            )
        assert not destination.exists()
    else:
        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("model ownership rejection must precede checkpoint I/O")

        monkeypatch.setattr(trainer_adapter, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="different model"):
            trainer_adapter.load_trainer_checkpoint(
                destination, model=wrong_model, trainer=trainer,
            )
    np.testing.assert_array_equal(owned_model.weights, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(wrong_model.weights, [9.0, 9.0, 9.0])
    assert owned_model.loads == 0
    assert wrong_model.loads == 0
    assert trainer.loads == 0
    assert trainer._failure_reason is None


@pytest.mark.parametrize("operation", ["save", "load"])
def test_real_d02_trainer_rejects_unowned_checkpoint_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    try:
        owned_model = torch.nn.Linear(3, 3)
        wrong_model = torch.nn.Linear(3, 3)
        trainer = Trainer(owned_model, TrainerConfig(max_steps=10, seed=703))
        before = [parameter.detach().clone() for parameter in wrong_model.parameters()]
        destination = tmp_path / "wrong-real-model"
        if operation == "save":
            with pytest.raises(CheckpointCompatibilityError, match="different model"):
                trainer_adapter.save_trainer_checkpoint(
                    destination, model=wrong_model, trainer=trainer, identity=identity(),
                )
            assert not destination.exists()
        else:
            def forbidden_read(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("real D02 model mismatch must precede checkpoint I/O")

            monkeypatch.setattr(trainer_adapter, "prepare_checkpoint_load", forbidden_read)
            with pytest.raises(CheckpointCompatibilityError, match="different model"):
                trainer_adapter.load_trainer_checkpoint(
                    destination, model=wrong_model, trainer=trainer,
                )
        for parameter, saved in zip(wrong_model.parameters(), before, strict=True):
            torch.testing.assert_close(parameter.detach(), saved)
        assert trainer._failure_reason is None
        assert trainer._update_incomplete is False
    finally:
        core.restore_rng_state(ambient)

@pytest.mark.parametrize("use_progress", [False, True])
def test_rejected_semantic_probe_does_not_advance_global_rng(
    tmp_path: Path,
    use_progress: bool,
) -> None:
    """Even a copied adapter's failed loader must not alter ambient RNG."""

    import torch

    ambient = core.capture_rng_state()
    try:
        random.seed(709)
        np.random.seed(709)
        torch.manual_seed(709)
        checkpoint = tmp_path / "rng-preflight"
        checkpoint_at(checkpoint)

        class DrawingRejectingTrainer(PlainTrainer):
            def load_state_dict(self, state: dict[str, object]) -> None:
                random.random()
                np.random.random_sample()
                torch.rand(())
                super().load_state_dict(state)
                raise RuntimeError("injected semantic preflight rejection")

        saved = core.capture_rng_state()
        expected_python = random.Random()
        expected_python.setstate(saved["python"])
        expected_np = np.random.RandomState()
        expected_np.set_state(saved["numpy"])
        expected_torch = torch.Generator(device="cpu")
        expected_torch.set_state(saved["torch"]["cpu"])
        expected = (
            expected_python.random(),
            expected_np.random_sample(),
            torch.rand((), generator=expected_torch).item(),
        )
        model = Model([9.0, 9.0, 9.0])
        trainer = DrawingRejectingTrainer()
        loader = (
            progress_trainer.load_trainer_checkpoint
            if use_progress else trainer_adapter.load_trainer_checkpoint
        )
        with pytest.raises(
            CheckpointCompatibilityError, match="isolated compatibility preflight"
        ):
            loader(checkpoint, model=model, trainer=trainer, restore_rng=False)

        assert (random.random(), np.random.random_sample(), torch.rand(()).item()) == expected
        np.testing.assert_array_equal(model.weights, [9.0, 9.0, 9.0])
        assert model.loads == 0
        assert trainer.loads == 0
    finally:
        core.restore_rng_state(ambient)

@pytest.mark.parametrize("use_progress", [False, True])
def test_successful_semantic_probe_adds_no_extra_rng_draw_on_opt_out(
    tmp_path: Path,
    use_progress: bool,
) -> None:
    """Only the actual trainer load may advance RNG when restore_rng=False."""

    import torch

    ambient = core.capture_rng_state()
    try:
        random.seed(719)
        np.random.seed(719)
        torch.manual_seed(719)
        checkpoint = tmp_path / "rng-opt-out"
        checkpoint_at(checkpoint)

        class DrawingTrainer(PlainTrainer):
            def load_state_dict(self, state: dict[str, object]) -> None:
                super().load_state_dict(state)
                random.random()
                np.random.random_sample()
                torch.rand(())

        saved = core.capture_rng_state()
        expected_python = random.Random()
        expected_python.setstate(saved["python"])
        expected_np = np.random.RandomState()
        expected_np.set_state(saved["numpy"])
        expected_torch = torch.Generator(device="cpu")
        expected_torch.set_state(saved["torch"]["cpu"])
        expected_python.random()
        expected_np.random_sample()
        torch.rand((), generator=expected_torch)
        expected = (
            expected_python.random(),
            expected_np.random_sample(),
            torch.rand((), generator=expected_torch).item(),
        )
        model = Model([9.0, 9.0, 9.0])
        trainer = DrawingTrainer()
        loader = (
            progress_trainer.load_trainer_checkpoint
            if use_progress else trainer_adapter.load_trainer_checkpoint
        )
        loader(checkpoint, model=model, trainer=trainer, restore_rng=False)
        assert trainer.loads == 1
        assert model.loads == 1
        np.testing.assert_array_equal(model.weights, [1.0, 2.0, 3.0])
        assert (random.random(), np.random.random_sample(), torch.rand(()).item()) == expected
    finally:
        core.restore_rng_state(ambient)


def test_semantic_preflight_preserves_torch_deterministic_warn_mode() -> None:
    import torch

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        trainer = PlainTrainer()
        trainer_adapter._preflight_trainer_state(
            trainer, trainer.state_dict(),
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
@pytest.mark.parametrize("probe_rejects", [False, True])
def test_failed_preflight_rng_rollback_poisons_canonical_target_before_model_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
    probe_rejects: bool,
) -> None:
    """Failed RNG rollback makes even an otherwise fresh target unsafe to reuse."""

    import torch

    checkpoint = tmp_path / "rollback-fault"
    checkpoint_at(checkpoint)
    model = Model([9.0, 9.0, 9.0])

    class ProbeTarget(CanonicalTarget):
        def load_state_dict(self, state: dict[str, object]) -> None:
            super().load_state_dict(state)
            if probe_rejects:
                raise RuntimeError("detached trainer probe rejected state")

    trainer = ProbeTarget(model)
    loader_module = progress_trainer if use_progress else trainer_adapter
    original_restore = core.restore_rng_state
    ambient = core.capture_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail_rng_rollback(_state: object) -> None:
        random.random()
        torch.use_deterministic_algorithms(not deterministic, warn_only=not warn_only)
        raise OSError("injected ambient RNG rollback failure")

    try:
        monkeypatch.setattr(core, "restore_rng_state", fail_rng_rollback)
        with pytest.raises(OSError, match="ambient RNG rollback failure") as raised:
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
        if probe_rejects:
            assert isinstance(raised.value.__context__, CheckpointCompatibilityError)
            assert "isolated compatibility preflight" in str(raised.value.__context__)

        assert trainer._failure_reason == "checkpoint_preflight_rng_rollback_failed"
        assert trainer._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled() is deterministic
        assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
        np.testing.assert_array_equal(model.weights, [9.0, 9.0, 9.0])
        assert model.loads == 0
        assert trainer.loads == 0

        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("poisoned retry must not read a checkpoint")

        monkeypatch.setattr(loader_module, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
    finally:
        original_restore(ambient)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
def test_real_d02_trainer_refuses_training_after_failed_probe_rng_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
) -> None:
    """An unrecoverable preflight RNG fault must poison the actual D02 runtime."""

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer, TrainingStateInvalidError

    from dataclasses import replace

    checkpoint = tmp_path / "real-d02-rollback"
    ambient = core.capture_rng_state()
    original_restore = core.restore_rng_state
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        source_model = torch.nn.Linear(3, 3)
        source_trainer = Trainer(source_model, TrainerConfig(max_steps=10, seed=703))
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source_trainer,
            identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, TrainerConfig(max_steps=10, seed=703))
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loader_module = progress_trainer if use_progress else trainer_adapter

        def fail_rollback(_state: object) -> None:
            raise OSError("injected actual D02 RNG rollback failure")

        monkeypatch.setattr(core, "restore_rng_state", fail_rollback)
        with pytest.raises(OSError, match="actual D02 RNG rollback failure"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
        assert trainer._failure_reason == "checkpoint_preflight_rng_rollback_failed"
        assert trainer._update_incomplete is True
        with pytest.raises(TrainingStateInvalidError, match="failed training transition"):
            trainer._assert_trainable()
        for parameter, saved in zip(model.parameters(), before, strict=True):
            torch.testing.assert_close(parameter.detach(), saved)

        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("poisoned real D02 retry must not read checkpoint")

        monkeypatch.setattr(loader_module, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
    finally:
        original_restore(ambient)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
def test_final_rng_replay_preserves_torch_warn_only_policy(
    tmp_path: Path,
    use_progress: bool,
) -> None:
    """Checkpoint replay must not silently change warning into hard failure."""

    import torch

    ambient = core.capture_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        checkpoint = tmp_path / "warn-only-final-rng"
        checkpoint_at(checkpoint)
        model = Model([9.0, 9.0, 9.0])
        trainer = PlainTrainer()
        loader = (
            progress_trainer.load_trainer_checkpoint
            if use_progress else trainer_adapter.load_trainer_checkpoint
        )
        loader(checkpoint, model=model, trainer=trainer, restore_rng=True)
        assert model.loads == 1
        assert trainer.loads == 1
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
def test_real_d02_replay_keeps_configured_torch_warn_only(
    tmp_path: Path,
    use_progress: bool,
) -> None:
    """A resumed actual D02 trainer retains its configured deterministic policy."""

    from dataclasses import replace

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        config = TrainerConfig(max_steps=10, seed=703, deterministic_warn_only=True)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        checkpoint = tmp_path / "real-d02-warn-only"
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        loader = (
            progress_trainer.load_trainer_checkpoint
            if use_progress else trainer_adapter.load_trainer_checkpoint
        )
        loader(checkpoint, model=model, trainer=trainer, restore_rng=True)
        assert trainer.optimizer_step == 0
        assert trainer._failure_reason is None
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

@pytest.mark.parametrize("use_progress", [False, True])
@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_partial_final_rng_replay_rolls_back_torch_mode_and_poisons_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
    interruption: type[BaseException],
) -> None:
    """A partial final RNG replay must not leak a changed global PyTorch mode."""

    import torch

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        checkpoint = tmp_path / "partial-final-rng-mode"
        checkpoint_at(checkpoint)
        model = Model([9.0, 9.0, 9.0])
        trainer = CanonicalTarget(model)
        loader_module = progress_trainer if use_progress else trainer_adapter
        original_error = interruption("injected partial final RNG replay")

        def fail_after_mode_change(_state: object) -> None:
            torch.use_deterministic_algorithms(False, warn_only=False)
            raise original_error

        monkeypatch.setattr(loader_module, "restore_rng_state", fail_after_mode_change)
        with pytest.raises(interruption, match="injected partial final RNG replay") as raised:
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=True,
            )
        assert raised.value is original_error
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert model.loads == 1
        assert trainer.loads == 1
        assert trainer._failure_reason == "checkpoint_restore_apply_failed"
        assert trainer._update_incomplete is True

        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("poisoned retry must not read the checkpoint")

        monkeypatch.setattr(loader_module, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_failed_final_rng_mode_rollback_retains_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secondary PyTorch policy failure must not replace the RNG error."""

    import torch

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_use = torch.use_deterministic_algorithms
    try:
        original_use(True, warn_only=True)

        def fail_mode_rollback(requested: bool, *, warn_only: bool = False) -> None:
            if requested and warn_only:
                raise OSError("injected secondary mode rollback failure")
            original_use(requested, warn_only=warn_only)

        def fail_rng(_state: object) -> None:
            original_use(False, warn_only=False)
            raise RuntimeError("injected primary final RNG failure")

        monkeypatch.setattr(torch, "use_deterministic_algorithms", fail_mode_rollback)
        with pytest.raises(RuntimeError, match="primary final RNG failure") as raised:
            trainer_adapter._restore_checkpoint_rng_preserving_warn_only(
                {"torch": {"cpu": object()}}, restore=fail_rng,
            )
        assert any(
            "secondary mode rollback failure" in note
            for note in getattr(raised.value, "__notes__", ())
        )
    finally:
        original_use(enabled, warn_only=warn_only)

@pytest.mark.parametrize("use_progress", [False, True])
def test_real_d02_partial_final_rng_failure_poisons_and_preserves_torch_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
) -> None:
    """A partial final RNG fault must poison the actual trainer, not just test doubles."""

    from dataclasses import replace

    import torch
    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer, TrainingStateInvalidError

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        config = TrainerConfig(max_steps=10, seed=703, deterministic_warn_only=True)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        checkpoint = tmp_path / "real-d02-partial-final-rng"
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        loader_module = progress_trainer if use_progress else trainer_adapter
        original_error = RuntimeError("injected real D02 partial final RNG failure")

        def fail_after_mode_change(_state: object) -> None:
            torch.use_deterministic_algorithms(False, warn_only=False)
            raise original_error

        monkeypatch.setattr(loader_module, "restore_rng_state", fail_after_mode_change)
        with pytest.raises(RuntimeError, match="real D02 partial final RNG") as raised:
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=True,
            )
        assert raised.value is original_error
        assert trainer._failure_reason == "checkpoint_restore_apply_failed"
        assert trainer._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        with pytest.raises(TrainingStateInvalidError, match="failed training transition"):
            trainer._assert_trainable()

        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("poisoned real D02 retry must not read checkpoint")

        monkeypatch.setattr(loader_module, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

@pytest.mark.parametrize("use_progress", [False, True])
@pytest.mark.parametrize("probe_rejects", [False, True])
def test_preflight_double_rollback_fault_keeps_primary_rng_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
    probe_rejects: bool,
) -> None:
    """A secondary PyTorch mode error must not mask the original RNG fault."""

    import torch

    checkpoint = tmp_path / "preflight-double-rollback-fault"
    checkpoint_at(checkpoint)
    model = Model([9.0, 9.0, 9.0])

    class ProbeTarget(CanonicalTarget):
        def load_state_dict(self, state: dict[str, object]) -> None:
            super().load_state_dict(state)
            if probe_rejects:
                raise ValueError("injected isolated preflight rejection")

    trainer = ProbeTarget(model)
    loader_module = progress_trainer if use_progress else trainer_adapter
    ambient = core.capture_rng_state()
    original_restore = core.restore_rng_state
    original_use = torch.use_deterministic_algorithms
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_error = OSError("primary preflight RNG rollback failure")
    try:
        original_use(True, warn_only=True)

        def fail_rng(_state: object) -> None:
            raise original_error

        def fail_mode(requested: bool, *, warn_only: bool = False) -> None:
            if requested and warn_only:
                raise RuntimeError("secondary preflight mode rollback failure")
            original_use(requested, warn_only=warn_only)

        monkeypatch.setattr(core, "restore_rng_state", fail_rng)
        monkeypatch.setattr(torch, "use_deterministic_algorithms", fail_mode)
        with pytest.raises(OSError, match="primary preflight RNG rollback") as raised:
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
        assert raised.value is original_error
        assert any(
            "secondary preflight mode rollback failure" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        if probe_rejects:
            assert isinstance(raised.value.__context__, CheckpointCompatibilityError)
            assert "isolated compatibility preflight" in str(raised.value.__context__)
        assert trainer._failure_reason == "checkpoint_preflight_rng_rollback_failed"
        assert trainer._update_incomplete is True
        assert model.loads == 0
        assert trainer.loads == 0

        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("poisoned retry must not read a checkpoint")

        monkeypatch.setattr(loader_module, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
    finally:
        original_restore(ambient)
        original_use(enabled, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
def test_failed_final_success_policy_application_rolls_back_original_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
) -> None:
    """Failure after successful RNG replay must still restore the old mode."""

    import torch

    ambient = core.capture_rng_state()
    original_use = torch.use_deterministic_algorithms
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        original_use(True, warn_only=True)
        checkpoint = tmp_path / "final-policy-fault"
        checkpoint_at(checkpoint)
        model = Model([9.0, 9.0, 9.0])
        trainer = CanonicalTarget(model)
        loader_module = progress_trainer if use_progress else trainer_adapter

        def replay(_state: object) -> None:
            original_use(False, warn_only=False)

        def fail_success_policy(requested: bool, *, warn_only: bool = False) -> None:
            if not requested and warn_only:
                raise OSError("injected final success-policy failure")
            original_use(requested, warn_only=warn_only)

        monkeypatch.setattr(loader_module, "restore_rng_state", replay)
        monkeypatch.setattr(torch, "use_deterministic_algorithms", fail_success_policy)
        with pytest.raises(OSError, match="final success-policy failure"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=True,
            )
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert trainer._failure_reason == "checkpoint_restore_apply_failed"
        assert trainer._update_incomplete is True
        assert model.loads == 1
        assert trainer.loads == 1
    finally:
        core.restore_rng_state(ambient)
        original_use(enabled, warn_only=warn_only)

@pytest.mark.parametrize("use_progress", [False, True])
@pytest.mark.parametrize("restore_rng", [False, True])
def test_real_d02_rejects_checkpoint_rng_policy_drift_before_model_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
    restore_rng: bool,
) -> None:
    """A sealed checkpoint cannot silently override D02's deterministic config."""

    from dataclasses import asdict, replace

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        config = TrainerConfig(max_steps=10, seed=703, deterministic_algorithms=True)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        checkpoint = tmp_path / "mismatched-deterministic-policy"
        # Test the loader against an integrity-valid but semantically invalid
        # low-level checkpoint; the public D02 save adapter now rejects it.
        torch.use_deterministic_algorithms(False, warn_only=False)
        core.save_checkpoint(
            checkpoint,
            model=source_model,
            trainer_state=asdict(source.state_dict()),
            identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loader_module = progress_trainer if use_progress else trainer_adapter
        if restore_rng:
            def forbidden_materialization(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("policy mismatch must fail before model materialization")

            monkeypatch.setattr(
                loader_module, "_prepare_model_weights", forbidden_materialization,
            )
            with pytest.raises(CheckpointCompatibilityError, match="deterministic_algorithms"):
                loader_module.load_trainer_checkpoint(
                    checkpoint, model=model, trainer=trainer, restore_rng=True,
                )
            for parameter, saved in zip(model.parameters(), before, strict=True):
                torch.testing.assert_close(parameter.detach(), saved)
        else:
            # Opting out of checkpoint RNG replay is an explicit caller choice.
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
            for parameter, saved in zip(
                model.parameters(), source_model.parameters(), strict=True,
            ):
                torch.testing.assert_close(parameter.detach(), saved.detach())
        assert trainer._failure_reason is None
        assert trainer._update_incomplete is False
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

@pytest.mark.parametrize("drift", ["enabled", "warn_only"])
def test_real_d02_refuses_to_publish_checkpoint_with_ambient_policy_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    """A live policy mismatch must not create a non-resumable D02 checkpoint."""

    from dataclasses import replace

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        config = TrainerConfig(max_steps=10, seed=703, deterministic_warn_only=True)
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        if drift == "enabled":
            torch.use_deterministic_algorithms(False, warn_only=True)
        else:
            torch.use_deterministic_algorithms(True, warn_only=False)
        destination = tmp_path / "must-not-publish"
        with pytest.raises(CheckpointCompatibilityError, match="live torch deterministic policy"):
            trainer_adapter.save_trainer_checkpoint(
                destination,
                model=model,
                trainer=trainer,
                identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
            )
        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []
        assert trainer._failure_reason is None
        assert trainer._update_incomplete is False
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
@pytest.mark.parametrize("restore_rng", [False, True])
@pytest.mark.parametrize("drift", ["enabled", "warn_only"])
def test_real_d02_refuses_restore_under_live_policy_drift_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
    restore_rng: bool,
    drift: str,
) -> None:
    """Even opting out of replay cannot make an unsafe ambient D02 mode safe."""

    from dataclasses import replace

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        config = TrainerConfig(max_steps=10, seed=703, deterministic_warn_only=True)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        checkpoint = tmp_path / "valid-checkpoint"
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        if drift == "enabled":
            torch.use_deterministic_algorithms(False, warn_only=True)
        else:
            torch.use_deterministic_algorithms(True, warn_only=False)
        loader_module = progress_trainer if use_progress else trainer_adapter

        def forbidden_materialization(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("live policy mismatch must reject before materialization")

        monkeypatch.setattr(loader_module, "_prepare_model_weights", forbidden_materialization)
        with pytest.raises(CheckpointCompatibilityError, match="live torch deterministic policy"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=restore_rng,
            )
        for parameter, saved in zip(model.parameters(), before, strict=True):
            torch.testing.assert_close(parameter.detach(), saved)
        assert trainer._failure_reason is None
        assert trainer._update_incomplete is False
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

@pytest.mark.parametrize("use_progress", [False, True])
def test_successful_generic_rng_replay_ignores_trainer_policy_side_effect(
    tmp_path: Path,
    use_progress: bool,
) -> None:
    """Warn-only must come from pre-apply policy, not a loader side effect."""

    import torch

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        checkpoint = tmp_path / "policy-mutating-generic-trainer"
        checkpoint_at(checkpoint)

        class PolicyMutatingTrainer(PlainTrainer):
            def load_state_dict(self, state: dict[str, object]) -> None:
                super().load_state_dict(state)
                torch.use_deterministic_algorithms(False, warn_only=False)

        model = Model([9.0, 9.0, 9.0])
        trainer = PolicyMutatingTrainer()
        loader_module = progress_trainer if use_progress else trainer_adapter
        loader_module.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )
        assert model.loads == 1
        assert trainer.loads == 1
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
def test_real_d02_no_replay_rejects_loader_policy_side_effect_and_poison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
) -> None:
    """A successful D02 state load cannot return with a different live mode."""

    from dataclasses import replace

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        config = TrainerConfig(max_steps=10, seed=703, deterministic_warn_only=True)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        checkpoint = tmp_path / "d02-mutating-no-replay"
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        normal_load = trainer.load_state_dict

        def change_mode_after_load(state: object) -> None:
            normal_load(state)
            torch.use_deterministic_algorithms(False, warn_only=False)

        monkeypatch.setattr(trainer, "load_state_dict", change_mode_after_load)
        loader_module = progress_trainer if use_progress else trainer_adapter
        with pytest.raises(CheckpointCompatibilityError, match="live torch deterministic policy"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
        assert trainer._failure_reason == "checkpoint_restore_apply_failed"
        assert trainer._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

        def forbidden_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("poisoned retry must reject before checkpoint read")

        monkeypatch.setattr(loader_module, "prepare_checkpoint_load", forbidden_read)
        with pytest.raises(CheckpointCompatibilityError, match="poisoned"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=False,
            )
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt])
def test_real_d02_loader_interruption_restores_pre_apply_mode_and_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
    interruption: type[BaseException],
) -> None:
    """An interrupted trainer load must not leave global PyTorch mode corrupted."""

    from dataclasses import replace

    import torch

    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        config = TrainerConfig(max_steps=10, seed=703, deterministic_warn_only=True)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        checkpoint = tmp_path / "interrupted-d02-loader"
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=replace(identity(), parameter_count=12, step=0, tokens_seen=0),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        primary_error = interruption("injected trainer policy interruption")
        normal_load = trainer.load_state_dict

        def fail_after_load(state: object) -> None:
            normal_load(state)
            torch.use_deterministic_algorithms(False, warn_only=False)
            raise primary_error

        monkeypatch.setattr(trainer, "load_state_dict", fail_after_load)
        loader_module = progress_trainer if use_progress else trainer_adapter
        with pytest.raises(interruption, match="trainer policy interruption") as raised:
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=True,
            )
        assert raised.value is primary_error
        assert trainer._failure_reason == "checkpoint_restore_apply_failed"
        assert trainer._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

@pytest.mark.parametrize("use_progress", [False, True])
@pytest.mark.parametrize("restore_rng", [False, True])
def test_failed_apply_rolls_back_all_ambient_rng_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
    restore_rng: bool,
) -> None:
    """Partial trainer or final RNG failure must not corrupt unrelated draws."""

    import torch

    original = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        random.seed(1711)
        np.random.seed(1711)
        torch.manual_seed(1711)
        torch.use_deterministic_algorithms(True, warn_only=True)
        checkpoint = tmp_path / "ambient-rng-rollback"
        checkpoint_at(checkpoint)
        model = Model([9.0, 9.0, 9.0])

        class InterruptingTrainer(CanonicalTarget):
            def load_state_dict(self, state: dict[str, object]) -> None:
                super().load_state_dict(state)
                if self is trainer and not restore_rng:
                    random.random()
                    np.random.random_sample()
                    torch.rand(())
                    torch.use_deterministic_algorithms(False, warn_only=False)
                    raise RuntimeError("injected trainer load RNG failure")

        trainer = InterruptingTrainer(model)
        loader_module = progress_trainer if use_progress else trainer_adapter
        random.random()
        np.random.random_sample()
        torch.rand(())
        ambient = core.capture_rng_state()
        python_probe = random.Random()
        python_probe.setstate(ambient["python"])
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(ambient["numpy"])
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(ambient["torch"]["cpu"])
        expected = (
            python_probe.random(),
            numpy_probe.random_sample(),
            torch.rand((), generator=torch_probe).item(),
        )

        if restore_rng:
            def fail_replay(_state: object) -> None:
                random.random()
                np.random.random_sample()
                torch.rand(())
                torch.use_deterministic_algorithms(False, warn_only=False)
                raise RuntimeError("injected final replay RNG failure")

            monkeypatch.setattr(loader_module, "restore_rng_state", fail_replay)
        with pytest.raises(RuntimeError, match="injected .* RNG failure"):
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=restore_rng,
            )
        assert trainer._failure_reason == "checkpoint_restore_apply_failed"
        assert trainer._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert (random.random(), np.random.random_sample(), torch.rand(()).item()) == expected
    finally:
        core.restore_rng_state(original)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("use_progress", [False, True])
def test_failed_ambient_rng_rollback_preserves_primary_and_poisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_progress: bool,
) -> None:
    """The rollback failure cannot turn a failed restore into a false success."""

    import torch

    original = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_restore = core.restore_rng_state
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        checkpoint = tmp_path / "double-rng-failure"
        checkpoint_at(checkpoint)
        model = Model([9.0, 9.0, 9.0])
        trainer = CanonicalTarget(model)
        loader_module = progress_trainer if use_progress else trainer_adapter
        original_error = RuntimeError("primary checkpoint replay failure")
        rollbacks = 0

        def fail_ambient_rollback(state: object) -> None:
            nonlocal rollbacks
            rollbacks += 1
            if rollbacks >= 2:
                raise OSError("secondary ambient RNG rollback failure")
            original_restore(state)

        def fail_final_replay(_state: object) -> None:
            raise original_error

        monkeypatch.setattr(core, "restore_rng_state", fail_ambient_rollback)
        monkeypatch.setattr(loader_module, "restore_rng_state", fail_final_replay)
        with pytest.raises(RuntimeError, match="primary checkpoint replay") as raised:
            loader_module.load_trainer_checkpoint(
                checkpoint, model=model, trainer=trainer, restore_rng=True,
            )
        assert raised.value is original_error
        assert any(
            "secondary ambient RNG rollback failure" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        assert trainer._failure_reason == "checkpoint_restore_apply_failed"
        assert trainer._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        original_restore(original)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
