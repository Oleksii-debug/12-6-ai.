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


class CanonicalTarget(PlainTrainer):
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
        original = trainer_adapter._apply_model_weights

        def fail_after_model(*args: object, **kwargs: object) -> None:
            original(*args, **kwargs)
            raise RuntimeError("injected model failure after apply")

        monkeypatch.setattr(trainer_adapter, "_apply_model_weights", fail_after_model)
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
