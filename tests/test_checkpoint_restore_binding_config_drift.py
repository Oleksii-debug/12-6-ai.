"""D05 must pin native TrainerConfig values across the full restore boundary.

Synthetic CPU coverage only; this grants no corpus, training, or learned-weight evidence.
"""
from __future__ import annotations

import random
from dataclasses import asdict, fields, make_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, core
from twelve_six.checkpoint import progress_trainer, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


class _TinyLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "restore-binding-config-drift", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=919,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler={"name": "cosine"},
        environment_lock_hash="f" * 64,
    )


def _fresh_identity() -> CheckpointIdentity:
    return replace(_identity(), step=0, tokens_seen=0)


def _source() -> Trainer:
    source = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    return source


@pytest.mark.parametrize(
    ("loader", "final_restore_call"),
    [
        (trainer_adapter, 6),
        (progress_trainer, 3),
    ],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
def test_final_process_state_rollback_cannot_redefine_native_config_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    final_restore_call: int,
    restore_rng: bool,
) -> None:
    source = _source()
    checkpoint = tmp_path / "config-boundary-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    initial_max_steps = target.config.max_steps
    assert not torch.equal(initial_weights, source.model.weight.detach())

    original_restore = loader._restore_preapply_process_state
    original_bind = loader._bind_model_state_loader
    restore_calls = 0
    model_applications: list[bool] = []

    def restore_then_mutate(*args: Any, **kwargs: Any) -> Any:
        nonlocal restore_calls
        result = original_restore(*args, **kwargs)
        restore_calls += 1
        if restore_calls == final_restore_call:
            object.__setattr__(
                target.config,
                "max_steps",
                target.config.max_steps + 1,
            )
        return result

    def bind_tracked_model_loader(model: Any, strict: bool):
        apply = original_bind(model, strict)

        def tracked_apply(materialized: Any) -> Any:
            model_applications.append(True)
            return apply(materialized)

        return tracked_apply

    monkeypatch.setattr(
        loader,
        "_restore_preapply_process_state",
        restore_then_mutate,
    )
    monkeypatch.setattr(loader, "_bind_model_state_loader", bind_tracked_model_loader)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer
        else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="canonical trainer config changed during checkpoint restore",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert restore_calls == final_restore_call
    assert model_applications == []
    assert target.config.max_steps == initial_max_steps + 1
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)

@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_entry_config_snapshot_refuses_effectful_slot_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    original_slot = TrainerConfig.__dict__["max_steps"]
    hook_calls: list[bool] = []
    checkpoint_reads: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    class EffectfulSlot:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            hook_calls.append(True)
            random.random()
            np.random.random()
            torch.rand(1)
            return original_slot.__get__(instance, owner)

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("effectful config descriptor reached checkpoint I/O")

    monkeypatch.setattr(TrainerConfig, "max_steps", EffectfulSlot())
    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="config fields must remain inert slots",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=target.model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert hook_calls == []
    assert checkpoint_reads == []
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False



@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_entry_config_snapshot_requires_exact_native_config_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    ImpostorConfig = make_dataclass(
        "TrainerConfig",
        [(field.name, field.type) for field in fields(TrainerConfig)],
        namespace={"__module__": "twelve_six.training.config"},
        frozen=True,
        slots=True,
    )
    target.config = ImpostorConfig(**asdict(target.config))
    assert type(target.config).__module__ == "twelve_six.training.config"
    assert type(target.config).__name__ == "TrainerConfig"

    checkpoint_reads: list[bool] = []

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("spoofed native config reached checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="must remain canonical TrainerConfig",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=target.model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert checkpoint_reads == []
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_native_d02_detection_requires_exact_trainer_lineage() -> None:
    FakeTrainer = type(
        "Trainer",
        (),
        {"__module__": "twelve_six.training.trainer"},
    )
    target = FakeTrainer()
    target._failure_reason = None
    target._update_incomplete = False

    assert trainer_adapter._is_canonical_d02(target)
    assert not trainer_adapter._is_native_d02(target)


def test_native_checkpoint_save_rejects_instance_state_export_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    shadow_calls: list[bool] = []
    save_calls: list[bool] = []

    def shadow_state_dict() -> Any:
        shadow_calls.append(True)
        return target.__class__.state_dict(target)

    def forbid_save(*args: Any, **kwargs: Any) -> Any:
        save_calls.append(True)
        raise AssertionError("instance-shadowed trainer export reached checkpoint I/O")

    monkeypatch.setattr(target, "state_dict", shadow_state_dict)
    monkeypatch.setattr(trainer_adapter, "save_checkpoint", forbid_save)

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="state_dict must remain class-bound",
    ):
        trainer_adapter.save_trainer_checkpoint(
            tmp_path / "must-not-write",
            model=target.model,
            trainer=target,
            identity=_identity(),
        )

    assert shadow_calls == []
    assert save_calls == []
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_native_checkpoint_save_restores_export_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EffectfulExportTrainer(Trainer):
        def state_dict(self) -> Any:
            state = super().state_dict()
            random.random()
            np.random.random()
            torch.rand(1)
            torch.use_deterministic_algorithms(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=not torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            return state

    target = EffectfulExportTrainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    save_calls: list[bool] = []

    def observe_save(*args: Any, **kwargs: Any) -> dict[str, Any]:
        save_calls.append(True)
        assert random.getstate() == py_before
        np_now = np.random.get_state()
        assert np_now[0] == np_before[0]
        np.testing.assert_array_equal(np_now[1], np_before[1])
        assert np_now[2:] == np_before[2:]
        torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ) == policy_before
        return {"sealed": True}

    monkeypatch.setattr(trainer_adapter, "save_checkpoint", observe_save)

    result = trainer_adapter.save_trainer_checkpoint(
        tmp_path / "observed-only",
        model=target.model,
        trainer=target,
        identity=_fresh_identity(),
    )

    assert result == {"sealed": True}
    assert save_calls == [True]
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_native_checkpoint_save_rejects_export_counter_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CounterDriftExportTrainer(Trainer):
        def state_dict(self) -> Any:
            state = super().state_dict()
            self.tokens_seen += 1
            return state

    target = CounterDriftExportTrainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    save_calls: list[bool] = []

    def forbid_save(*args: Any, **kwargs: Any) -> Any:
        save_calls.append(True)
        raise AssertionError("drifted native export reached checkpoint publication")

    monkeypatch.setattr(trainer_adapter, "save_checkpoint", forbid_save)

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="post-load tokens_seen disagrees with checkpoint",
    ):
        trainer_adapter.save_trainer_checkpoint(
            tmp_path / "must-not-write",
            model=target.model,
            trainer=target,
            identity=_identity(),
        )

    assert save_calls == []
    assert target.tokens_seen == 1
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True


@pytest.mark.parametrize(
    ("method_name", "binder"),
    [
        ("state_dict", trainer_adapter._bind_trainer_state_exporter),
        ("load_state_dict", trainer_adapter._bind_trainer_state_loader),
    ],
    ids=["export", "load"],
)
def test_native_trainer_method_binding_does_not_execute_class_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    binder: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    descriptor_calls: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    class EffectfulMethod:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            descriptor_calls.append(True)
            random.random()
            np.random.random()
            torch.rand(1)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(Trainer, method_name, EffectfulMethod())

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match=f"{method_name} must remain class-bound",
    ):
        binder(target)

    assert descriptor_calls == []
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


def test_checkpoint_save_refuses_effectful_config_slot_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    original_slot = TrainerConfig.__dict__["max_steps"]
    descriptor_calls: list[bool] = []
    save_calls: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    class EffectfulSlot:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            descriptor_calls.append(True)
            random.random()
            np.random.random()
            torch.rand(1)
            return original_slot.__get__(instance, owner)

    def forbid_save(*args: Any, **kwargs: Any) -> Any:
        save_calls.append(True)
        raise AssertionError("effectful config descriptor reached checkpoint publication")

    monkeypatch.setattr(TrainerConfig, "max_steps", EffectfulSlot())
    monkeypatch.setattr(trainer_adapter, "save_checkpoint", forbid_save)

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="config fields must remain inert slots",
    ):
        trainer_adapter.save_trainer_checkpoint(
            tmp_path / "must-not-write",
            model=target.model,
            trainer=target,
            identity=_identity(),
        )

    assert descriptor_calls == []
    assert save_calls == []
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


def test_native_determinism_check_does_not_execute_config_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    original_slot = TrainerConfig.__dict__["deterministic_algorithms"]
    descriptor_calls: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    class EffectfulPolicySlot:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            descriptor_calls.append(True)
            random.random()
            np.random.random()
            torch.rand(1)
            return original_slot.__get__(instance, owner)

    monkeypatch.setattr(
        TrainerConfig,
        "deterministic_algorithms",
        EffectfulPolicySlot(),
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="config fields must remain inert slots",
    ):
        trainer_adapter._assert_live_d02_determinism(target)

    assert descriptor_calls == []
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "missing_field",
    [
        "_canonical_default_schedule",
        "_canonical_unscheduled_default_optimizer",
        "_canonical_default_optimizer_options",
    ],
)
def test_checkpoint_load_requires_complete_native_restore_policy_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    missing_field: str,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    del vars(target)[missing_field]
    checkpoint_reads: list[bool] = []

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("incomplete native policy reached checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="missing restore policy fields",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=target.model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert checkpoint_reads == []
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "missing_field",
    ["model", "optimizer", "scheduler", "scaler", "config", "device"],
)
def test_checkpoint_load_requires_complete_native_restore_binding_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    missing_field: str,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    owned_model = target.model
    del vars(target)[missing_field]
    checkpoint_reads: list[bool] = []

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("incomplete native bindings reached checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="missing restore binding fields",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=owned_model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert checkpoint_reads == []
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "missing_marker",
    ["_failure_reason", "_update_incomplete"],
)
def test_native_trainer_missing_recovery_marker_cannot_downgrade_to_generic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    missing_marker: str,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    owned_model = target.model
    del vars(target)[missing_marker]
    checkpoint_reads: list[bool] = []

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("marker-deficient native trainer reached checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    assert trainer_adapter._is_native_d02(target)
    assert not trainer_adapter._is_canonical_d02(target)
    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="recovery markers are unavailable",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=owned_model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert checkpoint_reads == []



@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("weight", "model changed during checkpoint publication"),
        ("counter", "post-load tokens_seen disagrees with checkpoint"),
        ("mode", "requires model training mode"),
    ],
)
def test_native_checkpoint_save_rejects_model_export_drift_before_publication(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    class EffectfulModel(_TinyLogits):
        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.owner: Trainer | None = None

        def state_dict(self, *args: Any, **kwargs: Any) -> Any:
            state = super().state_dict(*args, **kwargs)
            if not self.armed:
                return state
            if mutation == "weight":
                with torch.no_grad():
                    self.weight.add_(1.0)
            elif mutation == "counter":
                assert self.owner is not None
                self.owner.tokens_seen += 1
            else:
                self.eval()
            return state

    class ArmAfterTrainerExport(Trainer):
        def state_dict(self) -> Any:
            state = super().state_dict()
            self.model.armed = True
            return state

    model = EffectfulModel()
    target = ArmAfterTrainerExport(
        model,
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    model.owner = target
    weight_before = model.weight.detach().clone()
    checkpoint = tmp_path / f"prepublish-{mutation}-must-not-exist"

    with pytest.raises(core.CheckpointCompatibilityError, match=expected):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=target,
            identity=_fresh_identity(),
        )

    assert not checkpoint.exists()
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True
    if mutation == "weight":
        assert not torch.equal(model.weight.detach(), weight_before)
    elif mutation == "counter":
        assert target.tokens_seen == 1
    else:
        assert model.training is False



@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("step", 1),
        ("tokens_seen", 2),
        ("step", True),
        ("tokens_seen", False),
    ],
)
def test_native_checkpoint_save_rejects_progress_identity_mismatch_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    identity = replace(_fresh_identity(), **{field: bad_value})
    save_calls: list[bool] = []

    def forbid_save(*args: Any, **kwargs: Any) -> Any:
        save_calls.append(True)
        raise AssertionError("mismatched progress identity reached checkpoint I/O")

    monkeypatch.setattr(trainer_adapter, "save_checkpoint", forbid_save)

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="checkpoint save progress identity mismatch",
    ):
        trainer_adapter.save_trainer_checkpoint(
            tmp_path / f"bad-{field}",
            model=target.model,
            trainer=target,
            identity=identity,
        )

    assert save_calls == []
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False



@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("seed", 920),
        ("precision", "bf16"),
    ],
)
def test_native_checkpoint_save_rejects_config_identity_mismatch_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    identity = replace(_fresh_identity(), **{field: bad_value})
    save_calls: list[bool] = []

    def forbid_save(*args: Any, **kwargs: Any) -> Any:
        save_calls.append(True)
        raise AssertionError("mismatched config identity reached checkpoint I/O")

    monkeypatch.setattr(trainer_adapter, "save_checkpoint", forbid_save)

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="checkpoint native config identity mismatch",
    ):
        trainer_adapter.save_trainer_checkpoint(
            tmp_path / f"bad-config-{field}",
            model=target.model,
            trainer=target,
            identity=identity,
        )

    assert save_calls == []
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("seed", 920),
        ("precision", "bf16"),
    ],
)
def test_native_checkpoint_load_rejects_internal_config_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    restore_rng: bool,
    field: str,
    bad_value: Any,
) -> None:
    source = _source()
    checkpoint = tmp_path / f"bad-{field}-identity"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=replace(_identity(), **{field: bad_value}),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weight = target.model.weight.detach().clone()
    model_bind_calls: list[bool] = []
    original_bind = loader._bind_model_state_loader

    def track_model_bind(model: Any, strict: bool) -> Any:
        model_bind_calls.append(True)
        return original_bind(model, strict)

    monkeypatch.setattr(loader, "_bind_model_state_loader", track_model_bind)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="checkpoint native config identity mismatch",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            restore_rng=restore_rng,
            **extra,
        )

    assert model_bind_calls == []
    torch.testing.assert_close(target.model.weight, initial_weight, rtol=0, atol=0)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False



@pytest.mark.parametrize(
    "mutation",
    ["counter", "mode"],
)
def test_native_checkpoint_final_model_fingerprint_avoids_named_parameter_hook(
    tmp_path: Path,
    mutation: str,
) -> None:
    active_hook_calls: list[str] = []

    class FinalFingerprintEffectModel(_TinyLogits):
        def __init__(self) -> None:
            super().__init__()
            self.owner: Trainer | None = None
            self.arm_final_fingerprint = False

        def state_dict(self, *args: Any, **kwargs: Any) -> Any:
            state = super().state_dict(*args, **kwargs)
            self.arm_final_fingerprint = True
            return state

        def named_parameters(self, *args: Any, **kwargs: Any):
            if self.arm_final_fingerprint:
                active_hook_calls.append(mutation)
                assert self.owner is not None
                if mutation == "counter":
                    self.owner.tokens_seen += 1
                else:
                    self.eval()
            yield from super().named_parameters(*args, **kwargs)

    model = FinalFingerprintEffectModel()
    target = Trainer(
        model,
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    model.owner = target
    checkpoint = tmp_path / f"hook-free-final-fingerprint-{mutation}"

    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=target,
        identity=_fresh_identity(),
    )

    assert checkpoint.exists()
    assert active_hook_calls == []
    assert target.tokens_seen == 0
    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert model.training is True


def test_native_checkpoint_save_rejects_subclass_export_model_mutation(
    tmp_path: Path,
) -> None:
    class MutatingExporter(Trainer):
        def state_dict(self) -> Any:
            state = super().state_dict()
            with torch.no_grad():
                self.model.weight.add_(0.25)
            return state

    target = MutatingExporter(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    checkpoint = tmp_path / "subclass-export-model-drift-must-not-exist"

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="model changed during checkpoint export",
    ):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            identity=_fresh_identity(),
        )

    assert not checkpoint.exists()
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True


@pytest.mark.parametrize(
    "mutation",
    ["optimizer", "scheduler"],
)
def test_native_checkpoint_save_rejects_subclass_export_auxiliary_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    class MutatingExporter(Trainer):
        def state_dict(self) -> Any:
            state = super().state_dict()
            if mutation == "optimizer":
                live_slot = next(iter(self.optimizer.state.values()))
                live_slot["exp_avg"].add_(0.25)
            else:
                assert self.scheduler is not None
                self.scheduler.last_epoch += 1
            return state

    target = MutatingExporter(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    assert target.train_microbatch(_BATCH).optimizer_stepped
    checkpoint = tmp_path / f"subclass-export-{mutation}-drift-must-not-exist"

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="auxiliary state changed during checkpoint export",
    ):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            identity=_identity(),
        )

    assert not checkpoint.exists()
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True


@pytest.mark.parametrize(
    "authority",
    [
        "_model_export_fingerprint",
        "_checkpoint_auxiliary_fingerprint",
        "_require_exported_model_matches_live",
        "_require_exported_scheduler_matches_live",
        "_require_exported_scaler_matches_live",
        "_require_exported_optimizer_matches_live",
        "_exact_export_leaf_equal",
        "assert_checkpoint_safe",
        "_assert_trainable",
        "_require_finite_auxiliary_state",
        "_require_finite_committed_update",
        "_require_no_residual_model_gradients",
        "_require_deterministic_policy",
        "_require_optimizer_parameter_coverage",
        "_require_safe_optimizer_hyperparameters",
        "_require_default_optimizer_options",
        "_require_constant_default_rate",
        "_require_finite_state_tree",
        "_require_checkpoint_scaler_state",
        "_require_default_schedule_rates",
        "_require_checkpoint_scheduler_chronology",
        "_require_optimizer_state_parameter_order",
        "_optimizer_parameter_name_groups",
        "_mark_failed",
    ],
)
def test_native_checkpoint_save_rejects_subclass_safety_authority_override(
    tmp_path: Path,
    authority: str,
) -> None:
    class UnsafeAuthorityTrainer(Trainer):
        pass

    if authority == "_exact_export_leaf_equal":
        setattr(
            UnsafeAuthorityTrainer,
            authority,
            staticmethod(lambda saved, live: True),
        )
    elif authority in {
        "_model_export_fingerprint",
        "_checkpoint_auxiliary_fingerprint",
    }:
        setattr(UnsafeAuthorityTrainer, authority, lambda self: "0" * 64)
    else:
        setattr(UnsafeAuthorityTrainer, authority, lambda self, exported: None)

    target = UnsafeAuthorityTrainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    checkpoint = tmp_path / f"unsafe-authority-{authority}"

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="safety authority must remain canonical",
    ):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            identity=_fresh_identity(),
        )

    assert not checkpoint.exists()
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "forgery",
    ["value", "missing", "extra"],
)
def test_native_checkpoint_save_rejects_forged_model_state_dict(
    tmp_path: Path,
    forgery: str,
) -> None:
    class ForgedStateModel(_TinyLogits):
        def state_dict(self, *args: Any, **kwargs: Any) -> Any:
            state = super().state_dict(*args, **kwargs)
            if forgery == "value":
                state["weight"] = state["weight"].detach().clone() + 0.25
            elif forgery == "missing":
                del state["weight"]
            else:
                state["forged_extra"] = torch.zeros(1)
            return state

    model = ForgedStateModel()
    target = Trainer(
        model,
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    checkpoint = tmp_path / f"forged-model-export-{forgery}"

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="staged model export differs from live model state",
    ):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=target,
            identity=_fresh_identity(),
        )

    assert not checkpoint.exists()
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True


def test_native_checkpoint_save_rechecks_safety_after_temporary_subclass_bypass(
    tmp_path: Path,
) -> None:
    class TemporarySafetyBypassTrainer(Trainer):
        def state_dict(self) -> Any:
            vars(self)["_require_finite_committed_update"] = lambda: None
            try:
                return super().state_dict()
            finally:
                del vars(self)["_require_finite_committed_update"]

    target = TemporarySafetyBypassTrainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    with torch.no_grad():
        target.model.weight.fill_(float("nan"))
    checkpoint = tmp_path / "temporary-safety-bypass-must-not-exist"

    with pytest.raises(Exception, match="non-finite model weights"):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            identity=_fresh_identity(),
        )

    assert not checkpoint.exists()
    assert "_require_finite_committed_update" not in vars(target)
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True


@pytest.mark.parametrize(
    "mutation",
    ["optimizer", "scheduler"],
)
def test_native_checkpoint_save_rejects_auxiliary_drift_from_model_export(
    tmp_path: Path,
    mutation: str,
) -> None:
    class AuxiliaryDriftModel(_TinyLogits):
        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.owner: Trainer | None = None

        def state_dict(self, *args: Any, **kwargs: Any) -> Any:
            state = super().state_dict(*args, **kwargs)
            if not self.armed:
                return state
            assert self.owner is not None
            if mutation == "optimizer":
                self.owner.optimizer.param_groups[0]["lr"] *= 0.5
            else:
                assert self.owner.scheduler is not None
                self.owner.scheduler.last_epoch += 1
            return state

    class ArmAfterTrainerExport(Trainer):
        def state_dict(self) -> Any:
            state = super().state_dict()
            self.model.armed = True
            return state

    model = AuxiliaryDriftModel()
    target = ArmAfterTrainerExport(
        model,
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    model.owner = target
    original_lr = target.optimizer.param_groups[0]["lr"]
    original_epoch = target.scheduler.last_epoch if target.scheduler is not None else None
    checkpoint = tmp_path / f"auxiliary-{mutation}-must-not-exist"

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="auxiliary state changed during checkpoint publication",
    ):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=target,
            identity=_fresh_identity(),
        )

    assert not checkpoint.exists()
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True
    if mutation == "optimizer":
        assert target.optimizer.param_groups[0]["lr"] == original_lr * 0.5
    else:
        assert target.scheduler is not None
        assert target.scheduler.last_epoch == original_epoch + 1



@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "authority",
    [
        "_model_export_fingerprint",
        "_checkpoint_auxiliary_fingerprint",
        "_require_exported_model_matches_live",
        "_require_exported_scheduler_matches_live",
        "_require_exported_scaler_matches_live",
        "_require_exported_optimizer_matches_live",
        "_exact_export_leaf_equal",
        "assert_checkpoint_safe",
        "_assert_trainable",
        "_require_finite_auxiliary_state",
        "_require_finite_committed_update",
        "_require_no_residual_model_gradients",
        "_require_deterministic_policy",
        "_require_optimizer_parameter_coverage",
        "_require_safe_optimizer_hyperparameters",
        "_require_default_optimizer_options",
        "_require_constant_default_rate",
        "_require_finite_state_tree",
        "_require_checkpoint_scaler_state",
        "_require_default_schedule_rates",
        "_require_checkpoint_scheduler_chronology",
        "_require_optimizer_state_parameter_order",
        "_optimizer_parameter_name_groups",
        "_mark_failed",
    ],
)
def test_native_checkpoint_load_rejects_subclass_safety_authority_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    authority: str,
) -> None:
    class UnsafeAuthorityTrainer(Trainer):
        pass

    if authority == "_exact_export_leaf_equal":
        setattr(
            UnsafeAuthorityTrainer,
            authority,
            staticmethod(lambda saved, live: True),
        )
    elif authority in {
        "_model_export_fingerprint",
        "_checkpoint_auxiliary_fingerprint",
    }:
        setattr(UnsafeAuthorityTrainer, authority, lambda self: "0" * 64)
    else:
        setattr(UnsafeAuthorityTrainer, authority, lambda self, exported: None)

    target = UnsafeAuthorityTrainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    checkpoint_reads: list[bool] = []

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("unsafe native authority reached checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="safety authority must remain canonical",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=target.model,
            trainer=target,
            restore_rng=False,
        )

    assert checkpoint_reads == []
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_native_checkpoint_prepublish_closes_after_auxiliary_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ArmAfterExport(Trainer):
        observer_armed = False

        def state_dict(self) -> Any:
            state = super().state_dict()
            self.observer_armed = True
            return state

    target = ArmAfterExport(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert target.scheduler is not None
    original_epoch = target.scheduler.last_epoch
    original_equal = torch.equal

    def equal_then_mutate(left: Any, right: Any) -> bool:
        result = original_equal(left, right)
        if target.observer_armed:
            target.observer_armed = False
            assert target.scheduler is not None
            target.scheduler.last_epoch += 1
        return result

    monkeypatch.setattr(torch, "equal", equal_then_mutate)
    checkpoint = tmp_path / "auxiliary-observer-drift-must-not-exist"

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="auxiliary state changed during checkpoint publication",
    ):
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            identity=_identity(),
        )

    assert not checkpoint.exists()
    assert target.scheduler.last_epoch == original_epoch + 1
    assert target._failure_reason == "checkpoint_export_state_drift"
    assert target._update_incomplete is True


def test_generic_preflight_does_not_dispatch_native_only_authorities() -> None:
    descriptor_calls: list[str] = []

    class EffectfulUnusedAuthority:
        def __init__(self, name: str) -> None:
            self.name = name

        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            descriptor_calls.append(self.name)
            raise AssertionError(f"unused native authority dispatched: {self.name}")

    class GenericTrainer:
        _require_safe_optimizer_hyperparameters = EffectfulUnusedAuthority("optimizer")
        _require_checkpoint_scheduler_chronology = EffectfulUnusedAuthority("scheduler")

        def __init__(self) -> None:
            self.model = _TinyLogits()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
            self.scheduler = None
            self.scaler = None
            self.config = {
                "gradient_accumulation_steps": 1,
                "max_steps": 3,
            }

        def load_state_dict(self, state: Any) -> None:
            del state

    target = GenericTrainer()
    state = {
        "micro_step": 0,
        "optimizer_step": 0,
        "tokens_seen": 0,
        "optimizer": target.optimizer.state_dict(),
        "scheduler": None,
        "scaler": None,
        "config": dict(target.config),
    }

    trainer_adapter._preflight_trainer_state(
        target,
        state,
        manifest=None,
    )

    assert descriptor_calls == []
