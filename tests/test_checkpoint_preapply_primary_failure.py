"""Primary checkpoint failures must survive preapply process-state rollback faults."""

from __future__ import annotations

from typing import Any

import pytest

from twelve_six.checkpoint import progress_trainer, trainer_adapter


def _raise(exc: BaseException) -> None:
    raise exc


def test_preapply_rollback_failure_preserves_existing_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("primary checkpoint interruption")
    rollback = RuntimeError("forced preapply process-state rollback failure")

    monkeypatch.setattr(
        trainer_adapter._core,
        "restore_rng_state",
        lambda state: _raise(rollback),
    )

    trainer_adapter._restore_preapply_process_state(
        {},
        None,
        object(),
        primary_exc=primary,
        expected_canonical=False,
    )

    assert any(
        "pre-application process-state restoration also failed" in note
        for note in getattr(primary, "__notes__", ())
    )


def test_preapply_rollback_failure_without_primary_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollback = RuntimeError("forced preapply process-state rollback failure")

    monkeypatch.setattr(
        trainer_adapter._core,
        "restore_rng_state",
        lambda state: _raise(rollback),
    )

    with pytest.raises(RuntimeError) as caught:
        trainer_adapter._restore_preapply_process_state(
            {},
            None,
            object(),
            expected_canonical=False,
        )

    assert caught.value is rollback


def test_trainer_adapter_loader_passes_primary_into_prebind_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("primary trainer adapter bind failure")
    observed: list[BaseException | None] = []

    monkeypatch.setattr(
        trainer_adapter,
        "_snapshot_trainer_restore_bindings",
        lambda trainer: (False, {}),
    )
    monkeypatch.setattr(trainer_adapter, "capture_rng_state", dict)
    monkeypatch.setattr(
        trainer_adapter,
        "_snapshot_torch_policy",
        lambda state: None,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_snapshot_torch_execution_mode",
        lambda: (True, False),
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_bind_trainer_state_loader",
        lambda trainer: _raise(primary),
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_note_restore_binding_drift",
        lambda trainer, snapshot, exc: None,
    )

    def observe_finalizer(
        ambient: Any,
        policy: Any,
        trainer: Any,
        *,
        primary_exc: BaseException | None = None,
        **kwargs: Any,
    ) -> None:
        del ambient, policy, trainer, kwargs
        observed.append(primary_exc)

    monkeypatch.setattr(
        trainer_adapter,
        "_restore_preapply_process_state",
        observe_finalizer,
    )

    with pytest.raises(RuntimeError) as caught:
        trainer_adapter.load_trainer_checkpoint(
            "unused-checkpoint",
            model=object(),
            trainer=object(),
        )

    assert caught.value is primary
    assert observed == [primary]


def test_progress_loader_passes_primary_into_prebind_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("primary progress bind failure")
    observed: list[BaseException | None] = []

    monkeypatch.setattr(
        progress_trainer,
        "_snapshot_trainer_restore_bindings",
        lambda trainer: (False, {}),
    )
    monkeypatch.setattr(progress_trainer._core, "capture_rng_state", dict)
    monkeypatch.setattr(
        progress_trainer,
        "_snapshot_torch_policy",
        lambda state: None,
    )
    monkeypatch.setattr(
        progress_trainer,
        "_snapshot_torch_execution_mode",
        lambda: (True, False),
    )
    monkeypatch.setattr(
        progress_trainer,
        "_bind_trainer_state_loader",
        lambda trainer: _raise(primary),
    )
    monkeypatch.setattr(
        progress_trainer,
        "_note_restore_binding_drift",
        lambda trainer, snapshot, exc: None,
    )

    def observe_finalizer(
        ambient: Any,
        policy: Any,
        trainer: Any,
        *,
        primary_exc: BaseException | None = None,
        **kwargs: Any,
    ) -> None:
        del ambient, policy, trainer, kwargs
        observed.append(primary_exc)

    monkeypatch.setattr(
        progress_trainer,
        "_restore_preapply_process_state",
        observe_finalizer,
    )

    with pytest.raises(RuntimeError) as caught:
        progress_trainer.load_trainer_checkpoint(
            "unused-checkpoint",
            model=object(),
            trainer=object(),
        )

    assert caught.value is primary
    assert observed == [primary]


def test_save_export_passes_primary_into_process_state_finalizer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("primary checkpoint export failure")
    observed: list[BaseException | None] = []

    monkeypatch.setattr(
        trainer_adapter,
        "_snapshot_trainer_restore_bindings",
        lambda trainer: (True, {}),
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_bind_trainer_state_exporter",
        lambda trainer: dict,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_bind_native_model_export_validator",
        lambda trainer: None,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_bind_native_model_export_fingerprint",
        lambda trainer: lambda: _raise(primary),
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_bind_native_auxiliary_fingerprint",
        lambda trainer: None,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_bind_native_export_live_authorities",
        lambda trainer: (),
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_assert_trainer_model_binding",
        lambda model, trainer: None,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_assert_native_d02_model_training_mode",
        lambda model, trainer: None,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_assert_live_d02_determinism",
        lambda trainer: None,
    )
    monkeypatch.setattr(trainer_adapter, "capture_rng_state", dict)
    monkeypatch.setattr(
        trainer_adapter,
        "_snapshot_torch_policy",
        lambda state: None,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_snapshot_torch_execution_mode",
        lambda: (True, False),
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_note_restore_binding_drift",
        lambda trainer, snapshot, exc: None,
    )

    def observe_finalizer(
        ambient: Any,
        policy: Any,
        trainer: Any,
        *,
        primary_exc: BaseException | None = None,
        **kwargs: Any,
    ) -> None:
        del ambient, policy, trainer, kwargs
        observed.append(primary_exc)

    monkeypatch.setattr(
        trainer_adapter,
        "_restore_preapply_process_state",
        observe_finalizer,
    )

    with pytest.raises(RuntimeError) as caught:
        trainer_adapter.save_trainer_checkpoint(
            tmp_path / "checkpoint",
            model=object(),
            trainer=object(),
            identity=object(),  # type: ignore[arg-type]
        )

    assert caught.value is primary
    assert observed == [primary]
