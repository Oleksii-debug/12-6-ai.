from __future__ import annotations

from pathlib import Path

import conftest
import pytest
from test_bounded_pilot import _authority, _batch, _gate, _next, _trainer

from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
    BoundedPilotRecoveryRequiredError,
    _set_test_recovery_store_factory,
)
from twelve_six.training.resilience import RecoveryPolicy, RecoveryStore
from twelve_six.training.single_gpu import SingleDeviceStepRunner


def test_fresh_process_same_run_cannot_replay_ambiguous_committed_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "durable-run"
    saved_manifest: dict | None = None

    def shared_store_factory(gate):
        nonlocal saved_manifest
        if saved_manifest is None:
            saved_manifest = {
                "schema_version": "12-6.pytest-bounded-pilot-run-manifest.v1",
                "run_id": "fresh-process-replay-regression",
                "candidate": {"git_sha": "2" * 40},
                "recovery": {
                    "topology": {
                        "backend": "single-process-cpu",
                        "world_size": 1,
                        "rank_count": 1,
                        "resume_policy": "exact_topology",
                    }
                },
                "bounded_pilot": gate._bounded_start_projection(),
            }
        assert saved_manifest["bounded_pilot"] == gate._bounded_start_projection()
        store = RecoveryStore(
            root,
            run_manifest=saved_manifest,
            policy=RecoveryPolicy(
                checkpoint_every_steps=1,
                retain_last=2,
                max_restarts=0,
                max_preemptions=0,
            ),
        )
        return store, store.run_manifest_sha256, store.run_id

    previous = _set_test_recovery_store_factory(shared_store_factory)
    try:
        guard, plan, manifest = _authority(batch_count=1)
        trainer = _trainer(max_steps=1)
        runner = SingleDeviceStepRunner(trainer)
        synchronize_calls = 0

        def fail_after_trainer_commit() -> None:
            nonlocal synchronize_calls
            synchronize_calls += 1
            if synchronize_calls == 3:
                raise RuntimeError("synthetic post-step synchronization failure")

        runner._synchronize = fail_after_trainer_commit  # type: ignore[method-assign]
        gate = _gate(trainer, guard, plan, manifest, runner=runner)
        with pytest.raises(BoundedPilotRecoveryRequiredError):
            gate.train_authorized_microbatch(
                _batch(0),
                batch_index=0,
                expected_next_exposure_identity_sha256=_next(guard, plan, 0),
            )
        assert trainer.optimizer_step == 1
        assert guard.consumed_loss_positions == 2
        gate.close()

        # Simulate a new Python process: rebuild fresh model + stale zero D04 state
        # against the exact same authenticated run manifest and packet projection.
        fresh_guard, fresh_plan, fresh_manifest = _authority(batch_count=1)
        fresh_trainer = _trainer(max_steps=1)
        fresh_gate = _gate(fresh_trainer, fresh_guard, fresh_plan, fresh_manifest)
        before = [parameter.detach().clone() for parameter in fresh_trainer.model.parameters()]
        with pytest.raises(
            BoundedPilotRecoveryRequiredError,
            match="already has durable attempt/history",
        ):
            fresh_gate.train_authorized_microbatch(
                _batch(0),
                batch_index=0,
                expected_next_exposure_identity_sha256=_next(
                    fresh_guard,
                    fresh_plan,
                    0,
                ),
            )
        fresh_gate.close()
        assert fresh_trainer.optimizer_step == 0
        assert fresh_guard.consumed_loss_positions == 0
        assert all(
            left.equal(right)
            for left, right in zip(before, fresh_trainer.model.parameters(), strict=True)
        )
    finally:
        _set_test_recovery_store_factory(
            previous or conftest.bounded_pilot_test_recovery_store_factory
        )


def test_durable_receipt_binds_run_and_attempt() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate = _gate(trainer, guard, plan, manifest)
    _metrics, receipt = gate.train_authorized_microbatch(
        _batch(0),
        batch_index=0,
        expected_next_exposure_identity_sha256=_next(guard, plan, 0),
    )
    gate.close()
    assert receipt.run_id
    assert receipt.attempt == 1
    assert len(receipt.run_manifest_sha256) == 64
    assert len(receipt.attempt_state_sha256) == 64


def test_production_constructor_rejects_missing_durable_authority() -> None:
    previous = _set_test_recovery_store_factory(None)
    try:
        guard, plan, manifest = _authority(batch_count=1)
        trainer = _trainer(max_steps=1)
        with pytest.raises(
            BoundedPilotAuthorizationError,
            match="durable TRAIN39 run/attempt authority is required",
        ):
            _gate(trainer, guard, plan, manifest)
    finally:
        _set_test_recovery_store_factory(
            previous or conftest.bounded_pilot_test_recovery_store_factory
        )
