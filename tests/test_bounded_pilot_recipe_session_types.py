from __future__ import annotations

import copy
from pathlib import Path

import pytest
from test_bounded_pilot import _authority, _batch, _next
from test_bounded_pilot_durable_attempt import _constructor_inputs, _gate, _trainer

from twelve_six.learned20m_recipe import identity_sha256
from twelve_six.portable_run_binding import PortableRunBinding, canonical_sha256
from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
    BoundedPilotRecoveryRequiredError,
    BoundedPilotStepRunner,
)
from twelve_six.training.resilience import FailureClass, RecoveryPolicy, RecoveryStore


@pytest.mark.parametrize(
    "field",
    ["authorized_optimized_targets", "optimizer_updates_executed"],
)
def test_rehashed_float_zero_session_truth_is_rejected_type_strict(
    tmp_path: Path,
    field: str,
) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    runner, kwargs, original_store = _constructor_inputs(
        tmp_path / "original",
        trainer,
        guard,
        plan,
        manifest,
    )

    session = copy.deepcopy(kwargs["training_recipe_session"])
    session[field] = 0.0
    session_core = {
        key: value for key, value in session.items() if key != "session_identity_sha256"
    }
    session_root = identity_sha256(session_core)
    session["session_identity_sha256"] = session_root

    original_binding = kwargs["binding"]
    assert original_binding.packet is not None
    packet = copy.deepcopy(original_binding.packet)
    packet["recipe"]["training_config_sha256"] = session_root
    packet_root = canonical_sha256(packet)
    binding = PortableRunBinding(
        binding_ready=original_binding.binding_ready,
        mode=original_binding.mode,
        readiness_ready=original_binding.readiness_ready,
        overlay_contract_valid=original_binding.overlay_contract_valid,
        packet_contract_valid=original_binding.packet_contract_valid,
        blockers=original_binding.blockers,
        readiness_sha256=original_binding.readiness_sha256,
        overlay_sha256=original_binding.overlay_sha256,
        packet_sha256=packet_root,
        packet=packet,
    )

    run_manifest = copy.deepcopy(original_store.run_manifest)
    run_manifest["bounded_pilot"]["packet_sha256"] = packet_root
    run_manifest["bounded_pilot"]["training_config_sha256"] = session_root
    store = RecoveryStore(
        tmp_path / f"float-zero-{field}",
        run_manifest=run_manifest,
        policy=RecoveryPolicy(
            checkpoint_every_steps=1,
            retain_last=2,
            max_restarts=0,
            max_preemptions=0,
        ),
    )

    kwargs.update(
        {
            "binding": binding,
            "expected_packet_sha256": packet_root,
            "training_recipe_session": session,
            "expected_training_config_sha256": session_root,
            "recovery_store": store,
            "expected_run_manifest_sha256": store.run_manifest_sha256,
            "expected_run_id": store.run_id,
        }
    )

    with pytest.raises(
        BoundedPilotAuthorizationError,
        match="must remain exact integer zero",
    ):
        BoundedPilotStepRunner(runner, **kwargs)
    assert trainer.optimizer_step == 0
    assert guard.consumed_loss_positions == 0
    assert store.open()["attempt"] == 0


def test_durable_attempt_change_after_first_handoff_blocks_optimizer_hook(
    tmp_path: Path,
) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate, store = _gate(
        tmp_path / "attempt-tocou",
        trainer,
        guard,
        plan,
        manifest,
    )
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    original_preflight = gate._preflight_handoff
    calls = 0

    def mutate_after_first_handoff(**kwargs):
        nonlocal calls
        calls += 1
        result = original_preflight(**kwargs)
        if calls == 1:
            store.record_failure(
                FailureClass.PROCESS_LOSS,
                optimizer_step=trainer.optimizer_step,
                detail_code="injected-between-handoff-and-optimizer-hook",
            )
        return result

    gate._preflight_handoff = mutate_after_first_handoff  # type: ignore[method-assign]
    with pytest.raises(BoundedPilotRecoveryRequiredError):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    gate.close()

    assert calls == 2
    assert trainer.optimizer_step == 0
    assert trainer.micro_step == 1
    assert trainer.tokens_seen == 2
    assert guard.consumed_loss_positions == 0
    assert all(
        left.equal(right)
        for left, right in zip(before, trainer.model.parameters(), strict=True)
    )
    state = store.open()
    assert state["attempt"] == 1
    assert state["phase"] in {"RECOVERING", "FAILED"}
