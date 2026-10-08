"""Additional Plan 5 / Section 2 adversarial regression at the durable effect boundary."""
import sqlite3
import pytest

from twelve_six_agent_runtime.task_state import (
    PendingEffect,
    StaleEpoch,
    StaleRevision,
    StateError,
    TaskStore,
)


def _store(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite")
    return store, store.create(task_id="task:01", plan_id="plan:5", step_id="first")


def _unresolved(store, first):
    reserved = store.checkpoint(
        first.task_id, expected_epoch=first.control_epoch,
        expected_revision=first.revision, step_id="first", checkpoint_id="ck:1",
        pending_effects=(PendingEffect("effect:01", "external operation"),),
    )
    return store.issue_effect(
        first.task_id, "effect:01", expected_epoch=reserved.control_epoch,
        expected_revision=reserved.revision,
    )


@pytest.mark.parametrize(
    "replacement",
    [
        (),
        (PendingEffect("effect:01", "external operation", status="pending"),),
        (PendingEffect("effect:01", "external operation", status="resolved", receipt_id="forged"),),
    ],
)
def test_checkpoint_cannot_erase_or_reclassify_issued_effect(tmp_path, replacement):
    store, first = _store(tmp_path)
    unresolved = _unresolved(store, first)
    with pytest.raises(StateError):
        store.checkpoint(
            first.task_id, expected_epoch=unresolved.control_epoch,
            expected_revision=unresolved.revision,
            step_id="second", checkpoint_id="ck:2", pending_effects=replacement,
        )
    assert store.load(first.task_id) == unresolved


def test_restart_keeps_issued_effect_and_fences_stale_async_result(tmp_path):
    store, first = _store(tmp_path)
    issued = _unresolved(store, first)
    reopened = TaskStore(tmp_path / "tasks.sqlite")
    restart = reopened.resume(first.task_id)
    assert restart.control_epoch == issued.control_epoch + 1
    assert restart.step_id == issued.step_id
    assert restart.pending_effects == issued.pending_effects
    with pytest.raises(StaleEpoch):
        store.resolve_effect(
            first.task_id, "effect:01", "receipt:old",
            expected_epoch=issued.control_epoch, expected_revision=issued.revision,
        )
    with pytest.raises(StateError):
        reopened.issue_effect(
            first.task_id, "effect:01", expected_epoch=restart.control_epoch,
            expected_revision=restart.revision,
        )
    assert reopened.load(first.task_id) == restart


def test_revision_cas_rejects_stale_parallel_writer(tmp_path):
    store, first = _store(tmp_path)
    moved = store.checkpoint(
        first.task_id, expected_epoch=0, expected_revision=0,
        step_id="second", checkpoint_id="ck:1", pending_effects=(),
    )
    with pytest.raises(StaleRevision):
        TaskStore(tmp_path / "tasks.sqlite").checkpoint(
            first.task_id, expected_epoch=0, expected_revision=0,
            step_id="competing", checkpoint_id="ck:2", pending_effects=(),
        )
    assert store.load(first.task_id) == moved


def test_corrupt_stored_hash_is_not_silently_reinitialized(tmp_path):
    store, first = _store(tmp_path)
    db = sqlite3.connect(store.path)
    db.execute("UPDATE tasks SET digest=? WHERE task_id=?", ("0" * 64, first.task_id))
    db.commit()
    db.close()
    with pytest.raises(StateError, match="hash mismatch"):
        TaskStore(store.path).load(first.task_id)


def test_duplicate_task_identity_fails_closed(tmp_path):
    store, first = _store(tmp_path)
    with pytest.raises(StateError):
        store.create(task_id=first.task_id, plan_id="other", step_id="other")
    assert store.load(first.task_id) == first


def test_issued_effect_receipt_binds_only_exact_current_revision(tmp_path):
    store, first = _store(tmp_path)
    _unresolved(store, first)
    restarted = TaskStore(store.path).resume(first.task_id)
    resolved = store.resolve_effect(
        first.task_id, "effect:01", "receipt:verified",
        expected_epoch=restarted.control_epoch, expected_revision=restarted.revision,
    )
    assert resolved.pending_effects[0].receipt_id == "receipt:verified"
    with pytest.raises(StaleRevision):
        store.resolve_effect(
            first.task_id, "effect:01", "receipt:conflict",
            expected_epoch=restarted.control_epoch, expected_revision=restarted.revision,
        )
    assert store.load(first.task_id) == resolved
