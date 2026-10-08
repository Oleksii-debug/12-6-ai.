"""Plan 5 S6: deterministic scheduling, TaskStore integration and fail-closed recovery."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from twelve_six_agent_runtime.scheduler import (
    SchedulerError, SchedulerStore, WorkSpec,
)
from twelve_six_agent_runtime.task_state import PendingEffect, StateError, TaskStore


def spec(name="task-a", **kw):
    args = dict(
        task_id=name, plan_id="plan-5", goal_id="goal-a",
        priority=50, deadline_tick=100, max_steps=5, cpu_units=1,
        memory_mb=64, background=False,
    )
    args.update(kw)
    return WorkSpec(**args)


def scheduler(tmp_path):
    taskstore = TaskStore(tmp_path / "tasks.sqlite")
    return SchedulerStore(tmp_path / "queue.sqlite", taskstore)


def acquire(store, **kw):
    args = dict(tick=20, available_cpu=2, available_memory_mb=128)
    args.update(kw)
    return store.acquire(**args)


def test_priority_deadline_and_resources_deterministic(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec("b", priority=80, deadline_tick=90))
    s.register(spec("a", priority=80, deadline_tick=30))
    s.register(spec("c", priority=100, cpu_units=9))
    lease = acquire(s)
    assert lease.task_id == "a"
    assert acquire(s) is None  # one active lease, no duplicate launch
    assert s.complete(lease, used_steps=2).status == "done"
    assert acquire(s).task_id == "b"


def test_resource_pressure_throttle_pause_and_recover(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec("foreground", priority=30))
    s.register(spec("background", priority=90, background=True))
    lease = acquire(s, pressure_ppm=600_000)
    assert lease.task_id == "foreground"
    state = s.checkpoint(
        lease, used_steps=1, step_id="step1", checkpoint_id="cp1",
        pressure_ppm=900_000,
    )
    assert state.status == "paused" and state.pause_reason == "resource_pressure"
    assert acquire(s, pressure_ppm=950_000) is None
    with pytest.raises(SchedulerError, match="evidence"):
        s.reconcile("foreground", verifier=lambda _: False)
    ready = SchedulerStore(s.path, TaskStore(s.tasks.path))
    assert ready.snapshot() == s.snapshot()
    ready.reconcile("foreground", verifier=lambda st: st.step_id == "step1")
    assert acquire(ready, pressure_ppm=550_000).task_id == "foreground"


def test_checkpoint_restart_exact_epoch_and_budget_limit(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec(max_steps=2))
    token = acquire(s)
    saved = s.checkpoint(token, used_steps=2, step_id="s1", checkpoint_id="checkpoint1")
    assert saved.status == "paused" and saved.pause_reason == "budget_exhausted"
    assert s.tasks.load("task-a").checkpoint_id == "checkpoint1"
    restarted = SchedulerStore(s.path, TaskStore(s.tasks.path))
    assert restarted.snapshot() == (saved,)
    with pytest.raises(SchedulerError, match="budget"):
        restarted.reconcile("task-a", verifier=lambda _: True)
    assert acquire(restarted) is None
    with pytest.raises(SchedulerError, match="stale"):
        restarted.complete(token)


def test_stale_token_and_external_epoch_change_cannot_mutate(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec())
    token = acquire(s)
    external = s.tasks.resume("task-a")
    assert external.control_epoch > token.control_epoch
    with pytest.raises(SchedulerError, match="changed"):
        s.complete(token)
    assert s.snapshot()[0].status == "leased"
    with pytest.raises(SchedulerError, match="evidence"):
        s.reconcile("task-a", verifier=lambda _: False)
    assert s.reconcile("task-a", verifier=lambda _: True).status == "ready"
    assert acquire(s).control_epoch > external.control_epoch


def test_restart_leased_uncertain_requires_explicit_verification(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec())
    old = acquire(s)
    other = SchedulerStore(s.path, TaskStore(s.tasks.path))
    assert acquire(other) is None
    with pytest.raises(SchedulerError):
        other.reconcile("task-a", verifier=None)
    new = other.reconcile("task-a", verifier=lambda _: True)
    assert new.status == "ready"
    assert acquire(other).control_epoch > old.control_epoch
    with pytest.raises(SchedulerError):
        other.complete(old)


def test_unknown_effect_never_automatically_reissued(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec())
    token = acquire(s)
    nextstate = s.tasks.checkpoint(
        "task-a", expected_epoch=token.control_epoch,
        expected_revision=token.task_revision, step_id="tool",
        checkpoint_id="tool-cp", pending_effects=(PendingEffect("effect1", "remote"),),
    )
    issued = s.tasks.issue_effect(
        "task-a", "effect1", expected_epoch=token.control_epoch,
        expected_revision=nextstate.revision,
    )
    assert issued.pending_effects[0].status == "unknown"
    with pytest.raises(SchedulerError, match="unresolved"):
        s.reconcile("task-a", verifier=lambda _: True)
    with pytest.raises(SchedulerError):
        s.complete(token)
    final = s.tasks.resolve_effect(
        "task-a", "effect1", "receipt:external",
        expected_epoch=issued.control_epoch, expected_revision=issued.revision,
    )
    assert final.pending_effects[0].status == "resolved"
    assert s.reconcile("task-a", verifier=lambda _: True).status == "ready"
    assert acquire(s).task_id == "task-a"


def test_idempotent_taskstore_crash_gap_adoption_and_mismatch(tmp_path):
    s = scheduler(tmp_path)
    s.tasks.create(task_id="recovered", plan_id="plan-5", step_id="queued")
    assert s.register(spec("recovered")).status == "ready"
    with pytest.raises(SchedulerError, match="duplicate"):
        s.register(spec("recovered"))
    s.tasks.create(task_id="foreign", plan_id="unrelated", step_id="queued")
    with pytest.raises(SchedulerError, match="noninitial"):
        s.register(spec("foreign"))
    assert all(e.spec.task_id != "foreign" for e in s.snapshot())


def test_negative_adversarial_inputs_roll_back(tmp_path):
    s = scheduler(tmp_path)
    for bad in (
        spec(max_steps=0), spec(priority=101), spec(cpu_units=0),
        spec(memory_mb=True), spec(background=1), spec(deadline_tick=-1),
        spec(task_id=""),
    ):
        with pytest.raises((SchedulerError, StateError)):
            s.register(bad)
    assert s.snapshot() == ()
    s.register(spec())
    with pytest.raises((SchedulerError, StateError)):
        acquire(s, pressure_ppm=1_000_001)
    with pytest.raises((SchedulerError, StateError)):
        acquire(s, tick=True)
    assert s.snapshot()[0].status == "ready"


def test_concurrent_acquisition_only_one_lease(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec())
    def claim(_):
        return acquire(SchedulerStore(s.path, TaskStore(s.tasks.path)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, (1, 2)))
    assert sum(token is not None for token in results) == 1
    assert s.snapshot()[0].status == "leased"


def test_corrupt_queue_or_history_and_storage_version_fail_closed(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec())
    with sqlite3.connect(s.path) as db:
        db.execute("UPDATE queue SET payload='{}'")
    with pytest.raises(SchedulerError, match="digest"):
        s.snapshot()
    with sqlite3.connect(s.path) as db:
        db.execute("UPDATE queue SET payload=(SELECT payload FROM history LIMIT 1)")
        db.execute("DELETE FROM history")
    with pytest.raises(SchedulerError, match="history"):
        s.snapshot()
    with sqlite3.connect(s.path) as db:
        db.execute("PRAGMA user_version=20")
    with pytest.raises(SchedulerError, match="version"):
        SchedulerStore(s.path, TaskStore(s.tasks.path))


def test_deadline_and_pressure_cannot_be_self_authorized(tmp_path):
    s = scheduler(tmp_path)
    s.register(spec("expired", deadline_tick=19))
    s.register(spec("background", priority=90, background=True))
    assert acquire(s, tick=20, pressure_ppm=600_000) is None
    assert acquire(s, tick=20, pressure_ppm=0).task_id == "background"
