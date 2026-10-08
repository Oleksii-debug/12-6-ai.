from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from twelve_six_agent_runtime.scheduler import (
    DispatchLease,
    ResourcePolicy,
    ResourceReading,
    ScheduleSpec,
    SchedulerError,
    SchedulerStore,
)
from twelve_six_agent_runtime.task_state import PendingEffect, TaskStore


def make(tmp_path: Path, *, total: int = 4, parallel: int = 1):
    tasks = TaskStore(tmp_path / "tasks.db")
    policy = ResourcePolicy(parallel, 2, 512, total)
    sched = SchedulerStore(tmp_path / "schedule.db", tasks, policy)
    return tasks, sched, policy


def add(tasks, sched, name="a", *, priority=10, deadline=1000, cpu=1, mem=100,
        attempt_budget=3, kind="background"):
    tasks.create(task_id=name, plan_id="plan-5", step_id="s0")
    sched.register(ScheduleSpec(name, "goal", kind, priority, deadline, cpu, mem,
                                attempt_budget))


def reading(now=100, *, pressure="normal", cpu=2, mem=512):
    return ResourceReading(now, cpu, mem, pressure)


def trust(*args):
    return True


def test_deterministic_priority_deadline_admission_and_sealed_restart(tmp_path):
    tasks, scheduler, policy = make(tmp_path)
    add(tasks, scheduler, "later", priority=10, deadline=500)
    add(tasks, scheduler, "first", priority=20, deadline=300)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust).task_id == "first"
    recovered = SchedulerStore(scheduler.path, TaskStore(tasks.path), policy)
    assert recovered.offer(now=100, reading=reading(), verify_reading=trust) is None
    states = {item["spec"]["task_id"]: item["state"] for item in recovered.snapshot()}
    assert states == {"first": "running", "later": "ready"}


def test_repeated_or_stale_effect_lease_never_executes_again(tmp_path):
    tasks, scheduler, _ = make(tmp_path)
    add(tasks, scheduler)
    lease = scheduler.offer(now=100, reading=reading(), verify_reading=trust)
    assert isinstance(lease, DispatchLease)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None
    scheduler.complete(lease, outcome="done", evidence_id="receipt-1", verify_evidence=trust)
    with pytest.raises(SchedulerError, match="stale|consumed"):
        scheduler.complete(lease, outcome="done", evidence_id="receipt-2", verify_evidence=trust)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None


def test_checkpoint_resume_requires_evidence_and_respects_budget(tmp_path):
    tasks, scheduler, _ = make(tmp_path, total=2)
    add(tasks, scheduler)
    first = scheduler.offer(now=100, reading=reading(), verify_reading=trust)
    scheduler.complete(first, outcome="checkpoint", evidence_id="cp1", verify_evidence=trust)
    with pytest.raises(SchedulerError, match="checkpoint"):
        scheduler.retry_checkpoint("a", evidence_id="wrong", verifier=trust)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None
    scheduler.retry_checkpoint("a", evidence_id="cp1", verifier=trust)
    second = scheduler.offer(now=100, reading=reading(), verify_reading=trust)
    assert second.attempt == 2 and second.lease_id != first.lease_id
    scheduler.complete(second, outcome="checkpoint", evidence_id="cp2", verify_evidence=trust)
    scheduler.retry_checkpoint("a", evidence_id="cp2", verifier=trust)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None


def test_resource_throttle_pause_checkpoint_and_recovery(tmp_path):
    tasks, scheduler, _ = make(tmp_path)
    add(tasks, scheduler, "a")
    scheduler.pause_for_pressure(reading(pressure="high"), trust)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None
    with pytest.raises(SchedulerError, match="unverified"):
        scheduler.resume_admission(reading(), lambda _: False)
    scheduler.resume_admission(reading(), trust)
    lease = scheduler.offer(now=100, reading=reading(), verify_reading=trust)
    assert lease is not None
    assert scheduler.offer(now=100, reading=reading(pressure="high"),
                           verify_reading=trust) is None
    scheduler.complete(lease, outcome="checkpoint", evidence_id="pause-cp",
                       verify_evidence=trust)
    scheduler.retry_checkpoint("a", evidence_id="pause-cp", verifier=trust)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None
    scheduler.resume_admission(reading(), trust)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is not None


def test_unknown_effect_fails_closed_without_blind_retry(tmp_path):
    tasks, scheduler, _ = make(tmp_path)
    add(tasks, scheduler)
    lease = scheduler.offer(now=100, reading=reading(), verify_reading=trust)
    pending = (PendingEffect("effect", "side effect"),)
    snap = tasks.checkpoint("a", expected_epoch=0, expected_revision=0,
                            step_id="s1", checkpoint_id="c1", pending_effects=pending)
    tasks.issue_effect("a", "effect", expected_epoch=0, expected_revision=snap.revision)
    with pytest.raises(SchedulerError, match="reconciliation"):
        scheduler.complete(lease, outcome="done", evidence_id="fake", verify_evidence=trust)
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None
    snap = tasks.load("a")
    tasks.resolve_effect("a", "effect", "trusted-receipt", expected_epoch=0,
                         expected_revision=snap.revision)
    scheduler.complete(lease, outcome="done", evidence_id="receipt", verify_evidence=trust)


def test_host_resource_evidence_unknown_stale_and_oversubscription(tmp_path):
    tasks, scheduler, _ = make(tmp_path, parallel=2)
    add(tasks, scheduler, "a", cpu=2)
    add(tasks, scheduler, "b", cpu=1)
    with pytest.raises(SchedulerError, match="unverified"):
        scheduler.offer(now=100, reading=reading(), verify_reading=lambda _: False)
    with pytest.raises(SchedulerError, match="stale"):
        scheduler.offer(now=100, reading=reading(1), verify_reading=trust)
    assert scheduler.offer(now=100, reading=reading(cpu=0), verify_reading=trust) is None
    one = scheduler.offer(now=100, reading=reading(), verify_reading=trust)
    assert one.task_id == "a"
    assert scheduler.offer(now=100, reading=reading(), verify_reading=trust) is None
    assert tasks.load("b").control_epoch == 0


def test_deadline_rejection_and_priority_tie_break(tmp_path):
    tasks, sched, _ = make(tmp_path, parallel=2)
    add(tasks, sched, "expired", deadline=99, priority=100)
    add(tasks, sched, "b", deadline=500)
    add(tasks, sched, "a", deadline=500)
    assert sched.offer(now=100, reading=reading(), verify_reading=trust).task_id == "a"


def test_parallel_sqlite_claim_has_exactly_one_winner(tmp_path):
    tasks, sched, _ = make(tmp_path, parallel=1)
    add(tasks, sched)
    def attempt(_):
        return sched.offer(now=100, reading=reading(), verify_reading=trust)
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(attempt, range(2)))
    assert sum(lease is not None for lease in result) == 1
    assert sched.snapshot()[0]["attempt"] == 1


def test_storage_corruption_and_policy_drift_fail_closed(tmp_path):
    tasks, sched, policy = make(tmp_path)
    add(tasks, sched)
    with pytest.raises(SchedulerError, match="policy"):
        SchedulerStore(sched.path, tasks, ResourcePolicy(3, 2, 512, 4))
    with sqlite3.connect(sched.path) as db:
        db.execute("UPDATE jobs SET raw='{}' WHERE task_id='a'")
    with pytest.raises(SchedulerError, match="digest"):
        sched.snapshot()
    with pytest.raises(SchedulerError, match="digest"):
        sched.offer(now=100, reading=reading(), verify_reading=trust)


def test_adversarial_types_and_untrusted_verifiers(tmp_path):
    tasks, sched, _ = make(tmp_path)
    with pytest.raises((SchedulerError, ValueError)):
        ScheduleSpec("t", "g", "background", True, 100, 1, 1, 1).validate()
    with pytest.raises((SchedulerError, ValueError)):
        ResourcePolicy(True, 1, 1, 1).validate()
    add(tasks, sched)
    lease = sched.offer(now=100, reading=reading(), verify_reading=trust)
    with pytest.raises(SchedulerError, match="unverified"):
        sched.complete(lease, outcome="done", evidence_id="fake",
                       verify_evidence=lambda *_: 1)
    with pytest.raises(SchedulerError, match="epoch"):
        tasks.resume("a")
        sched.complete(lease, outcome="done", evidence_id="receipt", verify_evidence=trust)


def test_crash_resume_epoch_requires_attested_reconciliation(tmp_path):
    tasks, sched, policy = make(tmp_path)
    add(tasks, sched)
    lease = sched.offer(now=100, reading=reading(), verify_reading=trust)
    tasks.resume("a")
    recovered = SchedulerStore(sched.path, TaskStore(tasks.path), policy)
    assert recovered.offer(now=100, reading=reading(), verify_reading=trust) is None
    with pytest.raises(SchedulerError, match="verified"):
        recovered.reconcile_interrupted(lease, checkpoint_id="stop-proof",
                                        verifier=lambda *_: False)
    recovered.reconcile_interrupted(lease, checkpoint_id="stop-proof", verifier=trust)
    recovered.retry_checkpoint("a", evidence_id="stop-proof", verifier=trust)
    next_lease = recovered.offer(now=100, reading=reading(), verify_reading=trust)
    assert next_lease.control_epoch == 1 and next_lease.attempt == 2


def test_unresolved_effect_refuses_stranded_lease_reconciliation(tmp_path):
    tasks, sched, _ = make(tmp_path)
    add(tasks, sched)
    lease = sched.offer(now=100, reading=reading(), verify_reading=trust)
    snap = tasks.checkpoint(
        "a", expected_epoch=0, expected_revision=0, step_id="s1",
        checkpoint_id="c1", pending_effects=(PendingEffect("e1", "unsafe"),),
    )
    tasks.issue_effect("a", "e1", expected_epoch=0, expected_revision=snap.revision)
    tasks.resume("a")
    with pytest.raises(SchedulerError, match="incomplete"):
        sched.reconcile_interrupted(lease, checkpoint_id="receipt", verifier=trust)
    assert sched.snapshot()[0]["state"] == "running"
