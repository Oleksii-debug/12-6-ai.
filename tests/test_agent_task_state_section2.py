from __future__ import annotations

import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from twelve_six_agent_runtime.task_state import (
    PendingEffect,
    StaleEpoch,
    StaleRevision,
    StateError,
    TaskStore,
)


def test_restart_retains_exact_task_plan_step_checkpoint_effects_and_history(tmp_path: Path) -> None:
    db = tmp_path / "tasks.sqlite"
    store = TaskStore(db)
    initial = store.create(task_id="task-1", plan_id="plan-5", step_id="step-0")
    effect = PendingEffect("tool-call-1", "write a file")
    saved = store.checkpoint(
        "task-1", expected_epoch=0, expected_revision=0,
        step_id="step-1", checkpoint_id="cp-1", pending_effects=(effect,),
    )
    assert saved.revision == 1
    assert TaskStore(db).load("task-1") == saved
    assert TaskStore(db).load_revision("task-1", 0) == initial

    # A copy of a quiescent durable store represents a machine move / cold restart.
    migrated = tmp_path / "recovered.sqlite"
    shutil.copy2(db, migrated)
    resumed = TaskStore(migrated).resume("task-1")
    assert (resumed.task_id, resumed.plan_id, resumed.step_id, resumed.checkpoint_id) == (
        "task-1", "plan-5", "step-1", "cp-1",
    )
    assert resumed.pending_effects == (effect,)
    assert (resumed.revision, resumed.control_epoch) == (2, 1)
    assert TaskStore(migrated).load_revision("task-1", 1) == saved


def test_stale_async_epoch_and_stale_revision_cannot_change_new_state(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    store.create(task_id="task", plan_id="plan", step_id="s0")
    next_epoch = TaskStore(store.path).resume("task")
    with pytest.raises(StaleEpoch, match="stale"):
        store.checkpoint(
            "task", expected_epoch=0, expected_revision=0, step_id="bad",
            checkpoint_id="bad", pending_effects=(),
        )
    with pytest.raises(StaleRevision):
        store.checkpoint(
            "task", expected_epoch=1, expected_revision=0, step_id="bad",
            checkpoint_id="bad", pending_effects=(),
        )
    assert store.load("task") == next_epoch
    good = store.checkpoint(
        "task", expected_epoch=1, expected_revision=1, step_id="s1",
        checkpoint_id="cp1", pending_effects=(),
    )
    assert (good.control_epoch, good.revision, good.step_id) == (1, 2, "s1")


def test_unknown_issued_effect_is_never_blindly_replayed_on_restart(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    store.create(task_id="t", plan_id="p", step_id="s")
    store.checkpoint(
        "t", expected_epoch=0, expected_revision=0, step_id="s1",
        checkpoint_id="c1", pending_effects=(PendingEffect("e1", "remote effect"),),
    )
    issued = store.issue_effect("t", "e1", expected_epoch=0, expected_revision=1)
    assert issued.pending_effects[0].status == "unknown"
    resumed = TaskStore(store.path).resume("t")
    assert resumed.pending_effects == issued.pending_effects
    with pytest.raises(StateError, match="already issued"):
        store.issue_effect(
            "t", "e1", expected_epoch=1, expected_revision=resumed.revision,
        )
    resolved = store.resolve_effect(
        "t", "e1", "external-receipt", expected_epoch=1,
        expected_revision=resumed.revision,
    )
    assert resolved.pending_effects[0].status == "resolved"
    assert resolved.pending_effects[0].receipt_id == "external-receipt"
    with pytest.raises(StateError):
        store.resolve_effect(
            "t", "e1", "duplicate-receipt", expected_epoch=1,
            expected_revision=resolved.revision,
        )


def test_duplicate_identity_and_bad_checkpoint_roll_back(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    old = store.create(task_id="t", plan_id="p", step_id="s")
    with pytest.raises(StateError, match="already exists"):
        store.create(task_id="t", plan_id="alternate", step_id="s2")
    with pytest.raises(StateError, match="duplicate effect_id"):
        store.checkpoint(
            "t", expected_epoch=0, expected_revision=0, step_id="s1",
            checkpoint_id="c1",
            pending_effects=(PendingEffect("e", "x"), PendingEffect("e", "y")),
        )
    with pytest.raises(StateError, match="new checkpoint_id"):
        store.checkpoint(
            "t", expected_epoch=0, expected_revision=0, step_id="s1",
            checkpoint_id="initial", pending_effects=(),
        )
    assert TaskStore(store.path).load("t") == old
    assert store.load_revision("t", 0) == old
    with pytest.raises(StateError):
        store.load_revision("t", 1)


def test_two_concurrent_writers_have_one_cas_winner(tmp_path: Path) -> None:
    db = tmp_path / "tasks.sqlite"
    store = TaskStore(db)
    store.create(task_id="t", plan_id="p", step_id="s")

    def write(name: str) -> str:
        try:
            TaskStore(db).checkpoint(
                "t", expected_epoch=0, expected_revision=0,
                step_id=name, checkpoint_id=f"checkpoint-{name}", pending_effects=(),
            )
        except StaleRevision:
            return "stale"
        return "success"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, ("a", "b")))
    assert sorted(outcomes) == ["stale", "success"]
    assert store.load("t").revision == 1


def test_corrupted_row_or_history_fails_closed(tmp_path: Path) -> None:
    db = tmp_path / "tasks.sqlite"
    store = TaskStore(db)
    store.create(task_id="t", plan_id="p", step_id="s")
    with sqlite3.connect(db) as con:
        con.execute("UPDATE tasks SET snapshot='{}' WHERE task_id='t'")
    with pytest.raises(StateError, match="hash mismatch"):
        store.load("t")
    with sqlite3.connect(db) as con:
        con.execute("UPDATE tasks SET snapshot=(SELECT snapshot FROM history WHERE task_id='t')")
        con.execute("UPDATE history SET digest='bad' WHERE task_id='t'")
    with pytest.raises(StateError, match="checkpoint/history mismatch"):
        store.load("t")


def test_storage_version_and_adversarial_types_rejected(tmp_path: Path) -> None:
    db = tmp_path / "tasks.sqlite"
    store = TaskStore(db)
    with pytest.raises(StateError):
        store.create(task_id="t", plan_id="p", step_id="")
    store.create(task_id="t", plan_id="p", step_id="s")
    with pytest.raises(StateError):
        store.checkpoint(
            "t", expected_epoch=True, expected_revision=0, step_id="x",
            checkpoint_id="c1", pending_effects=(),
        )
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA user_version=99")
    with pytest.raises(StateError, match="unsupported"):
        TaskStore(db)
    with pytest.raises(StateError, match="unsupported"):
        store.load("t")
