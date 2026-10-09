from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from twelve_six.integration.plan10_release_state_binding import (
    KINDS,
    StateBindingError,
    capture,
    restore,
)


def make_components(root: Path) -> dict[str, Path]:
    result = {}
    for kind in KINDS:
        folder = root / kind
        folder.mkdir()
        if kind in {"tasks", "memory"}:
            with sqlite3.connect(folder / "state.sqlite") as db:
                db.execute("PRAGMA user_version=1")
                if kind == "tasks":
                    db.execute(
                        "CREATE TABLE tasks (task_id TEXT PRIMARY KEY, "
                        "snapshot TEXT NOT NULL, digest TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE history (task_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                        "snapshot TEXT NOT NULL, digest TEXT NOT NULL, "
                        "PRIMARY KEY(task_id, revision))"
                    )
                else:
                    db.execute("CREATE TABLE meta (revision INTEGER NOT NULL)")
                    db.execute("INSERT INTO meta VALUES (0)")
                    db.execute(
                        "CREATE TABLE records (memory_id TEXT PRIMARY KEY, "
                        "sequence INTEGER UNIQUE NOT NULL, payload TEXT NOT NULL, "
                        "digest TEXT NOT NULL, replaces_id TEXT UNIQUE)"
                    )
        else:
            (folder / "state.json").write_text(json.dumps({"kind": kind}))
        result[kind] = folder
    return result


def test_roundtrip_all_components_and_restart(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    snapshot = tmp_path / "backup"
    digest = capture(roots, snapshot, writers_stopped=True)
    restore(snapshot, tmp_path / "restored", manifest_sha256=digest)
    for kind in KINDS:
        source = snapshot / kind
        output = tmp_path / "restored" / kind
        assert sorted(p.name for p in source.iterdir()) == sorted(p.name for p in output.iterdir())
        if kind in {"tasks", "memory"}:
            with sqlite3.connect(output / "state.sqlite") as db:
                assert db.execute("PRAGMA user_version").fetchone() == (1,)
                assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert digest == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()


def test_swapped_store_schema_rejected(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    (roots["tasks"] / "state.sqlite").unlink()
    (roots["tasks"] / "state.sqlite").write_bytes(
        (roots["memory"] / "state.sqlite").read_bytes()
    )
    with pytest.raises(StateBindingError, match="canonical component store"):
        capture(roots, tmp_path / "backup", writers_stopped=True)


def test_quiescence_required(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    with pytest.raises(StateBindingError, match="quiescence"):
        capture(roots, tmp_path / "backup", writers_stopped=False)
    assert not (tmp_path / "backup").exists()


def test_missing_categories_fail_closed(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    roots.pop("checkpoints")
    with pytest.raises(StateBindingError):
        capture(roots, tmp_path / "backup", writers_stopped=True)


def test_corrupt_payload_prevents_any_publish(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    digest = capture(roots, tmp_path / "backup", writers_stopped=True)
    (tmp_path / "backup" / "model" / "state.json").write_text("tampered")
    with pytest.raises(StateBindingError, match="mismatch"):
        restore(tmp_path / "backup", tmp_path / "restored", manifest_sha256=digest)
    assert not (tmp_path / "restored").exists()


def test_forged_manifest_prevents_restore(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    digest = capture(roots, tmp_path / "backup", writers_stopped=True)
    (tmp_path / "backup" / "manifest.json").write_text("{}")
    with pytest.raises(StateBindingError, match="untrusted manifest"):
        restore(tmp_path / "backup", tmp_path / "restored", manifest_sha256=digest)
    assert not (tmp_path / "restored").exists()


def test_partial_restore_or_extra_member_fails(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    digest = capture(roots, tmp_path / "backup", writers_stopped=True)
    (tmp_path / "backup" / "configs" / "state.json").unlink()
    with pytest.raises(StateBindingError, match="incomplete"):
        restore(tmp_path / "backup", tmp_path / "restored", manifest_sha256=digest)
    assert not (tmp_path / "restored").exists()


def test_symlink_rejected_before_snapshot(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    (roots["configs"] / "host-file").symlink_to(tmp_path / "memory" / "state.sqlite")
    with pytest.raises(StateBindingError, match="nonregular"):
        capture(roots, tmp_path / "backup", writers_stopped=True)
    assert not (tmp_path / "backup").exists()


def test_versioned_sqlite_migration_fail_closed(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    with sqlite3.connect(roots["tasks"] / "state.sqlite") as db:
        db.execute("PRAGMA user_version=2")
    with pytest.raises(StateBindingError, match="unsupported"):
        capture(roots, tmp_path / "backup", writers_stopped=True)


def test_restore_existing_destination_does_not_clobber(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    digest = capture(roots, tmp_path / "backup", writers_stopped=True)
    dest = tmp_path / "restored"
    dest.mkdir()
    (dest / "marker").write_text("keep")
    with pytest.raises(StateBindingError, match="unsafe"):
        restore(tmp_path / "backup", dest, manifest_sha256=digest)
    assert (dest / "marker").read_text() == "keep"


def test_duplicate_snapshot_refuses_overwrite(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    capture(roots, tmp_path / "backup", writers_stopped=True)
    with pytest.raises(StateBindingError, match="existing"):
        capture(roots, tmp_path / "backup", writers_stopped=True)


def test_malformed_digest_does_not_publish(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    capture(roots, tmp_path / "backup", writers_stopped=True)
    with pytest.raises(StateBindingError, match="unsafe"):
        restore(tmp_path / "backup", tmp_path / "restored", manifest_sha256="abc")


def test_snapshot_symlink_rejected_before_restore(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    digest = capture(roots, tmp_path / "backup", writers_stopped=True)
    item = tmp_path / "backup" / "memory" / "state.sqlite"
    item.unlink()
    item.symlink_to(roots["memory"] / "state.sqlite")
    with pytest.raises(StateBindingError, match="nonregular"):
        restore(tmp_path / "backup", tmp_path / "restored", manifest_sha256=digest)


@pytest.mark.parametrize(("kind", "table"), [("tasks", "tasks"), ("memory", "records")])
def test_canonical_store_rejects_extra_columns(
    tmp_path: Path, kind: str, table: str
) -> None:
    roots = make_components(tmp_path)
    with sqlite3.connect(roots[kind] / "state.sqlite") as db:
        db.execute(f"ALTER TABLE {table} ADD COLUMN untrusted TEXT")
    with pytest.raises(StateBindingError, match="canonical component store"):
        capture(roots, tmp_path / "backup", writers_stopped=True)
    assert not (tmp_path / "backup").exists()


def test_canonical_store_requires_unique_memory_identity(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    with sqlite3.connect(roots["memory"] / "state.sqlite") as db:
        db.execute("DROP TABLE records")
        db.execute(
            "CREATE TABLE records (memory_id TEXT PRIMARY KEY, "
            "sequence INTEGER NOT NULL, payload TEXT NOT NULL, "
            "digest TEXT NOT NULL, replaces_id TEXT)"
        )
    with pytest.raises(StateBindingError, match="canonical component store"):
        capture(roots, tmp_path / "backup", writers_stopped=True)
    assert not (tmp_path / "backup").exists()


def test_sqlite_uri_escapes_fragment_in_existing_path(tmp_path: Path) -> None:
    safe_path = tmp_path / "state#fragment"
    safe_path.mkdir()
    roots = make_components(safe_path)
    digest = capture(roots, safe_path / "backup", writers_stopped=True)
    restore(safe_path / "backup", safe_path / "restored", manifest_sha256=digest)
    assert (safe_path / "restored" / "tasks" / "state.sqlite").is_file()



def test_plan5_real_task_and_memory_records_survive_cold_restart(tmp_path: Path) -> None:
    from twelve_six_agent_runtime.memory import MemoryRecord, MemoryStore
    from twelve_six_agent_runtime.task_state import TaskStore

    roots = make_components(tmp_path)
    tasks = TaskStore(roots["tasks"] / "state.sqlite")
    before_task = tasks.create(task_id="task-a", plan_id="plan-a", step_id="initial")
    memory = MemoryStore(roots["memory"] / "state.sqlite")
    before_memory = MemoryRecord(
        memory_id="memo-a", kind="project", content="restored",
        source_id="owner", source_kind="owner", evidence_id="evidence-a",
        confidence_ppm=1_000_000, created_at=1,
    )
    assert memory.append(before_memory, expected_revision=0) == 1

    digest = capture(roots, tmp_path / "snapshot", writers_stopped=True)
    restore(tmp_path / "snapshot", tmp_path / "restored", manifest_sha256=digest)
    assert TaskStore(tmp_path / "restored" / "tasks" / "state.sqlite").load(
        "task-a"
    ) == before_task
    assert MemoryStore(tmp_path / "restored" / "memory" / "state.sqlite").history() == (
        before_memory,
    )


def test_task_history_orphan_fails_even_with_valid_sqlite(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    with sqlite3.connect(roots["tasks"] / "state.sqlite") as db:
        db.execute(
            "INSERT INTO history VALUES ('orphan', 0, '{}', 'bogus')"
        )
    with pytest.raises(StateBindingError, match="canonical task history"):
        capture(roots, tmp_path / "snapshot", writers_stopped=True)
    assert not (tmp_path / "snapshot").exists()


def test_task_history_hash_mismatch_fails_snapshot(tmp_path: Path) -> None:
    from twelve_six_agent_runtime.task_state import TaskStore

    roots = make_components(tmp_path)
    TaskStore(roots["tasks"] / "state.sqlite").create(
        task_id="task-a", plan_id="plan-a", step_id="initial"
    )
    with sqlite3.connect(roots["tasks"] / "state.sqlite") as db:
        db.execute("UPDATE history SET digest=?", ("0" * 64,))
    with pytest.raises(StateBindingError, match="canonical task history"):
        capture(roots, tmp_path / "snapshot", writers_stopped=True)


def test_memory_revision_and_digest_corruption_refuse_snapshot(tmp_path: Path) -> None:
    from twelve_six_agent_runtime.memory import MemoryRecord, MemoryStore

    roots = make_components(tmp_path)
    MemoryStore(roots["memory"] / "state.sqlite").append(
        MemoryRecord(
            memory_id="memo-a", kind="project", content="valid",
            source_id="owner", source_kind="owner", evidence_id="evidence-a",
            confidence_ppm=900_000, created_at=1,
        ),
        expected_revision=0,
    )
    path = roots["memory"] / "state.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("UPDATE meta SET revision=9")
    with pytest.raises(StateBindingError, match="canonical memory history"):
        capture(roots, tmp_path / "snapshot", writers_stopped=True)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE meta SET revision=1")
        db.execute("UPDATE records SET digest=?", ("0" * 64,))
    with pytest.raises(StateBindingError, match="canonical memory history"):
        capture(roots, tmp_path / "snapshot", writers_stopped=True)


def test_resealed_but_semantically_invalid_memory_restore_refused(tmp_path: Path) -> None:
    roots = make_components(tmp_path)
    snapshot = tmp_path / "snapshot"
    capture(roots, snapshot, writers_stopped=True)
    with sqlite3.connect(snapshot / "memory" / "state.sqlite") as db:
        db.execute("UPDATE meta SET revision=77")
    member = (snapshot / "memory" / "state.sqlite").read_bytes()
    manifest = json.loads((snapshot / "manifest.json").read_bytes())
    manifest["members"]["memory/state.sqlite"] = hashlib.sha256(member).hexdigest()
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    (snapshot / "manifest.json").write_bytes(canonical)
    with pytest.raises(StateBindingError, match="canonical memory history"):
        restore(
            snapshot, tmp_path / "restored",
            manifest_sha256=hashlib.sha256(canonical).hexdigest(),
        )
    assert not (tmp_path / "restored").exists()


def test_restore_fences_old_task_epoch_and_never_reissues_unknown_effect(
    tmp_path: Path,
) -> None:
    from twelve_six_agent_runtime.task_state import PendingEffect, StaleEpoch, TaskStore

    roots = make_components(tmp_path)
    task = TaskStore(roots["tasks"] / "state.sqlite")
    task.create(task_id="task-a", plan_id="plan-a", step_id="initial")
    task.checkpoint(
        "task-a", expected_epoch=0, expected_revision=0,
        step_id="external-tool", checkpoint_id="checkpoint-1",
        pending_effects=(PendingEffect("effect-a", "external action"),),
    )
    before = task.issue_effect(
        "task-a", "effect-a", expected_epoch=0, expected_revision=1
    )
    assert before.pending_effects[0].status == "unknown"
    digest = capture(roots, tmp_path / "snapshot", writers_stopped=True)
    restore(tmp_path / "snapshot", tmp_path / "restored", manifest_sha256=digest)

    recovered = TaskStore(tmp_path / "restored" / "tasks" / "state.sqlite")
    assert recovered.load("task-a") == before
    resumed = recovered.resume("task-a")
    assert resumed.control_epoch == before.control_epoch + 1
    assert resumed.pending_effects == before.pending_effects
    with pytest.raises(StaleEpoch):
        recovered.issue_effect(
            "task-a", "effect-a",
            expected_epoch=before.control_epoch,
            expected_revision=resumed.revision,
        )
