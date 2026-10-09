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
