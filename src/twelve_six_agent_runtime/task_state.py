"""Crash-durable, model-neutral agent task-state authority (Plan 5 / Section 2).

No tool is executed by this module. Issued effects must be reconciled against
external receipts; a restart never assumes retry is safe.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

SCHEMA = "12-6.agent-task-state.v1"
STORAGE_VERSION = 1
EffectStatus = Literal["pending", "unknown", "resolved"]


class StateError(ValueError):
    """Invalid identity, broken durable store, or unsafe state transition."""


class StaleEpoch(StateError):
    """An asynchronous result belongs to a superseded control epoch."""


class StaleRevision(StateError):
    """A competing transition has already advanced durable state."""


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or not (1 <= len(value) <= 256) or not value.strip():
        raise StateError(f"{name} must be a nonempty bounded identifier")
    return value


def _uint(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise StateError(f"{name} must be a nonnegative integer")
    return value


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in items:
        if key in out:
            raise StateError("duplicate JSON member")
        out[key] = value
    return out


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PendingEffect:
    effect_id: str
    description: str
    status: EffectStatus = "pending"
    receipt_id: str | None = None

    def validate(self) -> None:
        _id(self.effect_id, "effect_id")
        if not isinstance(self.description, str) or len(self.description) > 2000:
            raise StateError("effect description must be bounded text")
        if self.status not in ("pending", "unknown", "resolved"):
            raise StateError("invalid effect status")
        if self.status == "resolved":
            _id(self.receipt_id, "receipt_id")
        elif self.receipt_id is not None:
            raise StateError("unresolved effect cannot assert a receipt")


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    plan_id: str
    step_id: str
    checkpoint_id: str
    revision: int
    control_epoch: int
    pending_effects: tuple[PendingEffect, ...] = ()

    def validate(self) -> None:
        for field in ("task_id", "plan_id", "step_id", "checkpoint_id"):
            _id(getattr(self, field), field)
        _uint(self.revision, "revision")
        _uint(self.control_epoch, "control_epoch")
        if type(self.pending_effects) is not tuple or len(self.pending_effects) > 128:
            raise StateError("pending effects must be a bounded tuple")
        seen: set[str] = set()
        for effect in self.pending_effects:
            if not isinstance(effect, PendingEffect):
                raise StateError("invalid effect type")
            effect.validate()
            if effect.effect_id in seen:
                raise StateError("duplicate effect_id")
            seen.add(effect.effect_id)

    def encode(self) -> str:
        self.validate()
        value = {"schema": SCHEMA, **asdict(self)}
        encoded = _json(value)
        if len(encoded.encode("utf-8")) > 1024 * 1024:
            raise StateError("task state exceeds 1 MiB")
        return encoded


def _decode(raw: str, digest: str) -> TaskSnapshot:
    if _digest(raw) != digest:
        raise StateError("task state hash mismatch")
    try:
        value = json.loads(
            raw, object_pairs_hook=_pairs,
            parse_constant=lambda name: (_ for _ in ()).throw(StateError(name)),
        )
        if not isinstance(value, dict) or set(value) != {
            "schema", "task_id", "plan_id", "step_id", "checkpoint_id",
            "revision", "control_epoch", "pending_effects",
        } or value["schema"] != SCHEMA:
            raise StateError("incompatible task snapshot schema")
        effects = value.pop("pending_effects")
        value.pop("schema")
        if not isinstance(effects, list):
            raise StateError("effects must be an array")
        allowed = {"effect_id", "description", "status", "receipt_id"}
        if any(not isinstance(item, dict) or set(item) != allowed for item in effects):
            raise StateError("noncanonical effect schema")
        state = TaskSnapshot(**value, pending_effects=tuple(PendingEffect(**item) for item in effects))
        state.validate()
        if state.encode() != raw:
            raise StateError("noncanonical task state encoding")
        return state
    except (TypeError, KeyError, json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise StateError("malformed task state") from exc


class TaskStore:
    """SQLite-serialized task checkpoints with CAS and restart epoch fencing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink() or not self.path.parent.is_dir() or self.path.is_dir():
            raise StateError("task database path must be a regular local path")
        with self._transaction(initialize=True):
            pass

    @contextmanager
    def _transaction(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=10, isolation_level=None)) as db:
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("BEGIN IMMEDIATE")
            try:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in ((0, 1) if initialize else (1,)):
                    raise StateError("unsupported durable task storage version")
                if initialize and version == 0:
                    others = db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                    if others:
                        raise StateError("existing unversioned database cannot be claimed")
                    db.execute(
                        "CREATE TABLE tasks (task_id TEXT PRIMARY KEY, "
                        "snapshot TEXT NOT NULL, digest TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE history (task_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                        "snapshot TEXT NOT NULL, digest TEXT NOT NULL, "
                        "PRIMARY KEY(task_id, revision))"
                    )
                    db.execute("PRAGMA user_version=1")
                if db.execute("SELECT name FROM sqlite_master WHERE name='tasks'").fetchone() is None:
                    raise StateError("task table is missing")
                if db.execute("SELECT name FROM sqlite_master WHERE name='history'").fetchone() is None:
                    raise StateError("task history is missing")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def _read(self, db: sqlite3.Connection, task_id: str) -> TaskSnapshot:
        row = db.execute(
            "SELECT snapshot, digest FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise StateError("unknown task_id")
        state = _decode(*row)
        if state.task_id != task_id:
            raise StateError("durable task identity mismatch")
        history = db.execute(
            "SELECT revision, snapshot, digest FROM history "
            "WHERE task_id=? ORDER BY revision DESC LIMIT 1", (task_id,)
        ).fetchone()
        if history is None or history != (state.revision, *row):
            raise StateError("checkpoint/history mismatch")
        return state

    @staticmethod
    def _persist(db: sqlite3.Connection, state: TaskSnapshot, *, new: bool) -> None:
        encoded = state.encode()
        digest = _digest(encoded)
        if new:
            db.execute(
                "INSERT INTO tasks(task_id, snapshot, digest) VALUES (?, ?, ?)",
                (state.task_id, encoded, digest),
            )
        else:
            db.execute(
                "UPDATE tasks SET snapshot=?, digest=? WHERE task_id=?",
                (encoded, digest, state.task_id),
            )
        db.execute(
            "INSERT INTO history(task_id, revision, snapshot, digest) VALUES (?, ?, ?, ?)",
            (state.task_id, state.revision, encoded, digest),
        )

    def create(self, *, task_id: str, plan_id: str, step_id: str) -> TaskSnapshot:
        state = TaskSnapshot(task_id, plan_id, step_id, "initial", 0, 0)
        state.validate()
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM tasks WHERE task_id=?", (task_id,)).fetchone():
                raise StateError("task_id already exists")
            self._persist(db, state, new=True)
        return state

    def load(self, task_id: str) -> TaskSnapshot:
        _id(task_id, "task_id")
        with self._transaction() as db:
            return self._read(db, task_id)

    def load_revision(self, task_id: str, revision: int) -> TaskSnapshot:
        _id(task_id, "task_id")
        _uint(revision, "revision")
        with self._transaction() as db:
            self._read(db, task_id)
            row = db.execute(
                "SELECT snapshot, digest FROM history WHERE task_id=? AND revision=?",
                (task_id, revision),
            ).fetchone()
            if row is None:
                raise StateError("unknown checkpoint revision")
            return _decode(*row)

    def _change(
        self,
        task_id: str,
        *,
        expected_epoch: int,
        expected_revision: int,
        update: Callable[[TaskSnapshot], TaskSnapshot],
    ) -> TaskSnapshot:
        _id(task_id, "task_id")
        _uint(expected_epoch, "expected_epoch")
        _uint(expected_revision, "expected_revision")
        with self._transaction() as db:
            old = self._read(db, task_id)
            if old.control_epoch != expected_epoch:
                raise StaleEpoch("stale async control epoch")
            if old.revision != expected_revision:
                raise StaleRevision("stale state revision")
            state = update(old)
            if (
                state.task_id != old.task_id
                or state.plan_id != old.plan_id
                or state.revision != old.revision + 1
            ):
                raise StateError("invalid state lineage")
            self._persist(db, state, new=False)
        return state

    def checkpoint(
        self,
        task_id: str,
        *,
        expected_epoch: int,
        expected_revision: int,
        step_id: str,
        checkpoint_id: str,
        pending_effects: tuple[PendingEffect, ...],
    ) -> TaskSnapshot:
        _id(step_id, "step_id")
        _id(checkpoint_id, "checkpoint_id")

        def update(old: TaskSnapshot) -> TaskSnapshot:
            if checkpoint_id == old.checkpoint_id:
                raise StateError("new checkpoint_id required")
            if type(pending_effects) is not tuple:
                raise StateError("pending_effects must be a tuple")
            existing = {effect.effect_id: effect for effect in old.pending_effects}
            incoming = {effect.effect_id: effect for effect in pending_effects
                        if isinstance(effect, PendingEffect)}
            if len(incoming) != len(pending_effects):
                raise StateError("invalid or duplicate pending effects")
            if any(incoming.get(key) != effect for key, effect in existing.items()):
                raise StateError("checkpoint cannot erase or rewrite a reserved effect")
            if any(effect.status != "pending" for effect_id, effect in incoming.items()
                   if effect_id not in existing):
                raise StateError("new effects must begin pending")
            return replace(
                old, revision=old.revision + 1, step_id=step_id,
                checkpoint_id=checkpoint_id, pending_effects=pending_effects,
            )

        return self._change(
            task_id, expected_epoch=expected_epoch, expected_revision=expected_revision,
            update=update,
        )

    def issue_effect(
        self, task_id: str, effect_id: str, *, expected_epoch: int, expected_revision: int
    ) -> TaskSnapshot:
        _id(effect_id, "effect_id")

        def update(old: TaskSnapshot) -> TaskSnapshot:
            effects = []
            found = False
            for effect in old.pending_effects:
                if effect.effect_id == effect_id:
                    if effect.status != "pending":
                        raise StateError("effect already issued or resolved")
                    effect = replace(effect, status="unknown")
                    found = True
                effects.append(effect)
            if not found:
                raise StateError("unknown effect_id")
            return replace(old, revision=old.revision + 1, pending_effects=tuple(effects))

        return self._change(
            task_id, expected_epoch=expected_epoch, expected_revision=expected_revision,
            update=update,
        )

    def resolve_effect(
        self, task_id: str, effect_id: str, receipt_id: str,
        *, expected_epoch: int, expected_revision: int,
    ) -> TaskSnapshot:
        _id(effect_id, "effect_id")
        _id(receipt_id, "receipt_id")

        def update(old: TaskSnapshot) -> TaskSnapshot:
            effects = []
            found = False
            for effect in old.pending_effects:
                if effect.effect_id == effect_id:
                    if effect.status != "unknown":
                        raise StateError("effect must be issued and unresolved")
                    effect = replace(effect, status="resolved", receipt_id=receipt_id)
                    found = True
                effects.append(effect)
            if not found:
                raise StateError("unknown effect_id")
            return replace(old, revision=old.revision + 1, pending_effects=tuple(effects))

        return self._change(
            task_id, expected_epoch=expected_epoch, expected_revision=expected_revision,
            update=update,
        )

    def resume(self, task_id: str) -> TaskSnapshot:
        """Issue a new epoch lease; preserve exact step, checkpoint and effects."""
        _id(task_id, "task_id")
        with self._transaction() as db:
            old = self._read(db, task_id)
            state = replace(
                old, control_epoch=old.control_epoch + 1, revision=old.revision + 1
            )
            self._persist(db, state, new=False)
        return state
