"""Plan 5 Section 6: LOCAL_FREE reference scheduler, never a tool executor.

TaskStore is the only authority for task epochs, steps, and external effects.
Scheduler metadata is a separate fail-closed queue/lease projection; a crash
between the two stores cannot authorize replay of an unknown effect.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from .task_state import StateError, TaskSnapshot, TaskStore, _digest, _id, _json, _pairs, _uint

SCHEMA = "12-6.agent-scheduler.v1"
Status = Literal["ready", "leased", "paused", "done"]


class SchedulerError(ValueError):
    """Invalid admission, stale lease, or corrupt scheduler metadata."""


@dataclass(frozen=True)
class WorkSpec:
    task_id: str
    plan_id: str
    goal_id: str
    priority: int
    deadline_tick: int
    max_steps: int
    cpu_units: int
    memory_mb: int
    background: bool = False

    def validate(self) -> None:
        for name in ("task_id", "plan_id", "goal_id"):
            _id(getattr(self, name), name)
        for name in ("priority", "deadline_tick", "max_steps", "cpu_units", "memory_mb"):
            _uint(getattr(self, name), name)
        if self.priority > 100 or self.max_steps == 0:
            raise SchedulerError("priority or budget invalid")
        if self.cpu_units == 0 or self.memory_mb == 0:
            raise SchedulerError("resource envelope must be positive")
        if type(self.background) is not bool:
            raise SchedulerError("background must be boolean")


@dataclass(frozen=True)
class WorkEntry:
    spec: WorkSpec
    status: Status = "ready"
    revision: int = 0
    spent_steps: int = 0
    epoch: int | None = None
    task_revision: int | None = None
    pause_reason: str | None = None

    def validate(self) -> None:
        self.spec.validate()
        if self.status not in ("ready", "leased", "paused", "done"):
            raise SchedulerError("invalid scheduling state")
        _uint(self.revision, "revision")
        _uint(self.spent_steps, "spent_steps")
        if self.spent_steps > self.spec.max_steps:
            raise SchedulerError("overspent resource budget")
        if self.epoch is not None:
            _uint(self.epoch, "epoch")
        if self.task_revision is not None:
            _uint(self.task_revision, "task_revision")
        if (self.epoch is None) != (self.task_revision is None):
            raise SchedulerError("partial control token")
        if self.status == "leased" and self.epoch is None:
            raise SchedulerError("lease without control token")
        if self.status == "paused" and not self.pause_reason:
            raise SchedulerError("paused work requires reason")
        if self.status != "paused" and self.pause_reason is not None:
            raise SchedulerError("unjustified pause reason")


@dataclass(frozen=True)
class Lease:
    task_id: str
    scheduler_revision: int
    control_epoch: int
    task_revision: int


def _encode(item: WorkEntry) -> str:
    item.validate()
    return _json({"schema": SCHEMA, **asdict(item)})


def _decode(raw: str, digest: str) -> WorkEntry:
    if _digest(raw) != digest:
        raise SchedulerError("scheduler record digest mismatch")
    try:
        doc = json.loads(
            raw, object_pairs_hook=_pairs,
            parse_constant=lambda val: (_ for _ in ()).throw(SchedulerError(val)),
        )
        if not isinstance(doc, dict) or set(doc) != {
            "schema", "spec", "status", "revision", "spent_steps",
            "epoch", "task_revision", "pause_reason",
        } or doc.pop("schema") != SCHEMA:
            raise SchedulerError("scheduler schema incompatible")
        spec = doc.pop("spec")
        if not isinstance(spec, dict) or set(spec) != set(WorkSpec.__dataclass_fields__):
            raise SchedulerError("scheduler envelope malformed")
        item = WorkEntry(spec=WorkSpec(**spec), **doc)
        if raw != _encode(item):
            raise SchedulerError("noncanonical scheduler payload")
        return item
    except (TypeError, KeyError, json.JSONDecodeError, RecursionError) as exc:
        raise SchedulerError("malformed scheduler record") from exc


class SchedulerStore:
    """Durable scheduler projection with TaskStore as sole task/effect authority."""

    def __init__(self, path: str | Path, tasks: TaskStore) -> None:
        if not isinstance(tasks, TaskStore):
            raise SchedulerError("canonical TaskStore required")
        self.tasks = tasks
        self.path = Path(path)
        if self.path.is_symlink() or self.path.is_dir() or not self.path.parent.is_dir():
            raise SchedulerError("unsafe scheduler storage path")
        if self.path == tasks.path:
            raise SchedulerError("queue database may not replace TaskStore")
        with self._tx(initialize=True):
            pass

    @contextmanager
    def _tx(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=10, isolation_level=None)) as db:
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("PRAGMA synchronous=FULL")
            if initialize:
                db.execute("PRAGMA journal_mode=DELETE")
            db.execute("BEGIN IMMEDIATE")
            try:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in ((0, 1) if initialize else (1,)):
                    raise SchedulerError("unsupported scheduler storage version")
                if initialize and version == 0:
                    if db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                        raise SchedulerError("cannot claim unversioned queue")
                    db.execute(
                        "CREATE TABLE queue (task_id TEXT PRIMARY KEY, "
                        "payload TEXT NOT NULL, digest TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE history (task_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                        "payload TEXT NOT NULL, digest TEXT NOT NULL, "
                        "PRIMARY KEY(task_id, revision))"
                    )
                    db.execute("PRAGMA user_version=1")
                for table in ("queue", "history"):
                    if not db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone():
                        raise SchedulerError("incomplete queue schema")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def _all(db: sqlite3.Connection) -> tuple[WorkEntry, ...]:
        rows = db.execute("SELECT task_id,payload,digest FROM queue ORDER BY task_id").fetchall()
        result = []
        for task_id, raw, digest in rows:
            item = _decode(raw, digest)
            if item.spec.task_id != task_id:
                raise SchedulerError("scheduler identity mismatch")
            last = db.execute(
                "SELECT revision,payload,digest FROM history "
                "WHERE task_id=? ORDER BY revision DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if last != (item.revision, raw, digest):
                raise SchedulerError("scheduler history diverged")
            result.append(item)
        return tuple(result)

    @staticmethod
    def _write(db: sqlite3.Connection, item: WorkEntry, *, create: bool) -> None:
        raw = _encode(item)
        digest = _digest(raw)
        if create:
            db.execute(
                "INSERT INTO queue VALUES (?,?,?)", (item.spec.task_id, raw, digest)
            )
        else:
            db.execute(
                "UPDATE queue SET payload=?,digest=? WHERE task_id=?",
                (raw, digest, item.spec.task_id),
            )
        db.execute(
            "INSERT INTO history VALUES (?,?,?,?)",
            (item.spec.task_id, item.revision, raw, digest),
        )

    def snapshot(self) -> tuple[WorkEntry, ...]:
        with self._tx() as db:
            return self._all(db)

    def register(self, spec: WorkSpec) -> WorkEntry:
        if not isinstance(spec, WorkSpec):
            raise SchedulerError("typed scheduler envelope required")
        spec.validate()
        with self._tx() as db:
            if any(item.spec.task_id == spec.task_id for item in self._all(db)):
                raise SchedulerError("duplicate scheduler task identity")
            # Crash recovery between the two SQLite authorities: trust only
            # an unmodified canonical initial task, never create a duplicate.
            try:
                incumbent = self.tasks.load(spec.task_id)
            except StateError as exc:
                if str(exc) != "unknown task_id":
                    raise
                incumbent = self.tasks.create(
                    task_id=spec.task_id, plan_id=spec.plan_id, step_id="queued"
                )
            if (incumbent.plan_id != spec.plan_id or incumbent.revision != 0
                    or incumbent.step_id != "queued" or incumbent.pending_effects):
                raise SchedulerError("cannot adopt noninitial canonical task")
            entry = WorkEntry(spec)
            self._write(db, entry, create=True)
            return entry

    @staticmethod
    def _safe(snapshot: TaskSnapshot) -> bool:
        # An issued unknown effect MUST be externally reconciled, not retried.
        return all(effect.status == "resolved" for effect in snapshot.pending_effects)

    def acquire(
        self, *, tick: int, available_cpu: int, available_memory_mb: int,
        pressure_ppm: int = 0,
    ) -> Lease | None:
        for label, val in (
            ("tick", tick), ("available_cpu", available_cpu),
            ("available_memory_mb", available_memory_mb), ("pressure_ppm", pressure_ppm)
        ):
            _uint(val, label)
        if pressure_ppm > 1_000_000:
            raise SchedulerError("invalid resource pressure")
        with self._tx() as db:
            entries = self._all(db)
            if any(item.status == "leased" for item in entries) or pressure_ppm >= 900_000:
                return None
            eligible = sorted(
                (
                    item for item in entries
                    if item.status == "ready"
                    and item.spent_steps < item.spec.max_steps
                    and tick <= item.spec.deadline_tick
                    and item.spec.cpu_units <= available_cpu
                    and item.spec.memory_mb <= available_memory_mb
                    and not (item.spec.background and pressure_ppm >= 500_000)
                ),
                key=lambda item: (-item.spec.priority, item.spec.deadline_tick, item.spec.task_id),
            )
            for item in eligible:
                state = self.tasks.load(item.spec.task_id)
                if not self._safe(state):
                    continue
                resumed = self.tasks.resume(item.spec.task_id)
                active = replace(
                    item, status="leased", revision=item.revision + 1,
                    epoch=resumed.control_epoch, task_revision=resumed.revision,
                )
                self._write(db, active, create=False)
                return Lease(
                    active.spec.task_id, active.revision,
                    resumed.control_epoch, resumed.revision,
                )
            return None

    def _active(self, db: sqlite3.Connection, token: Lease) -> tuple[WorkEntry, TaskSnapshot]:
        if not isinstance(token, Lease):
            raise SchedulerError("typed lease required")
        item = next((x for x in self._all(db) if x.spec.task_id == token.task_id), None)
        if (
            item is None or item.status != "leased"
            or (item.revision, item.epoch, item.task_revision) != (
                token.scheduler_revision, token.control_epoch, token.task_revision
            )
        ):
            raise SchedulerError("stale or unknown scheduler lease")
        state = self.tasks.load(token.task_id)
        if (
            (state.control_epoch, state.revision) != (token.control_epoch, token.task_revision)
            or not self._safe(state)
        ):
            raise SchedulerError("canonical TaskStore changed or unresolved effect")
        return item, state

    def checkpoint(
        self, token: Lease, *, used_steps: int, step_id: str,
        checkpoint_id: str, pressure_ppm: int = 0,
    ) -> WorkEntry:
        _uint(used_steps, "used_steps")
        _uint(pressure_ppm, "pressure_ppm")
        if pressure_ppm > 1_000_000:
            raise SchedulerError("invalid resource pressure")
        _id(step_id, "step_id")
        _id(checkpoint_id, "checkpoint_id")
        with self._tx() as db:
            item, old = self._active(db, token)
            spent = item.spent_steps + used_steps
            if spent > item.spec.max_steps:
                raise SchedulerError("scheduler budget exhausted")
            saved = self.tasks.checkpoint(
                token.task_id, expected_epoch=token.control_epoch,
                expected_revision=token.task_revision,
                step_id=step_id, checkpoint_id=checkpoint_id,
                pending_effects=old.pending_effects,
            )
            reason = (
                "budget_exhausted" if spent == item.spec.max_steps
                else "resource_pressure" if pressure_ppm >= 900_000 else None
            )
            result = replace(
                item, status="paused" if reason else "ready",
                revision=item.revision + 1, spent_steps=spent,
                task_revision=saved.revision, pause_reason=reason,
            )
            self._write(db, result, create=False)
            return result

    def complete(self, token: Lease, *, used_steps: int = 0) -> WorkEntry:
        _uint(used_steps, "used_steps")
        with self._tx() as db:
            item, _ = self._active(db, token)
            if item.spent_steps + used_steps > item.spec.max_steps:
                raise SchedulerError("completion exceeds budget")
            finished = replace(
                item, status="done", revision=item.revision + 1,
                spent_steps=item.spent_steps + used_steps,
            )
            self._write(db, finished, create=False)
            return finished

    def reconcile(
        self, task_id: str, *, verifier: Callable[[TaskSnapshot], bool],
        release_pressure: bool = True,
    ) -> WorkEntry:
        """Trusted host proves lease/effects clear after restart; never blind retries."""
        _id(task_id, "task_id")
        if type(release_pressure) is not bool:
            raise SchedulerError("invalid release flag")
        with self._tx() as db:
            item = next((x for x in self._all(db) if x.spec.task_id == task_id), None)
            if item is None or item.status not in ("leased", "paused"):
                raise SchedulerError("nothing recoverable for this task")
            state = self.tasks.load(task_id)
            if not self._safe(state) or not callable(verifier) or verifier(state) is not True:
                raise SchedulerError("external reconciliation evidence absent")
            if item.status == "paused" and item.pause_reason == "budget_exhausted":
                raise SchedulerError("budget cannot be replenished by recovery")
            if item.status == "paused" and not release_pressure:
                return item
            resumed = self.tasks.resume(task_id)
            restored = replace(
                item, status="ready", revision=item.revision + 1,
                epoch=resumed.control_epoch, task_revision=resumed.revision,
                pause_reason=None,
            )
            self._write(db, restored, create=False)
            return restored
