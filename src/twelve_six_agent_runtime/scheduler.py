"""Reference LOCAL_FREE scheduler: durable admission, never an effect executor.

Plan 5 / Section 6. A lease is permission to *offer* one task to a trusted
host, not permission to execute tools, spend money, or treat model output as
resource evidence. TaskStore remains the sole task/effect authority.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .task_state import StateError, TaskStore, _digest, _id, _json, _pairs, _uint

SCHEMA = "12-6.agent-scheduler.v1"
State = Literal["ready", "running", "paused", "done"]


class SchedulerError(ValueError):
    """Incompatible policy, unverified resource state, or ambiguous lease."""


@dataclass(frozen=True)
class ResourcePolicy:
    max_parallel: int
    cpu_capacity: int
    memory_capacity_mb: int
    total_attempt_budget: int

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _uint(value, name)
            if value == 0 or value > 1_000_000:
                raise SchedulerError("resource policy must be bounded and positive")


@dataclass(frozen=True)
class ScheduleSpec:
    task_id: str
    goal_id: str
    kind: Literal["foreground", "background"]
    priority: int
    deadline: int
    cpu_units: int
    memory_mb: int
    attempt_budget: int

    def validate(self) -> None:
        _id(self.task_id, "task_id")
        _id(self.goal_id, "goal_id")
        if self.kind not in ("foreground", "background"):
            raise SchedulerError("invalid work kind")
        for name in ("priority", "deadline", "cpu_units", "memory_mb", "attempt_budget"):
            _uint(getattr(self, name), name)
        if not 0 <= self.priority <= 100 or not self.cpu_units or not self.memory_mb:
            raise SchedulerError("invalid task priority or resource request")
        if not 1 <= self.attempt_budget <= 100_000:
            raise SchedulerError("invalid bounded attempt budget")


@dataclass(frozen=True)
class ResourceReading:
    observed_at: int
    cpu_free: int
    memory_free_mb: int
    pressure: Literal["normal", "high"]

    def validate(self) -> None:
        for name in ("observed_at", "cpu_free", "memory_free_mb"):
            _uint(getattr(self, name), name)
        if self.pressure not in ("normal", "high"):
            raise SchedulerError("unknown resource pressure")


@dataclass(frozen=True)
class DispatchLease:
    task_id: str
    lease_id: str
    attempt: int
    control_epoch: int
    task_revision: int


def _decode(raw: str, digest: str) -> dict[str, object]:
    if not isinstance(raw, str) or _digest(raw) != digest:
        raise SchedulerError("scheduler record digest mismatch")
    try:
        data = json.loads(
            raw, object_pairs_hook=_pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(SchedulerError(v)),
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise SchedulerError("malformed scheduler record") from exc
    if not isinstance(data, dict) or _json(data) != raw:
        raise SchedulerError("noncanonical scheduler record")
    return data


class SchedulerStore:
    """SQLite FULL-sync admission and lease ledger; external effects never run here."""

    def __init__(self, path: str | Path, tasks: TaskStore, policy: ResourcePolicy) -> None:
        if not isinstance(tasks, TaskStore) or not isinstance(policy, ResourcePolicy):
            raise SchedulerError("TaskStore and policy are required")
        policy.validate()
        self.tasks, self.policy, self.path = tasks, policy, Path(path)
        if (self.path.is_symlink() or self.path.is_dir()
                or not self.path.parent.is_dir() or self.path == tasks.path):
            raise SchedulerError("unsafe or colliding scheduler storage")
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
                    raise SchedulerError("incompatible scheduler storage version")
                if initialize and version == 0:
                    if db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall():
                        raise SchedulerError("unversioned database cannot be claimed")
                    db.execute("CREATE TABLE config (raw TEXT NOT NULL, digest TEXT NOT NULL)")
                    db.execute(
                        "CREATE TABLE jobs (task_id TEXT PRIMARY KEY, raw TEXT NOT NULL, "
                        "digest TEXT NOT NULL)"
                    )
                    self._save_config(db, paused=False, charged=0)
                    db.execute("PRAGMA user_version=1")
                config = db.execute("SELECT raw, digest FROM config").fetchall()
                if len(config) != 1:
                    raise SchedulerError("missing or duplicate scheduler configuration")
                meta = _decode(*config[0])
                if set(meta) != {"schema", "policy", "paused", "charged"}:
                    raise SchedulerError("invalid scheduler configuration fields")
                if meta["schema"] != SCHEMA or meta["policy"] != asdict(self.policy):
                    raise SchedulerError("scheduler policy/version drift")
                if type(meta["paused"]) is not bool:
                    raise SchedulerError("invalid pause state")
                _uint(meta["charged"], "charged")
                if meta["charged"] > self.policy.total_attempt_budget:
                    raise SchedulerError("budget overflow")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def _save_config(self, db: sqlite3.Connection, *, paused: bool, charged: int) -> None:
        raw = _json({
            "schema": SCHEMA, "policy": asdict(self.policy),
            "paused": paused, "charged": charged,
        })
        db.execute("DELETE FROM config")
        db.execute("INSERT INTO config VALUES (?, ?)", (raw, _digest(raw)))

    @staticmethod
    def _records(db: sqlite3.Connection) -> dict[str, dict[str, object]]:
        rows = db.execute("SELECT task_id, raw, digest FROM jobs ORDER BY task_id").fetchall()
        jobs: dict[str, dict[str, object]] = {}
        for task_id, raw, digest in rows:
            record = _decode(raw, digest)
            if set(record) != {"schema", "spec", "state", "attempt", "lease_id",
                               "control_epoch", "task_revision", "checkpoint_id"}:
                raise SchedulerError("invalid scheduler job fields")
            spec = ScheduleSpec(**record["spec"])
            spec.validate()
            if record["schema"] != SCHEMA or spec.task_id != task_id:
                raise SchedulerError("scheduler identity mismatch")
            if record["state"] not in ("ready", "running", "paused", "done"):
                raise SchedulerError("invalid scheduler state")
            _uint(record["attempt"], "attempt")
            _uint(record["control_epoch"], "control_epoch")
            _uint(record["task_revision"], "task_revision")
            if record["attempt"] > spec.attempt_budget:
                raise SchedulerError("attempt budget corrupted")
            if record["state"] == "running":
                _id(record["lease_id"], "lease_id")
            elif record["lease_id"] is not None:
                raise SchedulerError("inactive job contains active lease")
            if record["checkpoint_id"] is not None:
                _id(record["checkpoint_id"], "checkpoint_id")
            jobs[task_id] = record
        return jobs

    @staticmethod
    def _put(db: sqlite3.Connection, job: dict[str, object]) -> None:
        raw = _json(job)
        db.execute(
            "INSERT INTO jobs VALUES (?, ?, ?) "
            "ON CONFLICT(task_id) DO UPDATE SET raw=excluded.raw, digest=excluded.digest",
            (job["spec"]["task_id"], raw, _digest(raw)),
        )

    def register(self, spec: ScheduleSpec) -> None:
        if not isinstance(spec, ScheduleSpec):
            raise SchedulerError("invalid task spec")
        spec.validate()
        if spec.cpu_units > self.policy.cpu_capacity or (
            spec.memory_mb > self.policy.memory_capacity_mb
        ):
            raise SchedulerError("task exceeds scheduler envelope")
        current = self.tasks.load(spec.task_id)
        with self._tx() as db:
            jobs = self._records(db)
            if spec.task_id in jobs:
                raise SchedulerError("duplicate scheduler task")
            self._put(db, {
                "schema": SCHEMA, "spec": asdict(spec), "state": "ready",
                "attempt": 0, "lease_id": None, "control_epoch": current.control_epoch,
                "task_revision": current.revision, "checkpoint_id": None,
            })

    def snapshot(self) -> tuple[dict[str, object], ...]:
        with self._tx() as db:
            return tuple(self._records(db).values())

    def offer(
        self, *, now: int, reading: ResourceReading,
        verify_reading: Callable[[ResourceReading], bool],
    ) -> DispatchLease | None:
        _uint(now, "now")
        if not isinstance(reading, ResourceReading):
            raise SchedulerError("missing trusted resource reading")
        reading.validate()
        if reading.observed_at > now or now - reading.observed_at > 60:
            raise SchedulerError("stale or future resource reading")
        if not callable(verify_reading) or verify_reading(reading) is not True:
            raise SchedulerError("unverified resource observation")
        with self._tx() as db:
            rows = db.execute("SELECT raw, digest FROM config").fetchone()
            meta = _decode(*rows)
            jobs = self._records(db)
            if reading.pressure == "high":
                self._save_config(db, paused=True, charged=meta["charged"])
                return None
            if meta["paused"]:
                return None
            running = [j for j in jobs.values() if j["state"] == "running"]
            if len(running) >= self.policy.max_parallel:
                return None
            cpu_used = sum(j["spec"]["cpu_units"] for j in running)
            mem_used = sum(j["spec"]["memory_mb"] for j in running)
            candidates = sorted(
                (j for j in jobs.values() if j["state"] == "ready"),
                key=lambda j: (
                    -j["spec"]["priority"], j["spec"]["deadline"],
                    j["spec"]["kind"] == "background", j["spec"]["task_id"],
                ),
            )
            for job in candidates:
                spec = ScheduleSpec(**job["spec"])
                if now > spec.deadline or job["attempt"] >= spec.attempt_budget:
                    continue
                if meta["charged"] >= self.policy.total_attempt_budget:
                    return None
                if spec.cpu_units > min(
                    self.policy.cpu_capacity - cpu_used, reading.cpu_free
                ) or spec.memory_mb > min(
                    self.policy.memory_capacity_mb - mem_used, reading.memory_free_mb
                ):
                    continue
                current = self.tasks.load(spec.task_id)
                if any(effect.status == "unknown" for effect in current.pending_effects):
                    continue
                if current.control_epoch < job["control_epoch"]:
                    raise SchedulerError("task epoch rollback")
                attempt = job["attempt"] + 1
                lease_id = f"{spec.task_id}:{attempt}:{current.control_epoch}:{current.revision}"
                job.update(
                    state="running", attempt=attempt, lease_id=lease_id,
                    control_epoch=current.control_epoch, task_revision=current.revision,
                )
                self._put(db, job)
                self._save_config(db, paused=False, charged=meta["charged"] + 1)
                return DispatchLease(
                    spec.task_id, lease_id, attempt,
                    current.control_epoch, current.revision,
                )
            return None

    def pause_for_pressure(
        self, reading: ResourceReading, verify_reading: Callable[[ResourceReading], bool],
    ) -> None:
        if not isinstance(reading, ResourceReading) or reading.pressure != "high":
            raise SchedulerError("high-pressure host reading required")
        reading.validate()
        if not callable(verify_reading) or verify_reading(reading) is not True:
            raise SchedulerError("unverified high-pressure reading")
        with self._tx() as db:
            meta = _decode(*db.execute("SELECT raw, digest FROM config").fetchone())
            self._records(db)
            self._save_config(db, paused=True, charged=meta["charged"])

    def resume_admission(
        self, reading: ResourceReading, verify_reading: Callable[[ResourceReading], bool],
    ) -> None:
        if not isinstance(reading, ResourceReading) or reading.pressure != "normal":
            raise SchedulerError("normal host reading required")
        reading.validate()
        if not callable(verify_reading) or verify_reading(reading) is not True:
            raise SchedulerError("unverified recovery reading")
        with self._tx() as db:
            meta = _decode(*db.execute("SELECT raw, digest FROM config").fetchone())
            self._records(db)
            self._save_config(db, paused=False, charged=meta["charged"])

    def complete(
        self, lease: DispatchLease, *, outcome: Literal["done", "checkpoint"],
        evidence_id: str, verify_evidence: Callable[[DispatchLease, str], bool],
    ) -> None:
        if not isinstance(lease, DispatchLease) or outcome not in ("done", "checkpoint"):
            raise SchedulerError("invalid completion lease/outcome")
        _id(evidence_id, "evidence_id")
        if not callable(verify_evidence) or verify_evidence(lease, evidence_id) is not True:
            raise SchedulerError("unverified completion/checkpoint")
        with self._tx() as db:
            jobs = self._records(db)
            job = jobs.get(lease.task_id)
            if job is None or job["state"] != "running" or (
                job["lease_id"] != lease.lease_id
                or job["attempt"] != lease.attempt
                or job["control_epoch"] != lease.control_epoch
                or job["task_revision"] != lease.task_revision
            ):
                raise SchedulerError("stale or already consumed dispatch lease")
            current = self.tasks.load(lease.task_id)
            if current.control_epoch != lease.control_epoch:
                raise SchedulerError("task epoch changed: reconcile before completion")
            if any(effect.status != "resolved" for effect in current.pending_effects):
                raise SchedulerError("pending/unknown effects require reconciliation")
            job.update(
                state="done" if outcome == "done" else "paused",
                lease_id=None, checkpoint_id=evidence_id,
                task_revision=current.revision,
            )
            self._put(db, job)

    def retry_checkpoint(
        self, task_id: str, *, evidence_id: str,
        verifier: Callable[[str, str], bool],
    ) -> None:
        _id(task_id, "task_id")
        _id(evidence_id, "evidence_id")
        if not callable(verifier) or verifier(task_id, evidence_id) is not True:
            raise SchedulerError("verified checkpoint recovery required")
        with self._tx() as db:
            jobs = self._records(db)
            job = jobs.get(task_id)
            if job is None or job["state"] != "paused":
                raise SchedulerError("no paused checkpoint to recover")
            if job["checkpoint_id"] != evidence_id:
                raise SchedulerError("checkpoint identity mismatch")
            current = self.tasks.load(task_id)
            if any(effect.status != "resolved" for effect in current.pending_effects):
                raise SchedulerError("unreconciled effect blocks retry")
            if job["attempt"] >= job["spec"]["attempt_budget"]:
                raise SchedulerError("task attempt budget exhausted")
            job.update(
                state="ready", control_epoch=current.control_epoch,
                task_revision=current.revision,
            )
            self._put(db, job)
