"""Source-aware durable memory reference boundary, Plan 5 / Section 3.

Memory is not a source of external truth or permissions. A trusted host must
independently verify external assertions against a live source every time.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .context import ContextEntry
from .task_state import StateError, _digest, _id, _json, _pairs, _uint

SCHEMA = "12-6.agent-memory.v1"
STORAGE_VERSION = 1
MemoryKind = Literal["episodic", "semantic", "procedural", "project"]
ProvenanceKind = Literal["owner", "tool", "retrieval", "observation", "model"]


class MemoryError(ValueError):
    """Invalid memory or unsafe durable memory transition."""


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    kind: MemoryKind
    content: str
    source_id: str
    source_kind: ProvenanceKind
    evidence_ref: str
    confidence: float
    observed_at: int
    expires_at: int | None = None
    corrects: str | None = None

    def validate(self) -> None:
        for field in ("memory_id", "source_id", "evidence_ref"):
            try:
                _id(getattr(self, field), field)
            except ValueError as exc:
                raise MemoryError(str(exc)) from exc
        if self.kind not in ("episodic", "semantic", "procedural", "project"):
            raise MemoryError("unsupported memory kind")
        if self.source_kind not in ("owner", "tool", "retrieval", "observation", "model"):
            raise MemoryError("unsupported provenance kind")
        if not isinstance(self.content, str) or not self.content.strip():
            raise MemoryError("memory content must be nonempty text")
        if len(self.content.encode("utf-8")) > 16_384:
            raise MemoryError("memory content exceeds 16 KiB")
        if type(self.confidence) not in (float, int) or not math.isfinite(self.confidence):
            raise MemoryError("confidence must be finite numeric data")
        if not 0.0 <= self.confidence <= 1.0:
            raise MemoryError("confidence outside [0, 1]")
        try:
            _uint(self.observed_at, "observed_at")
            if self.expires_at is not None:
                _uint(self.expires_at, "expires_at")
        except ValueError as exc:
            raise MemoryError(str(exc)) from exc
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise MemoryError("expiry must follow observation")
        if self.corrects is not None:
            try:
                _id(self.corrects, "corrects")
            except ValueError as exc:
                raise MemoryError(str(exc)) from exc
            if self.corrects == self.memory_id:
                raise MemoryError("self correction refused")

    def encode(self) -> str:
        self.validate()
        return _json({"schema": SCHEMA, **asdict(self)})


@dataclass(frozen=True)
class MemoryHit:
    record: MemoryRecord
    acceptance_ref: str
    # Never set this True from persisted memory, approval or model output.
    canonical_external_truth: bool = False


@dataclass(frozen=True)
class LiveFact:
    memory_id: str
    live_source_id: str
    verified_at: int
    # Ephemeral host-verified result, never persisted as a memory authority.
    verified_external: bool = True


def _decode(raw: str, checksum: str) -> MemoryRecord:
    if _digest(raw) != checksum:
        raise MemoryError("memory content hash mismatch")
    try:
        data = json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(MemoryError(value)))
        if not isinstance(data, dict) or set(data) != {
            "schema", "memory_id", "kind", "content", "source_id", "source_kind",
            "evidence_ref", "confidence", "observed_at", "expires_at", "corrects",
        } or data.pop("schema") != SCHEMA:
            raise MemoryError("incompatible memory schema")
        record = MemoryRecord(**data)
        record.validate()
        if record.encode() != raw:
            raise MemoryError("noncanonical memory encoding")
        return record
    except (TypeError, KeyError, StateError, json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise MemoryError("malformed stored memory") from exc


class MemoryStore:
    """Atomic append-only record/acceptance ledger, separate from TaskStore state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.is_symlink() or not self.path.parent.is_dir() or self.path.is_dir():
            raise MemoryError("memory path must be a local regular file")
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
                if version not in ((0, STORAGE_VERSION) if initialize else (STORAGE_VERSION,)):
                    raise MemoryError("unsupported durable memory storage version")
                if initialize and version == 0:
                    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone():
                        raise MemoryError("unversioned memory database cannot be claimed")
                    db.execute(
                        "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL, "
                        "digest TEXT NOT NULL, corrects TEXT UNIQUE, sequence INTEGER UNIQUE NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE acceptance (memory_id TEXT PRIMARY KEY, approval_ref TEXT NOT NULL, "
                        "FOREIGN KEY(memory_id) REFERENCES memories(memory_id))"
                    )
                    db.execute("PRAGMA user_version=1")
                for table in ("memories", "acceptance"):
                    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                  (table,)).fetchone() is None:
                        raise MemoryError("memory storage table missing")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def _read(db: sqlite3.Connection, memory_id: str) -> MemoryRecord:
        row = db.execute("SELECT snapshot, digest, corrects FROM memories WHERE memory_id=?",
                         (memory_id,)).fetchone()
        if row is None:
            raise MemoryError("unknown memory_id")
        result = _decode(row[0], row[1])
        if result.corrects != row[2]:
            raise MemoryError("memory correction metadata mismatch")
        if result.memory_id != memory_id:
            raise MemoryError("memory identity mismatch")
        return result

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        if not isinstance(record, MemoryRecord):
            raise MemoryError("memory must be a typed record")
        serialized = record.encode()
        with self._tx() as db:
            if record.corrects is not None:
                parent = self._read(db, record.corrects)
                if parent.kind != record.kind:
                    raise MemoryError("correction cannot rewrite memory class")
                if record.observed_at < parent.observed_at:
                    raise MemoryError("correction precedes original observation")
                if db.execute("SELECT 1 FROM memories WHERE corrects=?",
                              (record.corrects,)).fetchone():
                    raise MemoryError("memory has already been corrected")
            if db.execute("SELECT 1 FROM memories WHERE memory_id=?",
                          (record.memory_id,)).fetchone():
                raise MemoryError("duplicate memory_id")
            next_sequence = db.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM memories").fetchone()[0]
            db.execute(
                "INSERT INTO memories(memory_id, snapshot, digest, corrects, sequence) "
                "VALUES(?, ?, ?, ?, ?)",
                (record.memory_id, serialized, _digest(serialized), record.corrects, next_sequence),
            )
        return record

    def accept(self, memory_id: str, *, approval_ref: str, now: int) -> None:
        """Host-controlled approval of a recollection, never external fact authority."""
        try:
            _id(memory_id, "memory_id")
            _id(approval_ref, "approval_ref")
            _uint(now, "now")
        except ValueError as exc:
            raise MemoryError(str(exc)) from exc
        with self._tx() as db:
            record = self._read(db, memory_id)
            if (record.expires_at is not None and record.expires_at <= now) or record.observed_at > now:
                raise MemoryError("stale or future memory cannot be accepted")
            if db.execute("SELECT 1 FROM memories WHERE corrects=?", (memory_id,)).fetchone():
                raise MemoryError("superseded memory cannot be accepted")
            if db.execute("SELECT 1 FROM acceptance WHERE memory_id=?", (memory_id,)).fetchone():
                raise MemoryError("already accepted memory")
            db.execute("INSERT INTO acceptance(memory_id, approval_ref) VALUES (?, ?)",
                       (memory_id, approval_ref))

    def retrieve(
        self, *, now: int, query: str = "", kind: MemoryKind | None = None, limit: int = 20
    ) -> tuple[MemoryHit, ...]:
        """Only accepted, unexpired, not-corrected memory; never privileged facts."""
        try:
            _uint(now, "now")
        except ValueError as exc:
            raise MemoryError(str(exc)) from exc
        if type(limit) is not int or not (1 <= limit <= 100):
            raise MemoryError("memory limit must be in [1, 100]")
        if not isinstance(query, str) or len(query) > 512:
            raise MemoryError("query must be bounded text")
        if kind is not None and kind not in ("episodic", "semantic", "procedural", "project"):
            raise MemoryError("unsupported memory filter")
        with self._tx() as db:
            entries = db.execute(
                "SELECT m.memory_id, m.snapshot, m.digest, m.corrects, a.approval_ref, "
                "(SELECT 1 FROM memories WHERE corrects=m.memory_id) AS corrected "
                "FROM memories AS m LEFT JOIN acceptance AS a ON a.memory_id=m.memory_id "
                "ORDER BY m.sequence"
            ).fetchall()
            matches = []
            for memory_id, raw, checksum, corrects, approval_ref, corrected in entries:
                record = _decode(raw, checksum)
                if record.memory_id != memory_id or record.corrects != corrects:
                    raise MemoryError("memory identity/correction metadata mismatch")
                if approval_ref is not None:
                    try:
                        _id(approval_ref, "approval_ref")
                    except ValueError as exc:
                        raise MemoryError("invalid accepted memory authority") from exc
                if approval_ref is None or corrected:
                    continue
                if record.observed_at > now or (record.expires_at is not None and record.expires_at <= now):
                    continue
                if kind is not None and record.kind != kind:
                    continue
                if query.casefold() not in record.content.casefold():
                    continue
                matches.append(MemoryHit(record, approval_ref))
            matches.sort(key=lambda hit: (-hit.record.confidence, -hit.record.observed_at,
                                          hit.record.memory_id))
            return tuple(matches[:limit])


def as_context_entries(hits: tuple[MemoryHit, ...]) -> tuple[ContextEntry, ...]:
    """Memory enters bounded context as untrusted retrieval, never owner/system."""
    entries = []
    for hit in hits:
        if not isinstance(hit, MemoryHit) or hit.canonical_external_truth:
            raise MemoryError("memory hit cannot assert external authority")
        hit.record.validate()
        try:
            _id(hit.acceptance_ref, "acceptance_ref")
        except ValueError as exc:
            raise MemoryError("invalid memory approval") from exc
        entries.append(ContextEntry(
            entry_id=f"memory:{hit.record.memory_id}", source_id=hit.record.source_id,
            source_kind="retrieval", recency=hit.record.observed_at,
            priority=round(hit.record.confidence * 100),
            content=f"[{hit.record.kind}; evidence={hit.record.evidence_ref}] {hit.record.content}",
            critical=False,
        ))
    return tuple(entries)


def verify_live_external_fact(
    hit: MemoryHit, *, checked_at: int,
    verifier: Callable[[str, str, str], bool],
) -> LiveFact:
    """Ephemeral host-only verification. Never cache truth in SQLite.

    `verifier` must be supplied by the trusted external-source adapter, never
    by model/retrieved text. The memory's own approval/evidence is insufficient.
    """
    if not isinstance(hit, MemoryHit) or hit.canonical_external_truth:
        raise MemoryError("forged privileged memory hit")
    try:
        _uint(checked_at, "checked_at")
    except ValueError as exc:
        raise MemoryError(str(exc)) from exc
    hit.record.validate()
    try:
        _id(hit.acceptance_ref, "acceptance_ref")
    except ValueError as exc:
        raise MemoryError("invalid memory approval") from exc
    if hit.record.observed_at > checked_at or (
        hit.record.expires_at is not None and checked_at >= hit.record.expires_at
    ):
        raise MemoryError("stale or future memory cannot verify as live")
    try:
        verified = verifier(hit.record.source_id, hit.record.evidence_ref, hit.record.content)
    except Exception as exc:
        raise MemoryError("external verifier failed") from exc
    if verified is not True:
        raise MemoryError("live external verification required")
    return LiveFact(hit.record.memory_id, hit.record.source_id, checked_at)
