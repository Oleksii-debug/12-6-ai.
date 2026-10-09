"""Plan 5 / Section 3: append-only, source-aware durable agent memory.

Memory is recollection, never an authority for current external truth. This
standalone reference store does not grant tools or verify live external facts.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

MemoryKind = Literal["episodic", "semantic", "procedural", "project"]
SourceKind = Literal["owner", "tool", "external", "model"]
Action = Literal["assert", "correct", "retract"]
SCHEMA = "12-6.agent-memory.v1"


class MemoryError(ValueError):
    """Malformed memory, stale writer or compromised durable memory."""


def _id(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and bool(value.strip())


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    kind: MemoryKind
    content: str
    source_id: str
    source_kind: SourceKind
    evidence_id: str
    confidence_ppm: int
    created_at: int
    expires_at: int | None = None
    action: Action = "assert"
    replaces_id: str | None = None

    def validate(self) -> None:
        if not all(_id(x) for x in (self.memory_id, self.source_id, self.evidence_id)):
            raise MemoryError("memory/source/evidence identity required")
        if self.kind not in ("episodic", "semantic", "procedural", "project"):
            raise MemoryError("unknown memory kind")
        if self.source_kind not in ("owner", "tool", "external", "model"):
            raise MemoryError("unknown source kind")
        if not isinstance(self.content, str) or len(self.content.encode("utf-8")) > 16384:
            raise MemoryError("memory content exceeds bound")
        if self.action not in ("assert", "correct", "retract"):
            raise MemoryError("invalid memory action")
        if self.action == "assert" and self.replaces_id is not None:
            raise MemoryError("assert cannot replace an existing memory")
        if self.action != "assert" and not _id(self.replaces_id):
            raise MemoryError("correction needs an existing target")
        if self.action != "retract" and not self.content.strip():
            raise MemoryError("empty assertion")
        if type(self.confidence_ppm) is not int or not 0 <= self.confidence_ppm <= 1_000_000:
            raise MemoryError("invalid confidence")
        if type(self.created_at) is not int or self.created_at < 0:
            raise MemoryError("invalid timestamp")
        if self.expires_at is not None and (
            type(self.expires_at) is not int or self.expires_at <= self.created_at
        ):
            raise MemoryError("invalid expiry")


@dataclass(frozen=True)
class MemoryHit:
    record: MemoryRecord
    authority: Literal["memory_only"] = "memory_only"
    requires_live_verification: bool = True


def _decode(raw: str, digest: str) -> MemoryRecord:
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
        raise MemoryError("memory digest mismatch")
    try:
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise MemoryError("duplicate JSON key")
                result[key] = value
            return result

        value = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(value, dict) or value.pop("schema", None) != SCHEMA:
            raise MemoryError("unsupported memory schema")
        if set(value) != set(MemoryRecord.__dataclass_fields__):
            raise MemoryError("noncanonical memory fields")
        record = MemoryRecord(**value)
        record.validate()
        if _canonical({"schema": SCHEMA, **asdict(record)}) != raw:
            raise MemoryError("noncanonical memory encoding")
        return record
    except (TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise MemoryError("malformed stored memory") from exc


class MemoryStore:
    """Versioned SQLite memory ledger with CAS, corrections, expiry, and readback."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink() or not self.path.parent.is_dir() or self.path.is_dir():
            raise MemoryError("unsafe memory path")
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
                    raise MemoryError("unsupported memory store version")
                if initialize and version == 0:
                    if db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                        raise MemoryError("cannot claim existing unversioned database")
                    db.execute("CREATE TABLE meta (revision INTEGER NOT NULL)")
                    db.execute("INSERT INTO meta VALUES (0)")
                    db.execute("CREATE TABLE records (memory_id TEXT PRIMARY KEY, sequence INTEGER UNIQUE NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL, replaces_id TEXT UNIQUE)")
                    db.execute("PRAGMA user_version=1")
                if not db.execute("SELECT name FROM sqlite_master WHERE name='meta'").fetchone() or not db.execute("SELECT name FROM sqlite_master WHERE name='records'").fetchone():
                    raise MemoryError("incomplete memory tables")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def _rows(db: sqlite3.Connection) -> list[MemoryRecord]:
        rows = db.execute("SELECT memory_id,sequence,payload,digest,replaces_id FROM records ORDER BY sequence").fetchall()
        revision = db.execute("SELECT revision FROM meta").fetchone()
        if revision is None or type(revision[0]) is not int or revision[0] != len(rows):
            raise MemoryError("memory revision/history mismatch")
        seen: dict[str, MemoryRecord] = {}
        replaced: set[str] = set()
        for index, (memory_id, sequence, payload, digest, replaces_id) in enumerate(rows, 1):
            record = _decode(payload, digest)
            if sequence != index or record.memory_id != memory_id or record.replaces_id != replaces_id:
                raise MemoryError("memory lineage mismatch")
            if record.replaces_id is not None:
                prior = seen.get(record.replaces_id)
                if prior is None or prior.kind != record.kind or prior.memory_id in replaced or record.created_at < prior.created_at:
                    raise MemoryError("invalid correction lineage")
                replaced.add(prior.memory_id)
            seen[memory_id] = record
        return list(seen.values())

    def revision(self) -> int:
        with self._tx() as db:
            return len(self._rows(db))

    def append(self, record: MemoryRecord, *, expected_revision: int) -> int:
        if not isinstance(record, MemoryRecord):
            raise MemoryError("invalid memory record")
        record.validate()
        if type(expected_revision) is not int or expected_revision < 0:
            raise MemoryError("invalid expected revision")
        with self._tx() as db:
            prior = self._rows(db)
            if len(prior) != expected_revision:
                raise MemoryError("stale memory writer")
            if record.memory_id in {item.memory_id for item in prior}:
                raise MemoryError("duplicate memory identity")
            if record.replaces_id is not None:
                target = next((item for item in prior if item.memory_id == record.replaces_id), None)
                if target is None or target.kind != record.kind or target.created_at > record.created_at or any(item.replaces_id == target.memory_id for item in prior):
                    raise MemoryError("invalid or already corrected target")
            payload = _canonical({"schema": SCHEMA, **asdict(record)})
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            db.execute("INSERT INTO records VALUES (?,?,?,?,?)", (record.memory_id, len(prior) + 1, payload, digest, record.replaces_id))
            db.execute("UPDATE meta SET revision=?", (len(prior) + 1,))
            return len(prior) + 1

    def history(self) -> tuple[MemoryRecord, ...]:
        with self._tx() as db:
            return tuple(self._rows(db))

    def retrieve(self, *, now: int, kind: MemoryKind | None = None, query: str = "", limit: int = 20) -> tuple[MemoryHit, ...]:
        if type(now) is not int or now < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise MemoryError("invalid retrieval limits")
        if kind is not None and kind not in ("episodic", "semantic", "procedural", "project"):
            raise MemoryError("invalid filter kind")
        if not isinstance(query, str) or len(query) > 256:
            raise MemoryError("invalid search query")
        with self._tx() as db:
            rows = self._rows(db)
        replaced = {record.replaces_id for record in rows if record.replaces_id is not None}
        valid = [record for record in rows if record.memory_id not in replaced and record.action != "retract" and record.created_at <= now and (record.expires_at is None or now < record.expires_at) and (kind is None or kind == record.kind) and query.casefold() in record.content.casefold()]
        valid.sort(key=lambda record: (-record.confidence_ppm, -record.created_at, record.memory_id))
        return tuple(MemoryHit(record) for record in valid[:limit])
