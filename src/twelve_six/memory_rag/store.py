from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .contracts import MemoryItem, MemoryStoreKind, Provenance, VerificationState

_TABLES = {kind: kind.value for kind in MemoryStoreKind}


class MemoryIntegrityError(RuntimeError):
    """Stored content/provenance/version does not match its recorded digest."""


def _canonical_hash(*, content: str, provenance: Provenance, version: int) -> str:
    payload = {
        "content": content,
        "provenance": {
            "locator": provenance.locator,
            "source_id": provenance.source_id,
            "source_type": provenance.source_type,
            "source_version": provenance.source_version,
        },
        "version": version,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class MemoryDatabase:
    """Five logical memory stores in one SQLite database, with deterministic contracts."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        for table in _TABLES.values():
            self.connection.execute(
                f"""CREATE TABLE IF NOT EXISTS {table} (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    locator TEXT,
                    timestamp TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    verification TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    claim_key TEXT,
                    claim_value TEXT,
                    supersedes TEXT NOT NULL,
                    superseded_by TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    invalidated INTEGER NOT NULL DEFAULT 0,
                    supersession_base_verification TEXT
                )"""
            )
            self._ensure_column(table, "supersession_base_verification", "TEXT")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def compute_content_hash(*, content: str, provenance: Provenance, version: int) -> str:
        return _canonical_hash(content=content, provenance=provenance, version=version)

    def add(
        self,
        *,
        memory_id: str,
        store: MemoryStoreKind,
        content: str,
        provenance: Provenance,
        timestamp: datetime,
        version: int,
        confidence: float,
        verification: VerificationState,
        claim_key: str | None = None,
        claim_value: str | None = None,
        supersedes: Iterable[str] = (),
        metadata: dict[str, str] | None = None,
    ) -> MemoryItem:
        supersedes_tuple = tuple(sorted(set(supersedes)))
        if self._locate(memory_id) is not None:
            raise ValueError(f"memory_id already exists: {memory_id}")
        located_predecessors: dict[str, tuple[MemoryStoreKind, sqlite3.Row]] = {}
        for old_id in supersedes_tuple:
            located = self._locate(old_id)
            if located is None:
                raise KeyError(f"superseded memory does not exist: {old_id}")
            located_predecessors[old_id] = located

        item = MemoryItem(
            memory_id=memory_id,
            store=store,
            content=content,
            provenance=provenance,
            timestamp=timestamp,
            version=version,
            confidence=confidence,
            verification=verification,
            content_hash=_canonical_hash(content=content, provenance=provenance, version=version),
            claim_key=claim_key,
            claim_value=claim_value,
            supersedes=supersedes_tuple,
            metadata=metadata or {},
        )
        table = _TABLES[store]
        self.connection.execute(
            f"""INSERT INTO {table} (
                memory_id, content, source_type, source_id, source_version, locator,
                timestamp, version, confidence, verification, content_hash,
                claim_key, claim_value, supersedes, superseded_by, metadata,
                invalidated, supersession_base_verification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (
                item.memory_id,
                item.content,
                provenance.source_type,
                provenance.source_id,
                provenance.source_version,
                provenance.locator,
                timestamp.isoformat(),
                version,
                confidence,
                verification.value,
                item.content_hash,
                claim_key,
                claim_value,
                json.dumps(item.supersedes),
                "[]",
                json.dumps(dict(item.metadata), sort_keys=True),
            ),
        )
        for old_id, (old_store, old_row) in located_predecessors.items():
            links = sorted(set(json.loads(old_row["superseded_by"])) | {memory_id})
            base_state = old_row["supersession_base_verification"]
            if base_state is None and old_row["verification"] != VerificationState.SUPERSEDED.value:
                base_state = old_row["verification"]
            self.connection.execute(
                (
                    f"UPDATE {_TABLES[old_store]} SET superseded_by=?, verification=?, "
                    "supersession_base_verification=? WHERE memory_id=?"
                ),
                (
                    json.dumps(links),
                    VerificationState.SUPERSEDED.value,
                    base_state,
                    old_id,
                ),
            )
        self.connection.commit()
        return self.get(memory_id, include_inactive=True)

    def _locate(self, memory_id: str) -> tuple[MemoryStoreKind, sqlite3.Row] | None:
        for store, table in _TABLES.items():
            row = self.connection.execute(
                f"SELECT * FROM {table} WHERE memory_id=?", (memory_id,)
            ).fetchone()
            if row is not None:
                return store, row
        return None

    @staticmethod
    def _decode(store: MemoryStoreKind, row: sqlite3.Row) -> MemoryItem:
        item = MemoryItem(
            memory_id=row["memory_id"],
            store=store,
            content=row["content"],
            provenance=Provenance(
                source_type=row["source_type"],
                source_id=row["source_id"],
                source_version=row["source_version"],
                locator=row["locator"],
            ),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            version=row["version"],
            confidence=row["confidence"],
            verification=VerificationState(row["verification"]),
            content_hash=row["content_hash"],
            claim_key=row["claim_key"],
            claim_value=row["claim_value"],
            supersedes=tuple(json.loads(row["supersedes"])),
            superseded_by=tuple(json.loads(row["superseded_by"])),
            metadata=json.loads(row["metadata"]),
        )
        expected = _canonical_hash(
            content=item.content,
            provenance=item.provenance,
            version=item.version,
        )
        if not hmac.compare_digest(item.content_hash, expected):
            raise MemoryIntegrityError(f"content/provenance hash mismatch: {item.memory_id}")
        return item

    def verify_integrity(self, memory_id: str) -> bool:
        located = self._locate(memory_id)
        if located is None:
            raise KeyError(memory_id)
        self._decode(*located)
        return True

    def get(self, memory_id: str, *, include_inactive: bool = False) -> MemoryItem:
        located = self._locate(memory_id)
        if located is None:
            raise KeyError(memory_id)
        store, row = located
        if row["invalidated"] and not include_inactive:
            raise KeyError(memory_id)
        return self._decode(store, row)

    def active_items(
        self, stores: Iterable[MemoryStoreKind] | None = None
    ) -> tuple[MemoryItem, ...]:
        selected = tuple(stores) if stores is not None else tuple(MemoryStoreKind)
        items: list[MemoryItem] = []
        for store in selected:
            rows = self.connection.execute(
                (
                    f"SELECT * FROM {_TABLES[store]} "
                    "WHERE invalidated=0 AND verification NOT IN (?, ?)"
                ),
                (VerificationState.REJECTED.value, VerificationState.SUPERSEDED.value),
            ).fetchall()
            items.extend(self._decode(store, row) for row in rows)
        return tuple(sorted(items, key=lambda item: item.memory_id))

    def invalidate(self, memory_id: str) -> None:
        located = self._locate(memory_id)
        if located is None:
            raise KeyError(memory_id)
        store, _ = located
        self.connection.execute(
            f"UPDATE {_TABLES[store]} SET invalidated=1, verification=? WHERE memory_id=?",
            (VerificationState.REJECTED.value, memory_id),
        )
        self.connection.commit()

    def delete(self, memory_id: str) -> None:
        located = self._locate(memory_id)
        if located is None:
            raise KeyError(memory_id)
        store, row = located
        predecessors = tuple(json.loads(row["supersedes"]))
        successors = tuple(json.loads(row["superseded_by"]))

        # Preserve lineage across a deleted middle node by connecting its neighbors.
        # With no remaining successor, restore the predecessor's prior state.
        for predecessor_id in predecessors:
            predecessor_located = self._locate(predecessor_id)
            if predecessor_located is None:
                continue
            predecessor_store, predecessor_row = predecessor_located
            links = set(json.loads(predecessor_row["superseded_by"]))
            links.discard(memory_id)
            links.update(successors)
            verification = predecessor_row["verification"]
            base_state = predecessor_row["supersession_base_verification"]
            if not links and verification == VerificationState.SUPERSEDED.value:
                verification = base_state or VerificationState.UNVERIFIED.value
                base_state = None
            self.connection.execute(
                (
                    f"UPDATE {_TABLES[predecessor_store]} SET superseded_by=?, "
                    "verification=?, supersession_base_verification=? WHERE memory_id=?"
                ),
                (json.dumps(sorted(links)), verification, base_state, predecessor_id),
            )

        for successor_id in successors:
            successor_located = self._locate(successor_id)
            if successor_located is None:
                continue
            successor_store, successor_row = successor_located
            links = set(json.loads(successor_row["supersedes"]))
            links.discard(memory_id)
            links.update(predecessors)
            self.connection.execute(
                f"UPDATE {_TABLES[successor_store]} SET supersedes=? WHERE memory_id=?",
                (json.dumps(sorted(links)), successor_id),
            )

        self.connection.execute(
            f"DELETE FROM {_TABLES[store]} WHERE memory_id=?", (memory_id,)
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def table_counts(self) -> dict[MemoryStoreKind, int]:
        return {
            store: self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for store, table in _TABLES.items()
        }
