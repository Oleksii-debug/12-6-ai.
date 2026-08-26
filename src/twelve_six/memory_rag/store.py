from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .contracts import MemoryItem, MemoryStoreKind, Provenance, VerificationState

_TABLES = {kind: kind.value for kind in MemoryStoreKind}


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
    """Five physically separate SQLite tables with a shared deterministic contract."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
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
                    invalidated INTEGER NOT NULL DEFAULT 0
                )"""
            )
        self.connection.commit()

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
        missing = [old_id for old_id in supersedes_tuple if self._locate(old_id) is None]
        if missing:
            raise KeyError(f"superseded memory does not exist: {missing[0]}")
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
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
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
        for old_id in supersedes_tuple:
            old_store, old_row = self._locate(old_id)  # validated before insert
            links = sorted(set(json.loads(old_row["superseded_by"])) | {memory_id})
            self.connection.execute(
                (
                    f"UPDATE {_TABLES[old_store]} "
                    "SET superseded_by=?, verification=? WHERE memory_id=?"
                ),
                (json.dumps(links), VerificationState.SUPERSEDED.value, old_id),
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
        return MemoryItem(
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
        store, _ = located
        self.connection.execute(f"DELETE FROM {_TABLES[store]} WHERE memory_id=?", (memory_id,))
        for table in _TABLES.values():
            rows = self.connection.execute(
                f"SELECT memory_id, supersedes, superseded_by FROM {table}"
            ).fetchall()
            for row in rows:
                supersedes = [x for x in json.loads(row["supersedes"]) if x != memory_id]
                superseded_by = [
                    x for x in json.loads(row["superseded_by"]) if x != memory_id
                ]
                self.connection.execute(
                    f"UPDATE {table} SET supersedes=?, superseded_by=? WHERE memory_id=?",
                    (json.dumps(supersedes), json.dumps(superseded_by), row["memory_id"]),
                )
        self.connection.commit()

    def table_counts(self) -> dict[MemoryStoreKind, int]:
        return {
            store: self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for store, table in _TABLES.items()
        }
