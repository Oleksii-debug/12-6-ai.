"""Evidence-bound self-model and metacognitive advisory (Plan 5 / Section 5).

This ledger has no executor, policy authority, credentials, or permission grant.
A trusted host verifier, never an LLM judgment, must validate each observation.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .task_state import _digest, _id, _json, _pairs, _uint

SCHEMA = "12-6.agent-self-model.v1"
Category = Literal["capability", "limit", "readiness", "error_pattern", "resource"]
Origin = Literal["operator", "tool", "test"]


class SelfModelError(ValueError):
    """Untrusted evidence, corrupt ledger, or stale writer."""


@dataclass(frozen=True)
class SelfObservation:
    observation_id: str
    category: Category
    name: str
    value: str
    source_id: str
    evidence_id: str
    origin: Origin
    observed_at: int
    confidence_ppm: int
    expires_at: int | None = None
    uncertainty: str = ""

    def validate(self) -> None:
        for name in ("observation_id", "name", "source_id", "evidence_id"):
            _id(getattr(self, name), name)
        if self.category not in ("capability", "limit", "readiness", "error_pattern", "resource"):
            raise SelfModelError("invalid category")
        if self.origin not in ("operator", "tool", "test"):
            raise SelfModelError("model/memory output is not a trusted observation")
        if not isinstance(self.value, str) or not self.value or len(self.value) > 512:
            raise SelfModelError("invalid bounded observation value")
        if self.category == "readiness" and self.value not in (
            "unknown", "ready", "limited", "unavailable"
        ):
            raise SelfModelError("invalid readiness state")
        _uint(self.observed_at, "observed_at")
        _uint(self.confidence_ppm, "confidence_ppm")
        if self.confidence_ppm > 1_000_000:
            raise SelfModelError("confidence above one")
        if self.expires_at is not None:
            _uint(self.expires_at, "expires_at")
            if self.expires_at <= self.observed_at:
                raise SelfModelError("expiry must follow observation")
        if not isinstance(self.uncertainty, str) or len(self.uncertainty) > 1024:
            raise SelfModelError("invalid uncertainty")
        if self.confidence_ppm < 1_000_000 and not self.uncertainty:
            raise SelfModelError("imperfect confidence needs uncertainty evidence")


def _decode(raw: str, digest: str) -> SelfObservation:
    if _digest(raw) != digest:
        raise SelfModelError("observation digest mismatch")
    try:
        payload = json.loads(
            raw, object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(SelfModelError(value)),
        )
        expected = {field for field in SelfObservation.__dataclass_fields__}
        if not isinstance(payload, dict) or set(payload) != expected | {"schema"}:
            raise SelfModelError("invalid observation schema")
        if payload.pop("schema") != SCHEMA:
            raise SelfModelError("incompatible observation schema")
        item = SelfObservation(**payload)
        item.validate()
        if _json({"schema": SCHEMA, **asdict(item)}) != raw:
            raise SelfModelError("noncanonical observation bytes")
        return item
    except (TypeError, KeyError, json.JSONDecodeError, RecursionError) as exc:
        raise SelfModelError("malformed observation") from exc


class SelfModelStore:
    """Versioned SQLite FULL-sync evidence ledger; independent of model provider."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink() or self.path.is_dir() or not self.path.parent.is_dir():
            raise SelfModelError("unsafe self-model path")
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
                    raise SelfModelError("incompatible self-model storage version")
                if initialize and version == 0:
                    if db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                        raise SelfModelError("cannot claim existing unversioned tables")
                    db.execute("CREATE TABLE meta (revision INTEGER NOT NULL)")
                    db.execute("INSERT INTO meta VALUES (0)")
                    db.execute(
                        "CREATE TABLE observations (sequence INTEGER PRIMARY KEY, "
                        "observation_id TEXT NOT NULL UNIQUE, raw TEXT NOT NULL, "
                        "digest TEXT NOT NULL)"
                    )
                    db.execute("PRAGMA user_version=1")
                if not db.execute("SELECT name FROM sqlite_master WHERE name='meta'").fetchone():
                    raise SelfModelError("missing self-model meta")
                if not db.execute(
                    "SELECT name FROM sqlite_master WHERE name='observations'"
                ).fetchone():
                    raise SelfModelError("missing observations")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def _history(db: sqlite3.Connection) -> tuple[SelfObservation, ...]:
        rows = db.execute(
            "SELECT sequence, observation_id, raw, digest "
            "FROM observations ORDER BY sequence"
        ).fetchall()
        meta = db.execute("SELECT revision FROM meta").fetchall()
        if len(meta) != 1 or type(meta[0][0]) is not int or meta[0][0] != len(rows):
            raise SelfModelError("self-model revision mismatch")
        seen: set[str] = set()
        latest: dict[tuple[str, str], int] = {}
        records: list[SelfObservation] = []
        for sequence, ident, raw, digest in rows:
            item = _decode(raw, digest)
            if sequence != len(records) + 1 or item.observation_id != ident or ident in seen:
                raise SelfModelError("self-model event lineage mismatch")
            key = (item.category, item.name)
            if item.observed_at < latest.get(key, 0):
                raise SelfModelError("self-model event timestamp regression")
            latest[key] = item.observed_at
            seen.add(ident)
            records.append(item)
        return tuple(records)

    def history(self) -> tuple[SelfObservation, ...]:
        with self._tx() as db:
            return self._history(db)

    def append(
        self, observation: SelfObservation, *, expected_revision: int,
        verifier: Callable[[SelfObservation], bool],
    ) -> int:
        if not isinstance(observation, SelfObservation):
            raise SelfModelError("invalid observation")
        observation.validate()
        _uint(expected_revision, "expected_revision")
        if not callable(verifier) or verifier(observation) is not True:
            raise SelfModelError("trusted evidence verifier rejected observation")
        with self._tx() as db:
            history = self._history(db)
            if expected_revision != len(history):
                raise SelfModelError("stale self-model revision")
            if any(item.observation_id == observation.observation_id for item in history):
                raise SelfModelError("duplicate evidence identity")
            if any(
                item.category == observation.category and item.name == observation.name
                and observation.observed_at < item.observed_at for item in history
            ):
                raise SelfModelError("stale observation timestamp")
            raw = _json({"schema": SCHEMA, **asdict(observation)})
            db.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?)",
                (len(history) + 1, observation.observation_id, raw, _digest(raw)),
            )
            db.execute("UPDATE meta SET revision=?", (len(history) + 1,))
            return len(history) + 1

    def profile(self, *, now: int, min_confidence_ppm: int = 800_000) -> dict[str, object]:
        """Return advisory assessment; never an execution authorization."""
        _uint(now, "now")
        _uint(min_confidence_ppm, "min_confidence_ppm")
        if min_confidence_ppm > 1_000_000:
            raise SelfModelError("invalid confidence threshold")
        items = self.history()
        latest: dict[tuple[str, str], SelfObservation] = {}
        for item in items:
            if item.observed_at <= now:
                latest[(item.category, item.name)] = item
        categories: dict[str, dict[str, object]] = {
            cat: {} for cat in ("capability", "limit", "readiness", "error_pattern", "resource")
        }
        for (cat, name), item in sorted(latest.items()):
            expired = item.expires_at is not None and now >= item.expires_at
            uncertain = item.confidence_ppm < min_confidence_ppm
            categories[cat][name] = {
                "value": None if expired else item.value,
                "status": "stale" if expired else ("verify" if uncertain else "evidenced"),
                "confidence_ppm": item.confidence_ppm,
                "uncertainty": item.uncertainty,
                "source_id": item.source_id,
                "evidence_id": item.evidence_id,
                "observed_at": item.observed_at,
            }
        return {
            "schema": SCHEMA, "revision": len(items), "categories": categories,
            "advisory_only": True, "grants_permissions": False,
        }
