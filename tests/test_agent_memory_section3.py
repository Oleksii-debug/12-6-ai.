import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from twelve_six_agent_runtime.memory import MemoryError, MemoryRecord, MemoryStore


def entry(mid, kind="episodic", **kwargs):
    return MemoryRecord(mid, kind, kwargs.pop("content", "a recorded fact"),
                        kwargs.pop("source_id", "source-1"),
                        kwargs.pop("source_kind", "external"),
                        kwargs.pop("evidence_id", "recorded-evidence"),
                        kwargs.pop("confidence_ppm", 950000),
                        kwargs.pop("created_at", 100), **kwargs)


def test_four_kinds_are_separate_and_source_evidence_survive_restart(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    for idx, kind in enumerate(("episodic", "semantic", "procedural", "project")):
        assert store.append(entry(str(idx), kind=kind), expected_revision=idx) == idx + 1
    reopened = MemoryStore(store.path)
    assert reopened.revision() == 4
    assert [h.record.kind for h in reopened.retrieve(now=120)] == [
        "episodic", "semantic", "procedural", "project"]
    assert reopened.retrieve(now=120, kind="project")[0].record.evidence_id == "recorded-evidence"
    assert reopened.retrieve(now=120, kind="semantic")[0].authority == "memory_only"
    assert reopened.retrieve(now=120)[0].requires_live_verification


def test_expiry_search_confidence_priority_and_future_time(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    store.append(entry("old", content="Paris old", expires_at=111), expected_revision=0)
    store.append(entry("low", content="Paris low", confidence_ppm=200000), expected_revision=1)
    store.append(entry("high", content="Paris high", confidence_ppm=900000, created_at=120), expected_revision=2)
    assert [h.record.memory_id for h in store.retrieve(now=125, query="PARIS")] == ["high", "low"]
    assert [h.record.memory_id for h in store.retrieve(now=110, query="Paris")] == ["old", "low"]
    assert store.retrieve(now=99) == ()


def test_correction_and_retraction_are_append_only(tmp_path):
    db = tmp_path / "memory.sqlite"
    store = MemoryStore(db)
    store.append(entry("a", content="wrong"), expected_revision=0)
    store.append(entry("b", content="corrected", action="correct", replaces_id="a", created_at=101), expected_revision=1)
    store.append(entry("c", content="", action="retract", replaces_id="b", created_at=102), expected_revision=2)
    assert MemoryStore(db).retrieve(now=103) == ()
    assert [r.memory_id for r in MemoryStore(db).history()] == ["a", "b", "c"]
    with pytest.raises(MemoryError):
        store.append(entry("d", action="correct", replaces_id="a"), expected_revision=3)


def test_negative_types_identity_provenance_and_cas(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    bad = [entry("x", confidence_ppm=True), entry("x", confidence_ppm=1000001),
           entry("x", evidence_id=""), entry("x", source_kind="unknown"),
           entry("x", kind="bad"), entry("x", expires_at=100),
           entry("x", action="correct"), entry("x", content=""),
           entry("x", replaces_id="nonexistent")]
    for record in bad:
        with pytest.raises(MemoryError):
            store.append(record, expected_revision=0)
    assert store.append(entry("good"), expected_revision=0) == 1
    with pytest.raises(MemoryError, match="stale"):
        store.append(entry("new"), expected_revision=0)
    with pytest.raises(MemoryError, match="duplicate"):
        store.append(entry("good"), expected_revision=1)
    with pytest.raises(MemoryError):
        store.retrieve(now=True)


def test_corrupt_stored_record_and_version_fail_closed(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    store.append(entry("x"), expected_revision=0)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE records SET payload='{}' WHERE memory_id='x'")
    with pytest.raises(MemoryError, match="digest"):
        MemoryStore(store.path).retrieve(now=120)
    with sqlite3.connect(store.path) as db:
        db.execute("PRAGMA user_version=88")
    with pytest.raises(MemoryError, match="unsupported"):
        MemoryStore(store.path)


def test_simultaneous_writers_have_single_cas_winner(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    def write(mid):
        try:
            store.append(entry(mid), expected_revision=0)
        except MemoryError:
            return "rejected"
        return "accepted"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, ("a", "b"))) == ["accepted", "rejected"]
    assert len(MemoryStore(store.path).history()) == 1
