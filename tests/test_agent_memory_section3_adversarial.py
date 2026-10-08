"""Plan 5 Section 3 negative/corrupt/recovery/reference boundary tests."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from twelve_six_agent_runtime.memory import (
    MemoryError, MemoryHit, MemoryRecord, MemoryStore, as_context_entries,
    verify_live_external_fact,
)


def record(id="memory", *, confidence=0.8, corrects=None):
    return MemoryRecord(id, "semantic", "verified later", "source", "model", "evidence", confidence,
                        100, None, corrects)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.7"])
def test_rejects_adversarial_confidence_before_storage(tmp_path, bad):
    store = MemoryStore(tmp_path / "m.db")
    with pytest.raises(MemoryError):
        store.remember(record(confidence=bad))
    assert store.retrieve(now=110) == ()


@pytest.mark.parametrize("bad", [None, "", " ", 0])
def test_rejects_invalid_evidence_provenance_and_identity(tmp_path, bad):
    store = MemoryStore(tmp_path / "m.db")
    with pytest.raises(MemoryError):
        store.remember(MemoryRecord("memory", "semantic", "content", "source", "model",
                                    bad, 1.0, 100))
    with pytest.raises(MemoryError):
        store.remember(MemoryRecord("memory", bad, "content", "source", "model",
                                    "evidence", 1.0, 100))


def test_duplicate_ids_and_missing_correction_fail_closed(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    r = record()
    s.remember(r)
    with pytest.raises(MemoryError, match="duplicate"):
        s.remember(r)
    with pytest.raises(MemoryError, match="unknown"):
        s.remember(record("fix", corrects="missing"))
    assert not s.retrieve(now=110)


def test_checksum_tamper_and_metadata_forgery_fail_closed(tmp_path):
    path = tmp_path / "memory.db"
    s = MemoryStore(path)
    s.remember(record())
    s.accept("memory", approval_ref="receipt", now=105)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE memories SET snapshot='{}' WHERE memory_id='memory'")
    with pytest.raises(MemoryError, match="hash mismatch"):
        MemoryStore(path).retrieve(now=110)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE memories SET snapshot=(SELECT snapshot FROM memories WHERE memory_id='memory')")
        con.execute("UPDATE memories SET digest='bad' WHERE memory_id='memory'")
    with pytest.raises(MemoryError, match="hash mismatch"):
        s.retrieve(now=110)


def test_metadata_correction_and_approval_tamper_rejected(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    s.remember(record())
    s.accept("memory", approval_ref="receipt", now=105)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE acceptance SET approval_ref='' WHERE memory_id='memory'")
    with pytest.raises(MemoryError, match="invalid accepted"):
        s.retrieve(now=110)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE acceptance SET approval_ref='receipt' WHERE memory_id='memory'")
        con.execute("UPDATE memories SET corrects='forged' WHERE memory_id='memory'")
    with pytest.raises(MemoryError, match="metadata mismatch"):
        s.retrieve(now=110)


def test_storage_version_rollback_and_symlink_denied(tmp_path):
    p = tmp_path / "m.db"
    s = MemoryStore(p)
    s.remember(record())
    with sqlite3.connect(p) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(MemoryError, match="unsupported"):
        MemoryStore(p)
    with pytest.raises(MemoryError, match="unsupported"):
        s.retrieve(now=110)
    (tmp_path / "alias.db").symlink_to(p)
    with pytest.raises(MemoryError, match="regular"):
        MemoryStore(tmp_path / "alias.db")


def test_concurrent_duplicate_writers_only_one_identity_wins(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    def write(_):
        try:
            s.remember(record())
        except (MemoryError, sqlite3.IntegrityError):
            return "duplicate"
        return "ok"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, (0, 1))) == ["duplicate", "ok"]
    s.accept("memory", approval_ref="receipt", now=105)
    assert len(s.retrieve(now=110)) == 1


def test_new_process_replay_without_model_or_production_tools(tmp_path):
    p = tmp_path / "m.db"
    s = MemoryStore(p)
    s.remember(record())
    s.accept("memory", approval_ref="approval", now=105)
    code = (
        "import json,sys;from twelve_six_agent_runtime.memory import MemoryStore;"
        "h=MemoryStore(sys.argv[1]).retrieve(now=110);"
        "print(json.dumps([(x.record.memory_id,x.record.kind,x.acceptance_ref,"
        "x.canonical_external_truth) for x in h]))"
    )
    result = subprocess.run([sys.executable, "-c", code, str(p)], check=True,
                            capture_output=True, text=True, timeout=20)
    assert json.loads(result.stdout) == [["memory", "semantic", "approval", False]]


def test_forged_privileged_hit_and_invalid_approval_rejected():
    original = record()
    for hit in (MemoryHit(original, "receipt", canonical_external_truth=True),
                MemoryHit(original, "")):
        with pytest.raises(MemoryError):
            as_context_entries((hit,))
        with pytest.raises(MemoryError):
            verify_live_external_fact(hit, checked_at=105, verifier=lambda *_: True)
