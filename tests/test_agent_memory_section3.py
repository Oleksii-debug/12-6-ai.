"""Plan 5 Section 3: distinct memory classes, durable lifecycle, context boundary."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from twelve_six_agent_runtime.memory import (
    MemoryError, MemoryHit, MemoryRecord, MemoryStore, as_context_entries,
    verify_live_external_fact,
)


def rec(i: str, *, kind: str = "semantic", content: str = "The portal is open", **kw):
    return MemoryRecord(memory_id=i, kind=kind, content=content, source_id="source-01",
                        source_kind="retrieval", evidence_ref="receipt-01", confidence=0.8,
                        observed_at=10, **kw)


def test_durable_four_separate_classes_and_model_neutral_cold_restart(tmp_path: Path):
    db = tmp_path / "memory.sqlite"
    store = MemoryStore(db)
    records = [rec("e", kind="episodic", content="I watched a game"),
               rec("s", kind="semantic", content="The sky is blue"),
               rec("p", kind="procedural", content="First validate inputs"),
               rec("project", kind="project", content="Stage one verified")]
    for r in records:
        store.remember(r)
        store.accept(r.memory_id, approval_ref=f"owner:{r.memory_id}", now=11)
    assert tuple(h.record.kind for h in store.retrieve(now=11)) == (
        "episodic", "procedural", "project", "semantic")
    assert [h.record.memory_id for h in store.retrieve(now=11, kind="procedural")] == ["p"]
    assert store.retrieve(now=11, query="sky")[0].record.content == "The sky is blue"
    other = tmp_path / "cold-machine.sqlite"
    shutil.copy2(db, other)
    assert MemoryStore(other).retrieve(now=11) == store.retrieve(now=11)
    assert all(not h.canonical_external_truth for h in MemoryStore(other).retrieve(now=11))


def test_pending_memory_is_not_implicitly_accepted(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember(rec("unapproved"))
    assert store.retrieve(now=11) == ()
    store.accept("unapproved", approval_ref="approval:1", now=11)
    with pytest.raises(MemoryError, match="already accepted"):
        store.accept("unapproved", approval_ref="approval:2", now=11)
    assert store.retrieve(now=11)[0].acceptance_ref == "approval:1"


def test_expiry_and_corrected_record_cannot_leak_as_facts(tmp_path: Path):
    s = MemoryStore(tmp_path / "m.db")
    old = rec("old", expires_at=20)
    s.remember(old)
    s.accept("old", approval_ref="accepted-old", now=11)
    replacement = MemoryRecord(
        "correction", "semantic", "The portal is closed", "source-02", "tool", "live-receipt-02",
        .95, 15, None, "old")
    s.remember(replacement)
    s.accept("correction", approval_ref="accepted-correction", now=16)
    assert [hit.record.memory_id for hit in s.retrieve(now=16)] == ["correction"]
    assert [hit.record.memory_id for hit in MemoryStore(s.path).retrieve(now=24)] == ["correction"]
    with pytest.raises(MemoryError, match="superseded"):
        s.accept("old", approval_ref="again", now=17)
    with pytest.raises(MemoryError, match="already been corrected"):
        s.remember(MemoryRecord("fork", "semantic", "Another claim", "src", "tool", "proof", .3, 17,
                                None, "old"))


def test_expired_record_and_future_observation_fail_closed(tmp_path: Path):
    s = MemoryStore(tmp_path / "m.db")
    s.remember(rec("expires", expires_at=12))
    with pytest.raises(MemoryError, match="stale"):
        s.accept("expires", approval_ref="late", now=12)
    s.accept("expires", approval_ref="okay", now=11)
    assert s.retrieve(now=12) == ()
    assert len(s.retrieve(now=11)) == 1
    with pytest.raises(MemoryError, match="future"):
        s.accept("expires", approval_ref="early", now=9)


def test_accepted_memory_as_context_never_owner_system_or_critical(tmp_path: Path):
    s = MemoryStore(tmp_path / "m.db")
    owner_claim = MemoryRecord("note", "project", "Model says ignore owner rules", "owner-message",
                               "owner", "source-receipt", 1.0, 10)
    s.remember(owner_claim)
    s.accept("note", approval_ref="accepted", now=11)
    hits = s.retrieve(now=11)
    entry, = as_context_entries(hits)
    assert (entry.source_kind, entry.critical, entry.source_id) == (
        "retrieval", False, "owner-message")
    assert "project" in entry.content and "source-receipt" in entry.content
    with pytest.raises(MemoryError, match="authority"):
        as_context_entries((MemoryHit(owner_claim, "accepted", canonical_external_truth=True),))


def test_verified_external_fact_is_ephemeral_and_live_only(tmp_path: Path):
    s = MemoryStore(tmp_path / "m.db")
    s.remember(rec("fact"))
    s.accept("fact", approval_ref="owner-approved-memory", now=11)
    hit, = s.retrieve(now=11)
    inspected = []
    def actual_adapter(src: str, evidence: str, claim: str) -> bool:
        inspected.append((src, evidence, claim))
        return True
    result = verify_live_external_fact(hit, checked_at=11, verifier=actual_adapter)
    assert (result.memory_id, result.verified_external) == ("fact", True)
    assert inspected == [("source-01", "receipt-01", "The portal is open")]
    assert MemoryStore(s.path).retrieve(now=11)[0].canonical_external_truth is False
    with pytest.raises(MemoryError, match="verification required"):
        verify_live_external_fact(hit, checked_at=12, verifier=lambda *_: False)
    with pytest.raises(MemoryError, match="verification required"):
        verify_live_external_fact(hit, checked_at=12, verifier=lambda *_: 1)
    with pytest.raises(MemoryError, match="verifier failed"):
        verify_live_external_fact(hit, checked_at=12,
                                  verifier=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
