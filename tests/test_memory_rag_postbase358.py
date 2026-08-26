from __future__ import annotations

import pathlib
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone

from twelve_six.memory_rag import (
    EvidenceObject,
    LexicalRetriever,
    MemoryDatabase,
    MemoryIntegrityError,
    MemoryStoreKind,
    MockPostbase255Adapter,
    Provenance,
    VerificationState,
    feed_reasoning,
)

NOW = datetime(2026, 8, 26, 14, 0, tzinfo=timezone.utc)
SRC = Provenance("synthetic_test", "fixture-public-001", "v1", "synthetic://tests/1")


def add(
    db: MemoryDatabase,
    memory_id: str,
    store: MemoryStoreKind,
    content: str,
    **kwargs,
):
    return db.add(
        memory_id=memory_id,
        store=store,
        content=content,
        provenance=kwargs.pop("provenance", SRC),
        timestamp=kwargs.pop("timestamp", NOW),
        version=kwargs.pop("version", 1),
        confidence=kwargs.pop("confidence", 0.9),
        verification=kwargs.pop("verification", VerificationState.VERIFIED),
        **kwargs,
    )


class MemoryRagPostbase358Tests(unittest.TestCase):
    def test_store_provenance_roundtrip_and_integrity(self) -> None:
        db = MemoryDatabase()
        item = add(
            db,
            "p1",
            MemoryStoreKind.RESEARCH_DOCUMENTS,
            "synthetic provenance token",
        )
        loaded = db.get("p1")
        self.assertEqual(loaded.provenance, SRC)
        self.assertEqual(loaded.content_hash, item.content_hash)
        self.assertTrue(db.verify_integrity("p1"))

    def test_provenance_tamper_fails_closed(self) -> None:
        db = MemoryDatabase()
        add(db, "p2", MemoryStoreKind.VERIFIED_FACTS, "integrity sentinel")
        db.connection.execute(
            "UPDATE verified_facts SET source_id=? WHERE memory_id=?",
            ("tampered-source", "p2"),
        )
        db.connection.commit()
        with self.assertRaisesRegex(MemoryIntegrityError, "hash mismatch"):
            db.get("p2")
        with self.assertRaises(MemoryIntegrityError):
            LexicalRetriever(db).retrieve("integrity sentinel")

    def test_conflict_detection_normalizes_case_unicode_and_whitespace(self) -> None:
        db = MemoryDatabase()
        add(
            db,
            "c1",
            MemoryStoreKind.VERIFIED_FACTS,
            "release channel stable",
            claim_key=" Release.Channel ",
            claim_value="STABLE",
        )
        add(
            db,
            "c2",
            MemoryStoreKind.EXPERIMENT_RESULTS,
            "release channel candidate",
            claim_key="release.channel",
            claim_value="candidate",
        )
        result = LexicalRetriever(db).retrieve("release channel")
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(result.conflicts[0].claim_key, "release.channel")
        self.assertEqual(result.conflicts[0].memory_ids, ("c1", "c2"))

    def test_equal_claim_values_after_normalization_are_not_conflicts(self) -> None:
        db = MemoryDatabase()
        add(
            db,
            "c1",
            MemoryStoreKind.VERIFIED_FACTS,
            "status stable one",
            claim_key="status",
            claim_value=" Stable ",
        )
        add(
            db,
            "c2",
            MemoryStoreKind.EXPERIMENT_RESULTS,
            "status stable two",
            claim_key="STATUS",
            claim_value="stable",
        )
        self.assertFalse(LexicalRetriever(db).retrieve("status stable").conflicts)

    def test_supersession_excludes_old_and_delete_successor_restores_predecessor(
        self,
    ) -> None:
        db = MemoryDatabase()
        add(db, "old", MemoryStoreKind.USER_PROJECT, "synthetic phase alpha")
        add(
            db,
            "new",
            MemoryStoreKind.USER_PROJECT,
            "synthetic phase beta",
            version=2,
            supersedes=("old",),
        )
        self.assertEqual(db.get("old", include_inactive=True).superseded_by, ("new",))
        before = LexicalRetriever(db).retrieve("phase alpha")
        self.assertNotIn("old", [e.memory_id for e in before.evidence])
        db.delete("new")
        restored = db.get("old")
        self.assertIs(restored.verification, VerificationState.VERIFIED)
        self.assertEqual(restored.superseded_by, ())
        evidence = LexicalRetriever(db).retrieve("alpha").evidence
        self.assertEqual([e.memory_id for e in evidence], ["old"])

    def test_delete_middle_node_rewires_supersession_lineage(self) -> None:
        db = MemoryDatabase()
        add(db, "v1", MemoryStoreKind.USER_PROJECT, "phase one")
        add(
            db,
            "v2",
            MemoryStoreKind.USER_PROJECT,
            "phase two",
            version=2,
            supersedes=("v1",),
        )
        add(
            db,
            "v3",
            MemoryStoreKind.USER_PROJECT,
            "phase three",
            version=3,
            supersedes=("v2",),
        )
        db.delete("v2")
        self.assertEqual(db.get("v1", include_inactive=True).superseded_by, ("v3",))
        self.assertEqual(db.get("v3").supersedes, ("v1",))
        self.assertIs(
            db.get("v1", include_inactive=True).verification,
            VerificationState.SUPERSEDED,
        )

    def test_invalidation_wins_over_successor_deletion(self) -> None:
        db = MemoryDatabase()
        add(db, "old", MemoryStoreKind.VERIFIED_FACTS, "policy alpha")
        add(
            db,
            "new",
            MemoryStoreKind.VERIFIED_FACTS,
            "policy beta",
            supersedes=("old",),
        )
        db.invalidate("old")
        db.delete("new")
        old = db.get("old", include_inactive=True)
        self.assertIs(old.verification, VerificationState.REJECTED)
        self.assertFalse(LexicalRetriever(db).retrieve("policy alpha").evidence)

    def test_lexical_bm25_order_is_repeatable_with_stable_tie_break(self) -> None:
        db = MemoryDatabase()
        add(db, "z", MemoryStoreKind.RESEARCH_DOCUMENTS, "alpha beta gamma")
        add(db, "a", MemoryStoreKind.RESEARCH_DOCUMENTS, "alpha beta gamma")
        retriever = LexicalRetriever(db)
        observed = []
        result = None
        for _ in range(25):
            result = retriever.retrieve("alpha beta", limit=2)
            observed.append(tuple(e.memory_id for e in result.evidence))
        self.assertEqual(set(observed), {("a", "z")})
        assert result is not None
        self.assertEqual(
            result.evidence[0].lexical_score,
            result.evidence[1].lexical_score,
        )

    def test_superseded_deleted_and_invalidated_records_never_enter_retrieval(
        self,
    ) -> None:
        db = MemoryDatabase()
        add(db, "old", MemoryStoreKind.VERIFIED_FACTS, "token old active")
        add(
            db,
            "new",
            MemoryStoreKind.VERIFIED_FACTS,
            "token new active",
            supersedes=("old",),
        )
        add(db, "gone", MemoryStoreKind.VERIFIED_FACTS, "token gone active")
        db.invalidate("new")
        db.delete("gone")
        self.assertFalse(LexicalRetriever(db).retrieve("token active").evidence)

    def test_evidence_object_handoff_preserves_provenance_and_identity(self) -> None:
        db = MemoryDatabase()
        add(
            db,
            "e1",
            MemoryStoreKind.EXPERIMENT_RESULTS,
            "retrieval experiment passed",
        )
        result = LexicalRetriever(db).retrieve("retrieval experiment")
        adapter = MockPostbase255Adapter()
        response = feed_reasoning(result, adapter)
        self.assertEqual(adapter.last_evidence, result.evidence)
        self.assertIsInstance(adapter.last_evidence[0], EvidenceObject)
        self.assertEqual(adapter.last_evidence[0].provenance, SRC)
        self.assertEqual(response["evidence_ids"], ["e1"])
        self.assertEqual(response["source_ids"], [SRC.source_id])
        self.assertEqual(response["evidence_hashes"], [result.evidence[0].content_hash])

    def test_embedding_reranker_is_optional_and_fail_closed(self) -> None:
        db = MemoryDatabase()
        add(db, "x", MemoryStoreKind.RESEARCH_DOCUMENTS, "lexical route only")
        retriever = LexicalRetriever(db)
        self.assertTrue(retriever.retrieve("lexical route").evidence)
        with self.assertRaisesRegex(RuntimeError, "embedding adapter"):
            retriever.retrieve("lexical route", use_embedding_adapter=True)

    def test_no_model_torch_network_or_private_fixture_dependency(self) -> None:
        import twelve_six.memory_rag as package

        root = pathlib.Path(package.__file__).parent
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py"))
        )
        self.assertNotIn("import torch", source)
        self.assertNotIn("twelve_six.model", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)

    def test_existing_schema_is_migrated_without_data_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = pathlib.Path(tempdir) / "memory.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """CREATE TABLE verified_facts (
                    memory_id TEXT PRIMARY KEY, content TEXT NOT NULL,
                    source_type TEXT NOT NULL, source_id TEXT NOT NULL,
                    source_version TEXT NOT NULL, locator TEXT,
                    timestamp TEXT NOT NULL, version INTEGER NOT NULL,
                    confidence REAL NOT NULL, verification TEXT NOT NULL,
                    content_hash TEXT NOT NULL, claim_key TEXT, claim_value TEXT,
                    supersedes TEXT NOT NULL, superseded_by TEXT NOT NULL,
                    metadata TEXT NOT NULL, invalidated INTEGER NOT NULL DEFAULT 0
                )"""
            )
            content = "legacy synthetic record"
            digest = MemoryDatabase.compute_content_hash(
                content=content,
                provenance=SRC,
                version=1,
            )
            connection.execute(
                """INSERT INTO verified_facts VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "legacy",
                    content,
                    SRC.source_type,
                    SRC.source_id,
                    SRC.source_version,
                    SRC.locator,
                    NOW.isoformat(),
                    1,
                    0.9,
                    VerificationState.VERIFIED.value,
                    digest,
                    None,
                    None,
                    "[]",
                    "[]",
                    "{}",
                    0,
                ),
            )
            connection.commit()
            connection.close()

            db = MemoryDatabase(path)
            columns = {
                row["name"]
                for row in db.connection.execute(
                    "PRAGMA table_info(verified_facts)"
                ).fetchall()
            }
            self.assertIn("supersession_base_verification", columns)
            self.assertEqual(db.get("legacy").content, content)
            self.assertTrue(db.verify_integrity("legacy"))
            db.close()


if __name__ == "__main__":
    unittest.main()
