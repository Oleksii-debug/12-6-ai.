from __future__ import annotations

import unittest
from datetime import datetime, timezone

from twelve_six.memory_rag import (
    EvidenceObject,
    LexicalRetriever,
    MemoryDatabase,
    MemoryStoreKind,
    MockPostbase255Adapter,
    Provenance,
    VerificationState,
    feed_reasoning,
)

NOW = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)
SRC = Provenance("synthetic_test", "fixture-public-001", "v1", "tests")


def add(db: MemoryDatabase, memory_id: str, store: MemoryStoreKind, content: str, **kwargs):
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


class MemoryRagPostbase258Tests(unittest.TestCase):
    def test_five_separate_stores_and_required_metadata(self) -> None:
        db = MemoryDatabase()
        for index, store in enumerate(MemoryStoreKind):
            item = add(db, f"m{index}", store, f"synthetic memory {index}")
            self.assertEqual(item.provenance, SRC)
            self.assertEqual(item.timestamp, NOW)
            self.assertEqual(item.version, 1)
            self.assertAlmostEqual(item.confidence, 0.9)
            self.assertIs(item.verification, VerificationState.VERIFIED)
            self.assertEqual(len(item.content_hash), 64)
        self.assertEqual(set(db.table_counts()), set(MemoryStoreKind))
        self.assertTrue(all(count == 1 for count in db.table_counts().values()))

    def test_lexical_retrieval_is_deterministic_and_returns_evidence_objects(self) -> None:
        db = MemoryDatabase()
        add(db, "a", MemoryStoreKind.RESEARCH_DOCUMENTS, "static cache uses preallocated storage")
        add(db, "b", MemoryStoreKind.VERIFIED_FACTS, "dynamic cache grows storage")
        retriever = LexicalRetriever(db)
        first = retriever.retrieve("preallocated cache storage")
        second = retriever.retrieve("preallocated cache storage")
        self.assertEqual(first, second)
        self.assertEqual([e.memory_id for e in first.evidence], ["a", "b"])
        self.assertTrue(all(isinstance(e, EvidenceObject) for e in first.evidence))
        self.assertEqual(first.evidence[0].provenance.source_id, "fixture-public-001")

    def test_conflict_detection_is_structured_not_inferred_by_llm(self) -> None:
        db = MemoryDatabase()
        add(
            db,
            "c1",
            MemoryStoreKind.VERIFIED_FACTS,
            "release channel is stable",
            claim_key="release.channel",
            claim_value="stable",
        )
        add(
            db,
            "c2",
            MemoryStoreKind.EXPERIMENT_RESULTS,
            "release channel is candidate",
            claim_key="release.channel",
            claim_value="candidate",
        )
        result = LexicalRetriever(db).retrieve("release channel")
        self.assertEqual(len(result.conflicts), 1)
        conflict = result.conflicts[0]
        self.assertEqual(conflict.memory_ids, ("c1", "c2"))
        self.assertEqual(set(conflict.claim_values), {"stable", "candidate"})
        self.assertTrue(result.evidence[0].conflicts)

    def test_supersession_invalidation_and_delete_need_no_weight_update(self) -> None:
        db = MemoryDatabase()
        old = add(db, "old", MemoryStoreKind.USER_PROJECT, "project phase alpha")
        new = add(
            db,
            "new",
            MemoryStoreKind.USER_PROJECT,
            "project phase beta",
            version=2,
            supersedes=(old.memory_id,),
        )
        old_after = db.get("old", include_inactive=True)
        self.assertIs(old_after.verification, VerificationState.SUPERSEDED)
        self.assertEqual(old_after.superseded_by, (new.memory_id,))
        self.assertEqual(new.supersedes, (old.memory_id,))
        db.invalidate("new")
        self.assertFalse(LexicalRetriever(db).retrieve("phase beta").evidence)
        db.delete("old")
        with self.assertRaises(KeyError):
            db.get("old", include_inactive=True)

    def test_content_hash_changes_with_source_or_version(self) -> None:
        db = MemoryDatabase()
        first = add(db, "h1", MemoryStoreKind.HYPOTHESES, "same content", version=1)
        second = add(db, "h2", MemoryStoreKind.HYPOTHESES, "same content", version=2)
        third = add(
            db,
            "h3",
            MemoryStoreKind.HYPOTHESES,
            "same content",
            provenance=Provenance("synthetic_test", "fixture-public-002", "v1"),
        )
        self.assertEqual(len({first.content_hash, second.content_hash, third.content_hash}), 3)

    def test_mock_postbase255_receives_evidence_objects_not_flattened_blob(self) -> None:
        db = MemoryDatabase()
        add(db, "r1", MemoryStoreKind.EXPERIMENT_RESULTS, "retrieval experiment passed")
        result = LexicalRetriever(db).retrieve("retrieval experiment")
        adapter = MockPostbase255Adapter()
        response = feed_reasoning(result, adapter)
        self.assertEqual(adapter.last_evidence, result.evidence)
        self.assertIsInstance(adapter.last_evidence[0], EvidenceObject)
        self.assertEqual(response["evidence_ids"], ["r1"])
        self.assertEqual(response["evidence_hashes"], [result.evidence[0].content_hash])

    def test_embedding_adapter_is_optional_and_fail_closed_when_requested(self) -> None:
        db = MemoryDatabase()
        add(db, "x", MemoryStoreKind.RESEARCH_DOCUMENTS, "lexical path first")
        retriever = LexicalRetriever(db)
        self.assertTrue(retriever.retrieve("lexical path").evidence)
        with self.assertRaisesRegex(RuntimeError, "embedding adapter"):
            retriever.retrieve("lexical path", use_embedding_adapter=True)

    def test_memory_substrate_has_no_model_or_torch_dependency(self) -> None:
        import pathlib
        import twelve_six.memory_rag as package

        root = pathlib.Path(package.__file__).parent
        source = "\n".join(path.read_text() for path in sorted(root.glob("*.py")))
        self.assertNotIn("import torch", source)
        self.assertNotIn("twelve_six.model", source)


if __name__ == "__main__":
    unittest.main()
