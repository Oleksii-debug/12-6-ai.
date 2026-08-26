from __future__ import annotations

import pathlib
import unittest
from datetime import datetime, timezone

from twelve_six.memory_rag import (
    EvidenceObject,
    LexicalRetriever,
    MemoryDatabase,
    MemoryStoreKind,
    Provenance,
    VerificationState,
)
from twelve_six.postbase_hypothesis import Hypothesis, HypothesisSearch
from twelve_six.postbase_hypothesis_memory import (
    HypothesisMemoryRAG,
    MemoryEvidenceState,
)

NOW = datetime(2026, 8, 26, 18, 0, tzinfo=timezone.utc)
PROVENANCE = Provenance(
    "synthetic_fixture",
    "python-precedence-spec",
    "v2",
    "synthetic://next100-083/operator-precedence",
)


def add_memory(
    db: MemoryDatabase,
    memory_id: str,
    content: str,
    *,
    version: int = 1,
    confidence: float = 1.0,
    supersedes: tuple[str, ...] = (),
):
    return db.add(
        memory_id=memory_id,
        store=MemoryStoreKind.VERIFIED_FACTS,
        content=content,
        provenance=PROVENANCE,
        timestamp=NOW,
        version=version,
        confidence=confidence,
        verification=VerificationState.VERIFIED,
        claim_key="operator_precedence",
        claim_value="multiplication_before_addition",
        supersedes=supersedes,
    )


def precedence_relation(
    hypothesis: Hypothesis,
    evidence: EvidenceObject,
):
    statement = hypothesis.statement.casefold()
    if "multiplication before addition" in statement:
        return "support"
    if "addition before multiplication" in statement:
        return "contradiction"
    return "irrelevant"


class HypothesisMemoryRagNext100083Tests(unittest.TestCase):
    def test_retrieval_overturns_initial_preferred_hypothesis(self) -> None:
        db = MemoryDatabase()
        item = add_memory(
            db,
            "precedence-v2",
            "operator precedence multiplication before addition is the active rule",
            version=2,
        )
        search = HypothesisSearch()
        wrong = search.propose("addition before multiplication", initial_score=0.85)
        correct = search.propose("multiplication before addition", initial_score=0.60)
        self.assertEqual(search.best().id, wrong.id)

        integration = HypothesisMemoryRAG(search, db)
        wrong_bound = integration.retrieve_and_bind(
            wrong.id,
            "operator precedence multiplication addition",
            precedence_relation,
            max_score_delta=0.55,
        )
        correct_bound = integration.retrieve_and_bind(
            correct.id,
            "operator precedence multiplication addition",
            precedence_relation,
            max_score_delta=0.55,
        )

        self.assertEqual(integration.preferred().id, correct.id)
        self.assertLess(
            integration.scorecard(wrong.id).adjusted_score,
            integration.scorecard(correct.id).adjusted_score,
        )
        binding = correct_bound.bindings[0]
        self.assertEqual(binding.memory_id, item.memory_id)
        self.assertEqual(binding.source, "synthetic_fixture:python-precedence-spec@v2")
        self.assertEqual(binding.provenance, PROVENANCE)
        self.assertEqual(binding.version, 2)
        self.assertEqual(binding.content_hash, item.content_hash)
        self.assertIs(binding.verification, VerificationState.VERIFIED)
        self.assertEqual(binding.superseded_by, ())
        self.assertIs(binding.memory_state, MemoryEvidenceState.ACTIVE)
        self.assertEqual(wrong_bound.bindings[0].relation, "contradiction")
        self.assertEqual(correct_bound.bindings[0].relation, "support")

    def test_superseded_memory_stops_supporting_without_retrieval_refresh(self) -> None:
        db = MemoryDatabase()
        add_memory(
            db,
            "policy-v1",
            "operator precedence multiplication before addition legacy rule",
        )
        search = HypothesisSearch()
        candidate = search.propose("multiplication before addition", initial_score=0.45)
        rival = search.propose("unrelated rival", initial_score=0.50)
        integration = HypothesisMemoryRAG(search, db)
        bound = integration.retrieve_and_bind(
            candidate.id,
            "operator precedence multiplication addition",
            precedence_relation,
            max_score_delta=0.40,
        )
        self.assertEqual(integration.preferred().id, candidate.id)

        add_memory(
            db,
            "policy-v2",
            "operator precedence multiplication before addition replacement rule",
            version=2,
            supersedes=("policy-v1",),
        )

        self.assertEqual(integration.preferred().id, rival.id)
        refreshed = {binding.binding_id: binding for binding in integration.bindings()}
        self.assertIs(
            refreshed[bound.bindings[0].binding_id].memory_state,
            MemoryEvidenceState.SUPERSEDED,
        )
        self.assertEqual(integration.scorecard(candidate.id).memory_delta, 0.0)

    def test_deleted_memory_stops_supporting_active_hypothesis(self) -> None:
        db = MemoryDatabase()
        add_memory(
            db,
            "ephemeral",
            "operator precedence multiplication before addition temporary evidence",
        )
        search = HypothesisSearch()
        candidate = search.propose("multiplication before addition", initial_score=0.45)
        rival = search.propose("unrelated rival", initial_score=0.50)
        integration = HypothesisMemoryRAG(search, db)
        bound = integration.retrieve_and_bind(
            candidate.id,
            "operator precedence multiplication addition",
            precedence_relation,
            max_score_delta=0.40,
        )
        self.assertEqual(integration.preferred().id, candidate.id)

        db.delete("ephemeral")

        self.assertEqual(integration.preferred().id, rival.id)
        refreshed = {binding.binding_id: binding for binding in integration.bindings()}
        self.assertIs(
            refreshed[bound.bindings[0].binding_id].memory_state,
            MemoryEvidenceState.DELETED,
        )
        self.assertEqual(integration.scorecard(candidate.id).memory_delta, 0.0)

    def test_integrity_failure_fails_closed_and_removes_support(self) -> None:
        db = MemoryDatabase()
        add_memory(
            db,
            "tamper-target",
            "operator precedence multiplication before addition integrity sentinel",
        )
        search = HypothesisSearch()
        candidate = search.propose("multiplication before addition", initial_score=0.45)
        rival = search.propose("unrelated rival", initial_score=0.50)
        integration = HypothesisMemoryRAG(search, db)
        bound = integration.retrieve_and_bind(
            candidate.id,
            "operator precedence multiplication addition",
            precedence_relation,
            max_score_delta=0.40,
        )
        self.assertEqual(integration.preferred().id, candidate.id)

        db.connection.execute(
            "UPDATE verified_facts SET content=? WHERE memory_id=?",
            ("tampered content", "tamper-target"),
        )
        db.connection.commit()

        self.assertEqual(integration.preferred().id, rival.id)
        refreshed = {binding.binding_id: binding for binding in integration.bindings()}
        self.assertIs(
            refreshed[bound.bindings[0].binding_id].memory_state,
            MemoryEvidenceState.INTEGRITY_FAILED,
        )

    def test_duplicate_active_binding_does_not_double_count(self) -> None:
        db = MemoryDatabase()
        add_memory(
            db,
            "stable",
            "operator precedence multiplication before addition stable evidence",
        )
        search = HypothesisSearch()
        candidate = search.propose("multiplication before addition", initial_score=0.40)
        integration = HypothesisMemoryRAG(search, db)
        first = integration.retrieve_and_bind(
            candidate.id,
            "operator precedence multiplication addition",
            precedence_relation,
            max_score_delta=0.25,
        )
        second = integration.retrieve_and_bind(
            candidate.id,
            "operator precedence multiplication addition",
            precedence_relation,
            max_score_delta=0.25,
        )
        self.assertEqual(first.bindings[0].binding_id, second.bindings[0].binding_id)
        self.assertAlmostEqual(integration.scorecard(candidate.id).memory_delta, 0.25)

    def test_embedding_adapter_is_never_invoked(self) -> None:
        class BombEmbeddingAdapter:
            def rerank(self, query, evidence):
                raise AssertionError("embedding adapter must not be used")

        db = MemoryDatabase()
        add_memory(
            db,
            "lexical-only",
            "operator precedence multiplication before addition lexical evidence",
        )
        search = HypothesisSearch()
        candidate = search.propose("multiplication before addition", initial_score=0.40)
        retriever = LexicalRetriever(db, embedding_adapter=BombEmbeddingAdapter())
        integration = HypothesisMemoryRAG(search, db, retriever=retriever)
        result = integration.retrieve_and_bind(
            candidate.id,
            "operator precedence multiplication addition",
            precedence_relation,
        )
        self.assertTrue(result.bindings)

    def test_runtime_has_no_external_retrieval_dependency(self) -> None:
        import twelve_six.postbase_hypothesis_memory as module

        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("requests", source)
        self.assertNotIn("urllib", source)
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)
        self.assertNotIn("embedding_adapter=True", source)


if __name__ == "__main__":
    unittest.main()
