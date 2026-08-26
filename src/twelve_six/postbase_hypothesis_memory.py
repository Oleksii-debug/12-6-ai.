from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any, Callable, Literal, Sequence

from twelve_six.memory_rag import (
    ConflictEvidence,
    EvidenceObject,
    LexicalRetriever,
    MemoryDatabase,
    MemoryIntegrityError,
    MemoryStoreKind,
    Provenance,
    RetrievalResult,
    VerificationState,
)
from twelve_six.postbase_hypothesis import Hypothesis, HypothesisSearch

MemoryRelation = Literal["support", "contradiction", "irrelevant"]
RelationResolver = Callable[[Hypothesis, EvidenceObject], MemoryRelation]


class MemoryEvidenceState(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    REJECTED = "rejected"
    INTEGRITY_FAILED = "integrity_failed"
    IDENTITY_MISMATCH = "identity_mismatch"


@dataclass(frozen=True, slots=True)
class HypothesisMemoryEvidence:
    binding_id: str
    hypothesis_id: str
    hypothesis_statement: str
    relation: Literal["support", "contradiction"]
    memory_id: str
    source: str
    provenance: Provenance
    store: MemoryStoreKind
    timestamp: str
    version: int
    content_hash: str
    verification: VerificationState
    supersedes: tuple[str, ...]
    superseded_by: tuple[str, ...]
    lexical_score: float
    confidence: float
    score_delta: float
    conflicts: tuple[ConflictEvidence, ...]
    memory_state: MemoryEvidenceState


@dataclass(frozen=True, slots=True)
class BoundRetrieval:
    query: str
    hypothesis_id: str
    retrieval: RetrievalResult
    bindings: tuple[HypothesisMemoryEvidence, ...]


@dataclass(frozen=True, slots=True)
class HypothesisScorecard:
    hypothesis_id: str
    base_score: float
    memory_delta: float
    adjusted_score: float
    active_memory_binding_ids: tuple[str, ...]


class HypothesisMemoryRAG:
    """Deterministic SQLite/BM25 evidence overlay for hypothesis selection.

    Retrieval never mutates the underlying hypothesis history. Instead, active memory
    evidence contributes an overlay score. Before any selection/export, bindings are
    refreshed against SQLite so superseded, deleted, rejected, tampered, or identity-
    changed memories cannot continue to support an active hypothesis.
    """

    worker_id = "NEXT100-083-HYPOTHESIS-MEMORY-RAG"
    schema = "12-6.postbase-hypothesis-memory-rag.v1"

    def __init__(
        self,
        search: HypothesisSearch,
        database: MemoryDatabase,
        *,
        retriever: LexicalRetriever | None = None,
    ) -> None:
        self.search = search
        self.database = database
        self.retriever = retriever or LexicalRetriever(database)
        if self.retriever.database is not database:
            raise ValueError("retriever must use the same MemoryDatabase")
        self._bindings: dict[str, HypothesisMemoryEvidence] = {}
        self._binding_counter = 0

    def retrieve_and_bind(
        self,
        hypothesis_id: str,
        query: str,
        relation_resolver: RelationResolver,
        *,
        limit: int = 8,
        stores: Sequence[MemoryStoreKind] | None = None,
        max_score_delta: float = 0.40,
    ) -> BoundRetrieval:
        hypothesis = self.search.hypothesis(hypothesis_id)
        if hypothesis.status != "active":
            raise ValueError(f"hypothesis {hypothesis_id} is not active")
        if not 0.0 < max_score_delta <= 1.0:
            raise ValueError("max_score_delta must be in (0,1]")

        retrieval = self.retriever.retrieve(
            query,
            limit=limit,
            stores=stores,
            use_embedding_adapter=False,
        )
        created: list[HypothesisMemoryEvidence] = []
        for evidence in retrieval.evidence:
            relation = relation_resolver(hypothesis, evidence)
            if relation == "irrelevant":
                continue
            if relation not in ("support", "contradiction"):
                raise ValueError(f"unsupported memory relation: {relation!r}")
            existing = self._active_duplicate(
                hypothesis_id=hypothesis.id,
                evidence=evidence,
                relation=relation,
            )
            if existing is not None:
                created.append(existing)
                continue
            magnitude = max_score_delta * evidence.confidence
            signed_delta = magnitude if relation == "support" else -magnitude
            binding = HypothesisMemoryEvidence(
                binding_id=self._next_binding_id(),
                hypothesis_id=hypothesis.id,
                hypothesis_statement=hypothesis.statement,
                relation=relation,
                memory_id=evidence.memory_id,
                source=self._source(evidence.provenance),
                provenance=evidence.provenance,
                store=evidence.store,
                timestamp=evidence.timestamp.isoformat(),
                version=evidence.version,
                content_hash=evidence.content_hash,
                verification=evidence.verification,
                supersedes=evidence.supersedes,
                superseded_by=evidence.superseded_by,
                lexical_score=evidence.lexical_score,
                confidence=evidence.confidence,
                score_delta=signed_delta,
                conflicts=evidence.conflicts,
                memory_state=MemoryEvidenceState.ACTIVE,
            )
            self._bindings[binding.binding_id] = binding
            created.append(binding)
        return BoundRetrieval(
            query=query,
            hypothesis_id=hypothesis.id,
            retrieval=retrieval,
            bindings=tuple(created),
        )

    def refresh_bindings(self) -> tuple[HypothesisMemoryEvidence, ...]:
        refreshed: list[HypothesisMemoryEvidence] = []
        for binding_id in sorted(self._bindings):
            binding = self._bindings[binding_id]
            if binding.memory_state is not MemoryEvidenceState.ACTIVE:
                refreshed.append(binding)
                continue
            new_state = self._current_state(binding)
            if new_state is not binding.memory_state:
                binding = replace(binding, memory_state=new_state)
                self._bindings[binding_id] = binding
            refreshed.append(binding)
        return tuple(refreshed)

    def scorecard(self, hypothesis_id: str) -> HypothesisScorecard:
        self.refresh_bindings()
        hypothesis = self.search.hypothesis(hypothesis_id)
        active = tuple(
            binding
            for binding in self._bindings.values()
            if binding.hypothesis_id == hypothesis_id
            and binding.memory_state is MemoryEvidenceState.ACTIVE
        )
        delta = sum(binding.score_delta for binding in active)
        adjusted = min(1.0, max(0.0, hypothesis.score + delta))
        return HypothesisScorecard(
            hypothesis_id=hypothesis_id,
            base_score=hypothesis.score,
            memory_delta=delta,
            adjusted_score=adjusted,
            active_memory_binding_ids=tuple(
                binding.binding_id for binding in sorted(active, key=lambda item: item.binding_id)
            ),
        )

    def preferred(self) -> Hypothesis | None:
        self.refresh_bindings()
        active = [
            self.search.hypothesis(hypothesis_id)
            for hypothesis_id in self._hypothesis_ids()
            if self.search.hypothesis(hypothesis_id).status == "active"
        ]
        if not active:
            return None
        return max(
            active,
            key=lambda hypothesis: (
                self.scorecard(hypothesis.id).adjusted_score,
                hypothesis.id,
            ),
        )

    def bindings(self) -> tuple[HypothesisMemoryEvidence, ...]:
        return self.refresh_bindings()

    def export(self) -> dict[str, Any]:
        self.refresh_bindings()
        hypothesis_ids = sorted(
            {
                binding.hypothesis_id
                for binding in self._bindings.values()
            }
            | set(self._hypothesis_ids())
        )
        return {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "retrieval_mode": "sqlite_bm25_local_only",
            "embeddings_used": False,
            "external_retrieval_used": False,
            "bindings": [
                self._binding_to_dict(self._bindings[key])
                for key in sorted(self._bindings)
            ],
            "scorecards": [asdict(self.scorecard(key)) for key in hypothesis_ids],
            "selected_hypothesis_id": self.preferred().id if self.preferred() is not None else None,
            "hypothesis_search": self.search.export(),
        }

    def _active_duplicate(
        self,
        *,
        hypothesis_id: str,
        evidence: EvidenceObject,
        relation: Literal["support", "contradiction"],
    ) -> HypothesisMemoryEvidence | None:
        for binding in self._bindings.values():
            if (
                binding.memory_state is MemoryEvidenceState.ACTIVE
                and binding.hypothesis_id == hypothesis_id
                and binding.relation == relation
                and binding.memory_id == evidence.memory_id
                and binding.version == evidence.version
                and binding.content_hash == evidence.content_hash
            ):
                return binding
        return None

    def _current_state(self, binding: HypothesisMemoryEvidence) -> MemoryEvidenceState:
        try:
            item = self.database.get(binding.memory_id, include_inactive=True)
        except KeyError:
            return MemoryEvidenceState.DELETED
        except MemoryIntegrityError:
            return MemoryEvidenceState.INTEGRITY_FAILED

        if (
            item.version != binding.version
            or item.content_hash != binding.content_hash
            or item.provenance != binding.provenance
            or item.store != binding.store
        ):
            return MemoryEvidenceState.IDENTITY_MISMATCH
        if item.verification is VerificationState.REJECTED:
            return MemoryEvidenceState.REJECTED
        if (
            item.verification is VerificationState.SUPERSEDED
            or bool(item.superseded_by)
        ):
            return MemoryEvidenceState.SUPERSEDED
        return MemoryEvidenceState.ACTIVE

    def _hypothesis_ids(self) -> tuple[str, ...]:
        exported = self.search.export()
        return tuple(item["id"] for item in exported["hypotheses"])

    def _next_binding_id(self) -> str:
        self._binding_counter += 1
        return f"MR{self._binding_counter:03d}"

    @staticmethod
    def _source(provenance: Provenance) -> str:
        return (
            f"{provenance.source_type}:{provenance.source_id}"
            f"@{provenance.source_version}"
        )

    @staticmethod
    def _binding_to_dict(binding: HypothesisMemoryEvidence) -> dict[str, Any]:
        payload = asdict(binding)
        payload["memory_state"] = binding.memory_state.value
        payload["store"] = binding.store.value
        payload["verification"] = binding.verification.value
        return payload
