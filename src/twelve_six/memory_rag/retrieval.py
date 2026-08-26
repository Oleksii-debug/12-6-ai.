from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, Sequence

from .contracts import (
    ConflictEvidence,
    EvidenceObject,
    MemoryItem,
    MemoryStoreKind,
    RetrievalResult,
)
from .store import MemoryDatabase

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.casefold()))


class EmbeddingAdapter(Protocol):
    def rerank(
        self, query: str, evidence: Sequence[EvidenceObject]
    ) -> Sequence[EvidenceObject]: ...


class LexicalRetriever:
    """Deterministic BM25-style lexical retrieval with memory_id tie breaking."""

    def __init__(
        self,
        database: MemoryDatabase,
        embedding_adapter: EmbeddingAdapter | None = None,
    ) -> None:
        self.database = database
        self.embedding_adapter = embedding_adapter

    @staticmethod
    def _conflicts(items: Sequence[MemoryItem]) -> tuple[ConflictEvidence, ...]:
        grouped: dict[str, list[MemoryItem]] = {}
        for item in items:
            if item.claim_key is not None:
                grouped.setdefault(item.claim_key.casefold(), []).append(item)
        conflicts: list[ConflictEvidence] = []
        for key in sorted(grouped):
            records = grouped[key]
            values = {
                record.claim_value.casefold()
                for record in records
                if record.claim_value is not None
            }
            if len(values) > 1:
                ordered = sorted(records, key=lambda record: record.memory_id)
                conflicts.append(
                    ConflictEvidence(
                        claim_key=key,
                        memory_ids=tuple(record.memory_id for record in ordered),
                        claim_values=tuple(record.claim_value or "" for record in ordered),
                    )
                )
        return tuple(conflicts)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        stores: Sequence[MemoryStoreKind] | None = None,
        use_embedding_adapter: bool = False,
    ) -> RetrievalResult:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        items = self.database.active_items(stores)
        q_tokens = _tokens(query)
        if not q_tokens or not items:
            return RetrievalResult(query=query, evidence=(), conflicts=self._conflicts(items))
        docs = [_tokens(item.content) for item in items]
        avg_len = sum(len(doc) for doc in docs) / len(docs) or 1.0
        doc_freq = Counter(token for token in set(q_tokens) for doc in docs if token in doc)
        conflicts = self._conflicts(items)
        conflict_by_id = {
            memory_id: tuple(conflict for conflict in conflicts if memory_id in conflict.memory_ids)
            for memory_id in (item.memory_id for item in items)
        }
        scored: list[EvidenceObject] = []
        n_docs = len(items)
        for item, doc in zip(items, docs, strict=True):
            counts = Counter(doc)
            score = 0.0
            for token in q_tokens:
                tf = counts[token]
                if tf == 0:
                    continue
                df = doc_freq[token]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + 1.2 * (1.0 - 0.75 + 0.75 * len(doc) / avg_len)
                score += idf * (tf * 2.2) / denom
            if score <= 0.0:
                continue
            scored.append(
                EvidenceObject(
                    memory_id=item.memory_id,
                    store=item.store,
                    content=item.content,
                    provenance=item.provenance,
                    timestamp=item.timestamp,
                    version=item.version,
                    confidence=item.confidence,
                    verification=item.verification,
                    content_hash=item.content_hash,
                    lexical_score=score,
                    conflicts=conflict_by_id[item.memory_id],
                    supersedes=item.supersedes,
                    superseded_by=item.superseded_by,
                )
            )
        scored.sort(key=lambda evidence: (-evidence.lexical_score, evidence.memory_id))
        evidence: Sequence[EvidenceObject] = tuple(scored[:limit])
        if use_embedding_adapter:
            if self.embedding_adapter is None:
                raise RuntimeError("embedding adapter requested but not configured")
            evidence = tuple(self.embedding_adapter.rerank(query, evidence))[:limit]
        return RetrievalResult(query=query, evidence=tuple(evidence), conflicts=conflicts)
