from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import EvidenceObject, RetrievalResult


class Postbase255ReasoningAdapter(Protocol):
    def reason(self, query: str, evidence: Sequence[EvidenceObject]) -> object: ...


@dataclass(slots=True)
class MockPostbase255Adapter:
    """Test-only deterministic adapter proving typed evidence-object handoff."""

    last_evidence: tuple[EvidenceObject, ...] = ()

    def reason(self, query: str, evidence: Sequence[EvidenceObject]) -> dict[str, object]:
        self.last_evidence = tuple(evidence)
        return {
            "query": query,
            "evidence_ids": [item.memory_id for item in self.last_evidence],
            "evidence_hashes": [item.content_hash for item in self.last_evidence],
            "source_ids": [item.provenance.source_id for item in self.last_evidence],
        }


def feed_reasoning(result: RetrievalResult, adapter: Postbase255ReasoningAdapter) -> object:
    return adapter.reason(result.query, result.evidence)
