from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Mapping


class MemoryStoreKind(StrEnum):
    VERIFIED_FACTS = "verified_facts"
    USER_PROJECT = "user_project_memory"
    RESEARCH_DOCUMENTS = "research_documents"
    HYPOTHESES = "hypotheses"
    EXPERIMENT_RESULTS = "experiment_results"


class VerificationState(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class Provenance:
    source_type: str
    source_id: str
    source_version: str
    locator: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("source_type", self.source_type),
            ("source_id", self.source_id),
            ("source_version", self.source_version),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class MemoryItem:
    memory_id: str
    store: MemoryStoreKind
    content: str
    provenance: Provenance
    timestamp: datetime
    version: int
    confidence: float
    verification: VerificationState
    content_hash: str
    claim_key: str | None = None
    claim_value: str | None = None
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id must be non-empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if (self.claim_key is None) != (self.claim_value is None):
            raise ValueError("claim_key and claim_value must be supplied together")
        if self.claim_key is not None and not self.claim_key.strip():
            raise ValueError("claim_key must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class ConflictEvidence:
    claim_key: str
    memory_ids: tuple[str, ...]
    claim_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceObject:
    memory_id: str
    store: MemoryStoreKind
    content: str
    provenance: Provenance
    timestamp: datetime
    version: int
    confidence: float
    verification: VerificationState
    content_hash: str
    lexical_score: float
    conflicts: tuple[ConflictEvidence, ...] = ()
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    evidence: tuple[EvidenceObject, ...]
    conflicts: tuple[ConflictEvidence, ...]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
