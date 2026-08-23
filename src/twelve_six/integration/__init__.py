"""Stage composition and release bookkeeping contracts."""

from .manifest import (
    AuditEvidence,
    AuditVerdict,
    CandidateStatus,
    CIEvidence,
    ComponentDisposition,
    ComponentRef,
    ReleaseArtifactEvidence,
    StageCandidateManifest,
    validate_repository_evidence,
)

__all__ = [
    "AuditEvidence",
    "AuditVerdict",
    "CandidateStatus",
    "CIEvidence",
    "ComponentDisposition",
    "ComponentRef",
    "ReleaseArtifactEvidence",
    "StageCandidateManifest",
    "validate_repository_evidence",
]
