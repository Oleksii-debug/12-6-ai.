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
from .s0_runtime import S0TorchInferenceBackend

__all__ = [
    "AuditEvidence",
    "AuditVerdict",
    "CIEvidence",
    "CandidateStatus",
    "ComponentDisposition",
    "ComponentRef",
    "ReleaseArtifactEvidence",
    "S0TorchInferenceBackend",
    "StageCandidateManifest",
    "validate_repository_evidence",
]
