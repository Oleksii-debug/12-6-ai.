"""Stage composition and release bookkeeping contracts."""

from .manifest import (
    AuditVerdict,
    CandidateStatus,
    ComponentDisposition,
    ComponentRef,
    StageCandidateManifest,
)

__all__ = [
    "AuditVerdict",
    "CandidateStatus",
    "ComponentDisposition",
    "ComponentRef",
    "StageCandidateManifest",
]
