"""Isolated future post-training contracts.

Importing this package does not load or mutate model weights.
"""

from .experiment import (
    BaseLineageViolation,
    BehavioralTrainingNotAuthorized,
    PostTrainingExperimentConfig,
)
from .rollout import RolloutCandidate, RolloutProvider, RolloutRequest, SamplingSpec
from .schemas import (
    ArtifactRef,
    DataProvenance,
    ExecutionMode,
    FrameworkKind,
    Message,
    PostTrainingMethod,
    PreferenceRecord,
    SFTRecord,
    SourceKind,
    SyntheticProvenance,
    VerifierRecord,
    canonical_json,
    content_fingerprint,
)
from .verifiers import (
    ExactMatchVerifier,
    VerificationContext,
    Verifier,
    VerifierRegistry,
    VerifierResult,
)

__all__ = [
    "ArtifactRef",
    "BaseLineageViolation",
    "BehavioralTrainingNotAuthorized",
    "DataProvenance",
    "ExactMatchVerifier",
    "ExecutionMode",
    "FrameworkKind",
    "Message",
    "PostTrainingExperimentConfig",
    "PostTrainingMethod",
    "PreferenceRecord",
    "RolloutCandidate",
    "RolloutProvider",
    "RolloutRequest",
    "SFTRecord",
    "SamplingSpec",
    "SourceKind",
    "SyntheticProvenance",
    "VerificationContext",
    "Verifier",
    "VerifierRecord",
    "VerifierRegistry",
    "VerifierResult",
    "canonical_json",
    "content_fingerprint",
]
