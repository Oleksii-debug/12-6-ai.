"""Isolated post-training infrastructure for future 12-6 AI descendants.

This package must not mutate canonical Base checkpoints. It defines contracts,
provenance and verifier interfaces that can later be consumed by TRL, verl,
vLLM-backed rollout systems, or other approved frameworks.
"""

from .contracts import (
    BackendCapabilities,
    Candidate,
    CheckpointRef,
    ComputeClass,
    DatasetRecord,
    LineageKind,
    PostTrainingExperiment,
    RecordKind,
    Split,
    SyntheticProvenance,
    VerifierTask,
)
from .provenance import DatasetManifest, ManifestEntry, canonical_sha256
from .verifiers import (
    ExactTextVerifier,
    NumericToleranceVerifier,
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

__all__ = [
    "BackendCapabilities",
    "Candidate",
    "CheckpointRef",
    "ComputeClass",
    "DatasetManifest",
    "DatasetRecord",
    "ExactTextVerifier",
    "LineageKind",
    "ManifestEntry",
    "NumericToleranceVerifier",
    "PostTrainingExperiment",
    "RecordKind",
    "Split",
    "SyntheticProvenance",
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "VerifierTask",
    "canonical_sha256",
]
