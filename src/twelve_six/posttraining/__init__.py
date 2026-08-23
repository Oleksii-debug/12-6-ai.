"""Isolated post-training infrastructure for future 12-6 AI descendants.

This package must not mutate canonical Base checkpoints. It defines contracts,
provenance, dataset adapters and verifier interfaces that can later be consumed by
TRL, verl, vLLM-backed rollout systems, or other approved frameworks.
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
from .dataset_adapters import (
    TRL_TRAINER_RECORD_KINDS,
    DatasetSchemaError,
    to_trl_example,
    validate_trl_compatible_record,
)
from .harness import (
    CaseVerification,
    VerificationCase,
    VerifierHarnessReport,
    run_verifier_harness,
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
    "CaseVerification",
    "CheckpointRef",
    "ComputeClass",
    "DatasetManifest",
    "DatasetRecord",
    "DatasetSchemaError",
    "ExactTextVerifier",
    "LineageKind",
    "ManifestEntry",
    "NumericToleranceVerifier",
    "PostTrainingExperiment",
    "RecordKind",
    "Split",
    "SyntheticProvenance",
    "TRL_TRAINER_RECORD_KINDS",
    "VerificationCase",
    "VerificationResult",
    "Verifier",
    "VerifierHarnessReport",
    "VerifierRegistry",
    "VerifierTask",
    "canonical_sha256",
    "run_verifier_harness",
    "to_trl_example",
    "validate_trl_compatible_record",
]
