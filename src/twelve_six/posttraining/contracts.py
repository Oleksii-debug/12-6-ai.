"""Stable data and experiment contracts for future post-training work."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Mapping

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class LineageKind(StrEnum):
    """Checkpoint lineage classes.

    BASE is intentionally distinct from every post-training descendant.
    """

    BASE = "base"
    POSTTRAIN = "posttrain"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RecordKind(StrEnum):
    PROMPT_COMPLETION = "prompt_completion"
    PREFERENCE = "preference"
    VERIFIER_TASK = "verifier_task"
    CANDIDATE = "candidate"


class ComputeClass(StrEnum):
    LOCAL_FREE = "local_free"
    MATERIAL_PAID = "material_paid"


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Immutable identity for an input checkpoint."""

    checkpoint_id: str
    sha256: str
    git_sha: str
    stage: str
    lineage: LineageKind

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be non-empty")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("checkpoint sha256 must be a lowercase 64-hex digest")
        if not _GIT_SHA_RE.fullmatch(self.git_sha):
            raise ValueError("git_sha must be a 7-40 character lowercase hex SHA")
        if not self.stage.strip():
            raise ValueError("stage must be non-empty")


@dataclass(frozen=True, slots=True)
class SyntheticProvenance:
    """Provenance for a generated or transformed record."""

    source_id: str
    content_sha256: str
    synthetic: bool = False
    generator_id: str | None = None
    parent_sha256: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not _SHA256_RE.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase 64-hex digest")
        if self.synthetic and not (self.generator_id and self.generator_id.strip()):
            raise ValueError("synthetic provenance requires generator_id")
        for digest in self.parent_sha256:
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError("parent_sha256 contains an invalid digest")


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Framework-neutral dataset row.

    Payload keys are algorithm-specific, while identity, split and provenance are
    mandatory and stable across SFT/preference/verifier pipelines.
    """

    record_id: str
    kind: RecordKind
    split: Split
    payload: Mapping[str, str]
    provenance: SyntheticProvenance

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if not self.payload:
            raise ValueError("payload must be non-empty")
        if any(not key.strip() for key in self.payload):
            raise ValueError("payload keys must be non-empty")


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    prompt_id: str
    text: str
    checkpoint: CheckpointRef
    generation_config_sha256: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.prompt_id.strip():
            raise ValueError("candidate_id and prompt_id must be non-empty")
        if not _SHA256_RE.fullmatch(self.generation_config_sha256):
            raise ValueError("generation_config_sha256 must be a lowercase 64-hex digest")


@dataclass(frozen=True, slots=True)
class VerifierTask:
    task_id: str
    prompt: str
    reference: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared adapter capabilities for TRL/verl/vLLM or other frameworks."""

    backend_id: str
    supports_sft: bool = False
    supports_preferences: bool = False
    supports_online_rl: bool = False
    supports_external_verifiers: bool = False
    supports_vllm_rollouts: bool = False

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id must be non-empty")


@dataclass(frozen=True, slots=True)
class PostTrainingExperiment:
    """Fail-closed experiment declaration.

    A post-training run may consume a Base checkpoint as an immutable parent, but
    its output must always be a distinct POSTTRAIN lineage. This prevents an
    isolated experiment from being mislabeled as canonical Base.
    """

    experiment_id: str
    algorithm: str
    backend_id: str
    input_checkpoint: CheckpointRef
    output_lineage: LineageKind
    dataset_manifest_sha256: str
    seed: int
    compute_class: ComputeClass = ComputeClass.LOCAL_FREE
    compute_authorization_id: str | None = None
    verifier_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.algorithm.strip() or not self.backend_id.strip():
            raise ValueError("algorithm and backend_id must be non-empty")
        if self.output_lineage is LineageKind.BASE:
            raise ValueError("post-training output_lineage cannot be BASE")
        if not _SHA256_RE.fullmatch(self.dataset_manifest_sha256):
            raise ValueError("dataset_manifest_sha256 must be a lowercase 64-hex digest")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.compute_class is ComputeClass.MATERIAL_PAID and not (
            self.compute_authorization_id and self.compute_authorization_id.strip()
        ):
            raise ValueError("materially paid compute requires compute_authorization_id")
