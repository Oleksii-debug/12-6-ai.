"""Fail-closed experiment boundary that protects the canonical Base lineage."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import ArtifactRef, ExecutionMode, FrameworkKind, PostTrainingMethod


class BaseLineageViolation(RuntimeError):
    """Raised when post-training configuration could mutate canonical Base artifacts."""


class BehavioralTrainingNotAuthorized(RuntimeError):
    """Raised when a real behavioral-training run lacks an explicit owner decision."""


@dataclass(frozen=True, slots=True)
class PostTrainingExperimentConfig:
    experiment_id: str
    method: PostTrainingMethod
    source_checkpoint: ArtifactRef
    dataset_manifest_sha256: str
    output_lineage: str
    framework: FrameworkKind = FrameworkKind.CONTRACT_ONLY
    execution_mode: ExecutionMode = ExecutionMode.CONTRACT_ONLY
    owner_behavioral_training_authorization: str | None = None
    mutate_canonical_base: bool = False

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.dataset_manifest_sha256 or len(self.dataset_manifest_sha256) != 64:
            raise ValueError("dataset_manifest_sha256 must be a 64-character SHA-256 digest")
        if any(char not in "0123456789abcdef" for char in self.dataset_manifest_sha256):
            raise ValueError("dataset_manifest_sha256 must be lowercase hexadecimal")
        if not self.output_lineage.strip():
            raise ValueError("output_lineage must be non-empty")

    def assert_isolated_from_base(self) -> None:
        normalized = self.output_lineage.strip().lower().replace("\\", "/")
        canonical_base_target = normalized == "base" or normalized.startswith("base/")
        if self.mutate_canonical_base or canonical_base_target:
            raise BaseLineageViolation(
                "post-training outputs must not target canonical Base lineage"
            )

    def assert_execution_allowed(self) -> None:
        self.assert_isolated_from_base()
        if self.execution_mode is ExecutionMode.TRAIN:
            authorization = self.owner_behavioral_training_authorization
            if not authorization or not authorization.strip():
                raise BehavioralTrainingNotAuthorized(
                    "behavioral weight training requires explicit owner authorization"
                )
