"""Adapter interfaces for external post-training and rollout frameworks.

Implementations should wrap mature frameworks such as TRL or verl rather than
reimplementing their training loops in this repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import BackendCapabilities, Candidate, PostTrainingExperiment


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    request_id: str
    prompts: tuple[str, ...]
    generation: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.prompts:
            raise ValueError("rollout request must include at least one prompt")


@dataclass(frozen=True, slots=True)
class RunReceipt:
    run_id: str
    backend_id: str
    artifact_manifest_sha256: str
    metrics: Mapping[str, float]


@runtime_checkable
class RolloutBackend(Protocol):
    capabilities: BackendCapabilities

    def generate(self, request: RolloutRequest) -> Sequence[Candidate]:
        ...


@runtime_checkable
class PostTrainingBackend(Protocol):
    capabilities: BackendCapabilities

    def run(self, experiment: PostTrainingExperiment) -> RunReceipt:
        ...
