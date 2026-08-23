"""Dry-run-only rollout planning for future vLLM/TRL/verl integration.

The planner normalizes requests, versions, and provenance without importing or
executing an external inference or training runtime. It cannot mutate weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .compatibility import (
    CURRENT_RUNTIME_COMPATIBILITY,
    RuntimeCompatibilitySnapshot,
    SemanticVersion,
)
from .contracts import CheckpointRef
from .interfaces import RolloutRequest
from .provenance import canonical_sha256


class RolloutTarget(StrEnum):
    VLLM_OFFLINE = "vllm_offline"
    TRL_VLLM_SERVER = "trl_vllm_server"
    VERL_VLLM = "verl_vllm"


class RolloutPlanningError(ValueError):
    """Raised when a dry-run rollout declaration is invalid or incompatible."""


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0
    num_generations: int = 1

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise RolloutPlanningError("max_tokens must be > 0")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise RolloutPlanningError("temperature must be finite and >= 0")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise RolloutPlanningError("top_p must be finite and in (0, 1]")
        if self.top_k < -1:
            raise RolloutPlanningError("top_k must be -1 or >= 0")
        if self.seed < 0:
            raise RolloutPlanningError("seed must be non-negative")
        if self.num_generations <= 0:
            raise RolloutPlanningError("num_generations must be > 0")

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> SamplingPlan:
        allowed = {
            "max_new_tokens",
            "max_tokens",
            "n",
            "num_generations",
            "seed",
            "temperature",
            "top_k",
            "top_p",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise RolloutPlanningError(f"unsupported generation keys: {unknown}")
        if "max_tokens" in values and "max_new_tokens" in values:
            raise RolloutPlanningError("use only one of max_tokens or max_new_tokens")
        max_value = values.get("max_tokens", values.get("max_new_tokens"))
        if max_value is None:
            raise RolloutPlanningError("generation mapping requires max_tokens")
        if "n" in values and "num_generations" in values:
            raise RolloutPlanningError("use only one of n or num_generations")
        generation_count = values.get("num_generations", values.get("n", "1"))
        try:
            parsed = {
                "max_tokens": int(max_value),
                "temperature": float(values.get("temperature", "1.0")),
                "top_p": float(values.get("top_p", "1.0")),
                "top_k": int(values.get("top_k", "-1")),
                "seed": int(values.get("seed", "0")),
                "num_generations": int(generation_count),
            }
        except ValueError as exc:
            raise RolloutPlanningError("generation values must be numeric") from exc
        return cls(**parsed)

    def as_vllm_sampling(self) -> dict[str, int | float]:
        return {
            "max_tokens": self.max_tokens,
            "n": self.num_generations,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


@dataclass(frozen=True, slots=True)
class DryRunRolloutPlan:
    """Content-addressed rollout plan that explicitly cannot execute."""

    target: RolloutTarget
    request: RolloutRequest
    checkpoint: CheckpointRef
    sampling: SamplingPlan
    vllm_version: SemanticVersion
    compatibility_snapshot_id: str
    trl_version: SemanticVersion | None = None
    verl_version: SemanticVersion | None = None
    execution_enabled: bool = False

    def __post_init__(self) -> None:
        if self.execution_enabled:
            raise RolloutPlanningError("dry-run rollout plans cannot enable execution")
        if not self.compatibility_snapshot_id.strip():
            raise RolloutPlanningError("compatibility_snapshot_id must be non-empty")
        if self.target is RolloutTarget.TRL_VLLM_SERVER and self.trl_version is None:
            raise RolloutPlanningError("TRL vLLM plans require trl_version")
        if self.target is RolloutTarget.VERL_VLLM and self.verl_version is None:
            raise RolloutPlanningError("verl vLLM plans require verl_version")

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(
            {
                "generation": dict(self.request.generation),
                "prompts": list(self.request.prompts),
                "request_id": self.request.request_id,
            }
        )

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(self.normalized_payload())

    def normalized_payload(self) -> dict[str, Any]:
        runtime: dict[str, str] = {"vllm": str(self.vllm_version)}
        if self.trl_version is not None:
            runtime["trl"] = str(self.trl_version)
        if self.verl_version is not None:
            runtime["verl"] = str(self.verl_version)
        return {
            "checkpoint": {
                "checkpoint_id": self.checkpoint.checkpoint_id,
                "git_sha": self.checkpoint.git_sha,
                "lineage": self.checkpoint.lineage.value,
                "sha256": self.checkpoint.sha256,
                "stage": self.checkpoint.stage,
            },
            "compatibility_snapshot_id": self.compatibility_snapshot_id,
            "execution_enabled": False,
            "prompts": list(self.request.prompts),
            "request_id": self.request.request_id,
            "request_sha256": self.request_sha256,
            "runtime": runtime,
            "sampling": self.sampling.as_vllm_sampling(),
            "target": self.target.value,
        }


def build_dry_run_rollout_plan(
    request: RolloutRequest,
    checkpoint: CheckpointRef,
    target: RolloutTarget,
    *,
    snapshot: RuntimeCompatibilitySnapshot = CURRENT_RUNTIME_COMPATIBILITY,
    vllm_version: SemanticVersion | None = None,
) -> DryRunRolloutPlan:
    """Build a no-execution plan and fail closed on observed version incompatibility."""

    selected_vllm = vllm_version or snapshot.vllm_selected_version
    trl_version: SemanticVersion | None = None
    verl_version: SemanticVersion | None = None
    if target is RolloutTarget.TRL_VLLM_SERVER:
        if not snapshot.trl_supports_vllm(selected_vllm):
            raise RolloutPlanningError(
                f"TRL {snapshot.trl_version} does not support vLLM {selected_vllm} in this snapshot"
            )
        trl_version = snapshot.trl_version
    elif target is RolloutTarget.VERL_VLLM:
        if not snapshot.verl_supports_vllm(selected_vllm):
            raise RolloutPlanningError(
                f"verl {snapshot.verl_version} does not support vLLM {selected_vllm} in this snapshot"
            )
        verl_version = snapshot.verl_version

    return DryRunRolloutPlan(
        target=target,
        request=request,
        checkpoint=checkpoint,
        sampling=SamplingPlan.from_mapping(request.generation),
        vllm_version=selected_vllm,
        compatibility_snapshot_id=snapshot.snapshot_id,
        trl_version=trl_version,
        verl_version=verl_version,
    )
