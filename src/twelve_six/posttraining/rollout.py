"""Framework-neutral rollout contracts for candidate generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < -1:
            raise ValueError("top_k must be -1 or >= 0")


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    request_id: str
    prompt: str
    sampling: SamplingSpec
    num_candidates: int = 1

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.prompt:
            raise ValueError("request_id and prompt must be non-empty")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be > 0")


@dataclass(frozen=True, slots=True)
class RolloutCandidate:
    request_id: str
    candidate_id: str
    text: str
    finish_reason: str | None = None
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.candidate_id.strip() or not self.text:
            raise ValueError("candidate identifiers and text must be non-empty")
        if self.logprobs and len(self.logprobs) != len(self.token_ids):
            raise ValueError("logprobs must align one-to-one with token_ids")


class RolloutProvider(Protocol):
    """Adapter boundary for vLLM, verl rollouts, or future serving backends."""

    provider_id: str
    provider_version: str

    def generate(self, requests: Sequence[RolloutRequest]) -> Sequence[RolloutCandidate]:
        ...
