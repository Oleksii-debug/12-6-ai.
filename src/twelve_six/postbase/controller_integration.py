"""Bridges the immutable POSTBASE-351 Base adapter to post-Base controllers.

The bridge deliberately transports request values, never controller-owned mutable
state. Base provenance remains on ``PostBaseModelAdapter.base_evidence`` while
this module records only post-Base generation/orchestration evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from twelve_six.inference.contracts import GenerationConfig
from twelve_six.postbase.adapter import (
    ControllerGenerationPort,
    ControllerGenerationRequest,
    PostBaseGenerationEvidence,
)
from twelve_six.postbase_deliberation import Request as DeliberationRequest
from twelve_six.postbase_deliberation import Response as DeliberationResponse
from twelve_six.postbase_hypothesis import Critique, Hypothesis, HypothesisSearch

INTEGRATION_VERSION = "12-6.next100-085.base-controller-integration.v1"
Operation = Literal[
    "deliberation",
    "hypothesis_propose",
    "hypothesis_critique",
    "hypothesis_revise",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ControllerCallEvidence:
    """Post-Base-only call evidence; Base checkpoint fields are intentionally absent."""

    evidence_namespace: Literal["post_base"]
    integration_version: str
    operation: Operation
    request_sha256: str
    generation: PostBaseGenerationEvidence

    def __post_init__(self) -> None:
        if self.generation.evidence_namespace != "post_base":
            raise ValueError("generation evidence must remain in post_base namespace")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DeliberationBaseBridge:
    """Implements POSTBASE-255's ModelAdapter protocol using POSTBASE-351."""

    def __init__(self, port: ControllerGenerationPort) -> None:
        self._port = port
        self._calls: list[ControllerCallEvidence] = []

    @property
    def post_base_evidence(self) -> tuple[ControllerCallEvidence, ...]:
        return tuple(self._calls)

    def generate(self, request: DeliberationRequest) -> DeliberationResponse:
        payload = {
            "task": request.task,
            "stage": request.stage,
            "branch_id": request.branch_id,
            "candidate_id": request.candidate_id,
            "iteration": request.iteration,
            "current_text": request.current_text,
            "critique": request.critique,
            "tool_results": list(request.tool_results),
        }
        prompt = _canonical_json(payload)
        max_new_tokens = request.max_generated_tokens
        if max_new_tokens is None:
            max_new_tokens = 64
        response = self._port.generate(
            ControllerGenerationRequest(
                controller="deliberation",
                prompt=prompt,
                config=GenerationConfig(max_new_tokens=max_new_tokens),
            )
        )
        self._calls.append(
            ControllerCallEvidence(
                evidence_namespace="post_base",
                integration_version=INTEGRATION_VERSION,
                operation="deliberation",
                request_sha256=_sha256_text(prompt),
                generation=response.post_base_evidence,
            )
        )
        return DeliberationResponse(
            text=response.generation.text,
            generated_tokens=len(response.generation.generated_token_ids),
        )


class HypothesisBaseBridge:
    """Model-assisted facade around POSTBASE-256's deterministic hypothesis graph.

    Generated text is merely an input to graph operations. No mechanics fixture
    or model output is interpreted here as evidence of reasoning quality.
    """

    def __init__(
        self,
        port: ControllerGenerationPort,
        search: HypothesisSearch | None = None,
    ) -> None:
        self._port = port
        self.search = search or HypothesisSearch()
        self._calls: list[ControllerCallEvidence] = []

    @property
    def post_base_evidence(self) -> tuple[ControllerCallEvidence, ...]:
        return tuple(self._calls)

    def propose(
        self,
        task: str,
        *,
        assumptions: tuple[str, ...] = (),
        initial_score: float = 0.5,
        max_new_tokens: int = 64,
    ) -> Hypothesis:
        text = self._generate(
            "hypothesis_propose",
            {"operation": "propose", "task": task, "assumptions": list(assumptions)},
            max_new_tokens,
        )
        return self.search.propose(
            text,
            assumptions=assumptions,
            initial_score=initial_score,
        )

    def critique(
        self,
        hypothesis_id: str,
        task: str,
        *,
        score_delta: float = 0.0,
        max_new_tokens: int = 64,
    ) -> Critique:
        hypothesis = self.search.hypothesis(hypothesis_id)
        text = self._generate(
            "hypothesis_critique",
            {
                "operation": "critique",
                "task": task,
                "hypothesis_id": hypothesis.id,
                "statement": hypothesis.statement,
            },
            max_new_tokens,
        )
        return self.search.critique(
            hypothesis_id,
            text,
            score_delta=score_delta,
        )

    def revise(
        self,
        hypothesis_id: str,
        task: str,
        *,
        assumptions: tuple[str, ...] | None = None,
        initial_score: float | None = None,
        max_new_tokens: int = 64,
    ) -> Hypothesis:
        hypothesis = self.search.hypothesis(hypothesis_id)
        text = self._generate(
            "hypothesis_revise",
            {
                "operation": "revise",
                "task": task,
                "hypothesis_id": hypothesis.id,
                "statement": hypothesis.statement,
            },
            max_new_tokens,
        )
        return self.search.revise(
            hypothesis_id,
            text,
            assumptions=assumptions,
            initial_score=initial_score,
        )

    def _generate(
        self,
        operation: Operation,
        payload: dict[str, object],
        max_new_tokens: int,
    ) -> str:
        prompt = _canonical_json(payload)
        response = self._port.generate(
            ControllerGenerationRequest(
                controller="hypothesis",
                prompt=prompt,
                config=GenerationConfig(max_new_tokens=max_new_tokens),
            )
        )
        self._calls.append(
            ControllerCallEvidence(
                evidence_namespace="post_base",
                integration_version=INTEGRATION_VERSION,
                operation=operation,
                request_sha256=_sha256_text(prompt),
                generation=response.post_base_evidence,
            )
        )
        text = response.generation.text.strip()
        if not text:
            raise ValueError("Base generation produced empty hypothesis-controller text")
        return text
