"""Read-only post-Base adapter over verified first-party Base checkpoints.

This module deliberately owns no model weights, trainer, checkpoint writer, remote
client, or chat/tool policy.  It turns the existing verified first-party Base
inference backend into a small generation port that deliberation and tool
controllers can call while keeping Base provenance and post-Base execution
evidence in distinct typed records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import torch

from twelve_six.inference.contracts import GenerationConfig, GenerationResult
from twelve_six.inference.first_party import (
    FirstPartyInferenceBackend,
    load_first_party_backend,
)
from twelve_six.inference.generation import generate as run_generation
from twelve_six.model import ModelSpec, canonical_json_sha256
from twelve_six.tokenization import ByteTokenizer


ADAPTER_VERSION = "12-6.postbase-model-adapter.v1"
ControllerKind = Literal["deliberation", "tool"]


class PostBaseCompatibilityError(ValueError):
    """Raised when a verified checkpoint is not eligible for this adapter."""


def validate_postbase_compatible_spec(spec: ModelSpec) -> None:
    """Fail closed on semantics the maintained first-party path cannot consume.

    There is intentionally no parameter-count or named-stage allowlist here.
    Any future ModelSpec-v1 geometry, including a compatible ~20M geometry, is
    eligible when it preserves the canonical byte-token contract and can be
    loaded by the maintained decoder/checkpoint implementation.
    """

    tokenizer = ByteTokenizer()
    if spec.schema_version != 1:
        raise PostBaseCompatibilityError(
            f"unsupported ModelSpec schema_version: {spec.schema_version}"
        )
    if spec.vocab_size != tokenizer.vocab_size:
        raise PostBaseCompatibilityError(
            "post-Base adapter requires the canonical s0-byte-v1 vocabulary: "
            f"model={spec.vocab_size} tokenizer={tokenizer.vocab_size}"
        )


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BaseCheckpointEvidence:
    """Read-only provenance copied from the verified Base checkpoint snapshot."""

    evidence_namespace: Literal["base"] = "base"
    checkpoint_id: str = ""
    git_sha: str = ""
    model_spec_sha256: str = ""
    parameter_count: int = 0
    vocab_size: int = 0
    max_context_tokens: int = 0
    tokenizer_version: str = ""
    tokenizer_config_sha256: str = ""
    tokenizer_vocab_sha256: str = ""
    dataset_manifest_sha256: str = ""
    run_manifest_sha256: str = ""
    step: int = 0
    tokens_seen: int = 0
    device: str = ""

    @classmethod
    def from_backend(cls, backend: FirstPartyInferenceBackend) -> BaseCheckpointEvidence:
        raw = backend.diagnostics()
        evidence = cls(
            checkpoint_id=str(raw["checkpoint_id"]),
            git_sha=str(raw["git_sha"]),
            model_spec_sha256=str(raw["model_spec_sha256"]),
            parameter_count=int(raw["parameter_count"]),
            vocab_size=int(raw["vocab_size"]),
            max_context_tokens=int(raw["max_context_tokens"]),
            tokenizer_version=str(raw["tokenizer_version"]),
            tokenizer_config_sha256=str(raw["tokenizer_config_sha256"]),
            tokenizer_vocab_sha256=str(raw["tokenizer_vocab_sha256"]),
            dataset_manifest_sha256=str(raw["dataset_manifest_sha256"]),
            run_manifest_sha256=str(raw["run_manifest_sha256"]),
            step=int(raw["step"]),
            tokens_seen=int(raw["tokens_seen"]),
            device=str(raw["device"]),
        )
        if evidence.step <= 0 or evidence.tokens_seen <= 0:
            raise PostBaseCompatibilityError(
                "post-Base adapter accepts learned Base checkpoints only: "
                "step and tokens_seen must both be positive"
            )
        return evidence

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControllerGenerationRequest:
    """Controller-neutral request; policy and tool execution remain outside Base."""

    controller: ControllerKind
    prompt: str
    config: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        if self.controller not in {"deliberation", "tool"}:
            raise ValueError("controller must be 'deliberation' or 'tool'")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")


@dataclass(frozen=True, slots=True)
class PostBaseGenerationEvidence:
    """Execution-only evidence.  It intentionally contains no Base metrics."""

    evidence_namespace: Literal["post_base"]
    adapter_version: str
    runtime_policy: Literal["LOCAL_FREE"]
    controller: ControllerKind
    generation_config_sha256: str
    prompt_utf8_sha256: str
    prompt_token_count: int
    generated_token_count: int
    generated_token_ids_sha256: str
    stop_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControllerGenerationResponse:
    """Generation plus two deliberately separate evidence namespaces."""

    generation: GenerationResult
    base_evidence: BaseCheckpointEvidence
    post_base_evidence: PostBaseGenerationEvidence


@runtime_checkable
class ControllerGenerationPort(Protocol):
    """Minimal surface consumed by deliberation or tool controllers."""

    def generate(self, request: ControllerGenerationRequest) -> ControllerGenerationResponse: ...


class PostBaseModelAdapter:
    """One local, read-only adapter over the maintained first-party Base runtime."""

    def __init__(self, backend: FirstPartyInferenceBackend) -> None:
        validate_postbase_compatible_spec(backend.model.spec)
        self._backend = backend
        self._base_evidence = BaseCheckpointEvidence.from_backend(backend)

        if backend.model.spec.identity_sha256() != self._base_evidence.model_spec_sha256:
            raise PostBaseCompatibilityError(
                "loaded model semantics do not match verified Base checkpoint evidence"
            )
        if backend.model.spec.parameter_count() != self._base_evidence.parameter_count:
            raise PostBaseCompatibilityError(
                "loaded model parameter count does not match verified Base checkpoint evidence"
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        expected_model_spec_sha256: str | None = None,
    ) -> PostBaseModelAdapter:
        """Consume one D05-verified immutable snapshot; never write the source path."""

        backend = load_first_party_backend(Path(checkpoint))
        adapter = cls(backend)
        if (
            expected_model_spec_sha256 is not None
            and adapter.base_evidence.model_spec_sha256 != expected_model_spec_sha256
        ):
            raise PostBaseCompatibilityError(
                "verified Base ModelSpec identity does not match controller expectation"
            )
        return adapter

    @property
    def base_evidence(self) -> BaseCheckpointEvidence:
        return self._base_evidence

    @property
    def model_spec(self) -> ModelSpec:
        """Return immutable semantic geometry, not a mutable model handle."""

        return self._backend.model.spec

    def generate(self, request: ControllerGenerationRequest) -> ControllerGenerationResponse:
        """Run local Base generation without gradient/update/checkpoint side effects."""

        config_payload = asdict(request.config)
        prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()

        with torch.inference_mode():
            result = run_generation(self._backend, request.prompt, request.config)

        generated_ids = list(result.generated_token_ids)
        post_base = PostBaseGenerationEvidence(
            evidence_namespace="post_base",
            adapter_version=ADAPTER_VERSION,
            runtime_policy="LOCAL_FREE",
            controller=request.controller,
            generation_config_sha256=canonical_json_sha256(config_payload),
            prompt_utf8_sha256=prompt_sha256,
            prompt_token_count=len(result.prompt_token_ids),
            generated_token_count=len(result.generated_token_ids),
            generated_token_ids_sha256=_sha256_json(generated_ids),
            stop_reason=result.stop_reason,
        )
        return ControllerGenerationResponse(
            generation=result,
            base_evidence=self._base_evidence,
            post_base_evidence=post_base,
        )
