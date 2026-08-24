"""Dependency-free /v1/completions handoff for a future local HTTP server."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import GenerationConfig, GenerationResult, InferenceBackend
from .generation import generate


def _payload_number(payload: Mapping[str, Any], field: str, default: float) -> float:
    value = payload.get(field, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _payload_bool(payload: Mapping[str, Any], field: str, default: bool = False) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """Supported raw-Base subset of the OpenAI text-completions request shape."""

    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    stop: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CompletionRequest:
        if "messages" in payload:
            raise ValueError(
                "chat/messages semantics are not supported by raw canonical Base; use prompt"
            )
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        max_tokens = payload.get("max_tokens", 16)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        temperature = _payload_number(payload, "temperature", 1.0)
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        top_p = _payload_number(payload, "top_p", 1.0)
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        seed = payload.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")

        n = payload.get("n", 1)
        if not isinstance(n, int) or isinstance(n, bool) or n != 1:
            raise ValueError("only integer n=1 is supported by the minimal local completion handoff")
        if _payload_bool(payload, "stream"):
            raise ValueError("stream=true is not implemented by the minimal local handoff")
        if _payload_bool(payload, "echo"):
            raise ValueError("echo=true is not supported; responses contain completion text only")
        if payload.get("logprobs") is not None:
            raise ValueError("logprobs are not implemented by the minimal local handoff")

        return cls(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=_parse_stop(payload.get("stop")),
        )

    def generation_config(self) -> GenerationConfig:
        sample = self.temperature > 0
        return GenerationConfig(
            max_new_tokens=self.max_tokens,
            sample=sample,
            temperature=self.temperature if sample else 1.0,
            top_p=self.top_p,
            seed=self.seed,
            stop_strings=self.stop,
        )


def _parse_stop(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[str] = (value,)
    elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise TypeError("stop must be a string or a sequence of strings")
    if any(not item for item in values):
        raise ValueError("stop strings must not be empty")
    return tuple(values)


def _finish_reason(result: GenerationResult) -> str:
    if result.stop_reason in {"max_new_tokens", "context_limit"}:
        return "length"
    return "stop"


def completion_response(
    backend: InferenceBackend,
    payload: Mapping[str, Any],
    *,
    response_id: str = "cmpl-local",
    created: int = 0,
    model_name: str = "12-6-base",
) -> dict[str, object]:
    """Execute one raw completion and return a server-ready response mapping.

    This function adds no hidden prompt, system message, instruction template,
    refusal policy, or chat role. A future HTTP layer can expose it at
    ``POST /v1/completions`` and supply request-specific ``id``/``created``.
    """

    request = CompletionRequest.from_payload(payload)
    result = generate(backend, request.prompt, request.generation_config())
    prompt_tokens = len(result.prompt_token_ids)
    completion_tokens = len(result.generated_token_ids)
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": result.text,
                "finish_reason": _finish_reason(result),
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
