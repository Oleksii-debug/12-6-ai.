from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal backend boundary required by the S0 generation harness."""

    eos_token_id: int | None
    max_context_tokens: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]: ...


def _finite_number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be a finite number")
    return numeric


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 64
    sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int = 0
    stop_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    strip_stop_strings: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not isinstance(self.sample, bool):
            raise TypeError("sample must be bool")

        temperature = _finite_number(self.temperature, field="temperature")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer when set")

        top_p = _finite_number(self.top_p, field="top_p")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in self.stop_token_ids
        ):
            raise ValueError("stop_token_ids must contain non-negative integers")
        if any(not isinstance(stop, str) or not stop for stop in self.stop_strings):
            raise ValueError("stop_strings must contain non-empty strings")
        if not isinstance(self.strip_stop_strings, bool):
            raise TypeError("strip_stop_strings must be bool")


StopReason = Literal[
    "max_new_tokens",
    "context_limit",
    "eos",
    "stop_token",
    "stop_string",
]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    text: str
    stop_reason: StopReason
