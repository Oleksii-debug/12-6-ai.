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
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not isinstance(self.sample, bool):
            raise TypeError("sample must be a boolean")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
        ):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
                raise TypeError("top_k must be an integer when set")
            if self.top_k <= 0:
                raise ValueError("top_k must be > 0 when set")
        if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool):
            raise TypeError("top_p must be a real number")
        if not math.isfinite(float(self.top_p)) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        for token_id in self.stop_token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("stop token IDs must be integers")
            if token_id < 0:
                raise ValueError("stop token IDs must be >= 0")
        for stop in self.stop_strings:
            if not isinstance(stop, str):
                raise TypeError("stop strings must be strings")
            if not stop:
                raise ValueError("stop strings must not be empty")
        if not isinstance(self.strip_stop_strings, bool):
            raise TypeError("strip_stop_strings must be a boolean")


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
