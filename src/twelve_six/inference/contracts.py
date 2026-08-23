from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal backend boundary required by the S0 generation harness."""

    eos_token_id: int | None

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
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0 when set")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if any(not stop for stop in self.stop_strings):
            raise ValueError("stop strings must not be empty")


StopReason = Literal["max_new_tokens", "eos", "stop_token", "stop_string"]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    text: str
    stop_reason: StopReason
