from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.loader import load_backend


class SequenceBackend:
    eos_token_id = 3
    max_context_tokens = 4

    def __init__(self) -> None:
        self.seen_input_lengths: list[int] = []

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces = {0: "", 1: "A", 2: "B", 3: ""}
        return "".join(pieces[token_id] for token_id in token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        self.seen_input_lengths.append(len(input_ids))
        generated_count = len(input_ids) - 1
        next_ids = [1, 2, 3]
        next_id = next_ids[min(generated_count, len(next_ids) - 1)]
        logits = [0.0, 0.0, 0.0, 0.0]
        logits[next_id] = 10.0
        return logits


class LongPromptBackend(SequenceBackend):
    max_context_tokens = 2

    def encode(self, text: str) -> list[int]:
        return [0, 0, 0] if text else []


class LegacyBackend:
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [0]

    def decode(self, token_ids: Sequence[int]) -> str:
        return ""

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [1.0]


def test_greedy_generation_stops_on_eos() -> None:
    result = generate(SequenceBackend(), "x", GenerationConfig(max_new_tokens=10))
    assert result.generated_token_ids == (1, 2, 3)
    assert result.text == "AB"
    assert result.stop_reason == "eos"


def test_text_stop_can_be_stripped() -> None:
    config = GenerationConfig(max_new_tokens=10, stop_strings=("B",))
    result = generate(SequenceBackend(), "x", config)
    assert result.generated_token_ids == (1, 2)
    assert result.text == "A"
    assert result.stop_reason == "stop_string"


def test_seeded_sampling_repeats() -> None:
    config = GenerationConfig(max_new_tokens=2, sample=True, seed=9, temperature=1.0)
    first = generate(SequenceBackend(), "x", config)
    second = generate(SequenceBackend(), "x", config)
    assert first == second


def test_empty_encoded_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero tokens"):
        generate(SequenceBackend(), "")


def test_generation_stops_before_context_window_overflow() -> None:
    backend = SequenceBackend()
    backend.max_context_tokens = 2
    result = generate(backend, "x", GenerationConfig(max_new_tokens=10))
    assert result.generated_token_ids == (1,)
    assert result.stop_reason == "context_limit"
    assert backend.seen_input_lengths == [1]


def test_prompt_larger_than_context_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt has 3 tokens"):
        generate(LongPromptBackend(), "x")


def test_dynamic_loader_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    module = ModuleType("fake_d07_backend")
    module.make_backend = lambda path: SequenceBackend()  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)

    backend = load_backend("fake_d07_backend:make_backend", checkpoint)
    assert isinstance(backend, SequenceBackend)


def test_dynamic_loader_rejects_backend_without_context_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    module = ModuleType("legacy_d07_backend")
    module.make_backend = lambda path: LegacyBackend()  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)

    with pytest.raises(TypeError, match="max_context_tokens"):
        load_backend("legacy_d07_backend:make_backend", checkpoint)
