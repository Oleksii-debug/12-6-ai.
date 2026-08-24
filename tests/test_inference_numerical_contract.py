from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.generation import generate
from twelve_six.inference.openai_compat import CompletionRequest
from twelve_six.inference.sampling import sample_token


class MinimalBackend:
    eos_token_id = None
    max_context_tokens = 4

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(65 + token_id) for token_id in token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [4.0, 3.0]


class BoolContextBackend(MinimalBackend):
    max_context_tokens = True


class BoolPromptTokenBackend(MinimalBackend):
    def encode(self, text: str) -> list[int]:
        return [False]


class BadDecodeBackend(MinimalBackend):
    def decode(self, token_ids: Sequence[int]) -> str:  # type: ignore[override]
        return None  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_new_tokens": True}, ValueError),
        ({"max_new_tokens": 1.5}, ValueError),
        ({"sample": 1}, TypeError),
        ({"temperature": True}, TypeError),
        ({"temperature": math.nan}, ValueError),
        ({"temperature": math.inf}, ValueError),
        ({"top_k": True}, ValueError),
        ({"top_k": 1.5}, ValueError),
        ({"top_p": False}, TypeError),
        ({"top_p": math.nan}, ValueError),
        ({"seed": True}, TypeError),
        ({"stop_token_ids": (False,)}, ValueError),
        ({"stop_token_ids": (-1,)}, ValueError),
        ({"stop_strings": (1,)}, ValueError),  # type: ignore[arg-type]
        ({"strip_stop_strings": 1}, TypeError),
    ],
)
def test_generation_config_rejects_ambiguous_or_nonfinite_values(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        GenerationConfig(**kwargs)  # type: ignore[arg-type]


def test_sampling_remains_defined_at_smallest_positive_temperature() -> None:
    # The old divide-then-normalize path overflowed both finite logits to +inf,
    # producing NaN weights and eventually an empty-candidate IndexError.
    token = sample_token(
        [1_000.0, 999.0],
        rng=random.Random(17),
        temperature=5e-324,
    )
    assert token == 0


def test_sampling_direct_api_rejects_malformed_controls() -> None:
    with pytest.raises(TypeError, match="temperature"):
        sample_token([1.0, 0.0], rng=random.Random(1), temperature=True)
    with pytest.raises(ValueError, match="temperature"):
        sample_token([1.0, 0.0], rng=random.Random(1), temperature=math.inf)
    with pytest.raises(ValueError, match="top_k"):
        sample_token([1.0, 0.0], rng=random.Random(1), top_k=True)
    with pytest.raises(TypeError, match="top_p"):
        sample_token([1.0, 0.0], rng=random.Random(1), top_p=False)


def test_generation_rejects_ambiguous_backend_contract_values() -> None:
    with pytest.raises(ValueError, match="max_context_tokens"):
        generate(BoolContextBackend(), "x", GenerationConfig(max_new_tokens=1))
    with pytest.raises(ValueError, match="encoded prompt"):
        generate(BoolPromptTokenBackend(), "x", GenerationConfig(max_new_tokens=1))
    with pytest.raises(TypeError, match=r"decode\(\)"):
        generate(BadDecodeBackend(), "x", GenerationConfig(max_new_tokens=1))


def test_completion_request_rejects_json_scalar_type_coercion() -> None:
    with pytest.raises(TypeError, match="temperature"):
        CompletionRequest.from_payload({"prompt": "x", "temperature": "0.5"})
    with pytest.raises(TypeError, match="top_p"):
        CompletionRequest.from_payload({"prompt": "x", "top_p": True})
    with pytest.raises(ValueError, match="n=1"):
        CompletionRequest.from_payload({"prompt": "x", "n": True})
    with pytest.raises(TypeError, match="stream"):
        CompletionRequest.from_payload({"prompt": "x", "stream": 0})
    with pytest.raises(TypeError, match="echo"):
        CompletionRequest.from_payload({"prompt": "x", "echo": 0})


def test_seeded_sampling_repeatability_is_preserved() -> None:
    config = GenerationConfig(
        max_new_tokens=2,
        sample=True,
        temperature=0.7,
        top_p=0.9,
        seed=23,
    )
    assert generate(MinimalBackend(), "x", config) == generate(MinimalBackend(), "x", config)
