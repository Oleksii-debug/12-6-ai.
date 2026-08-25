from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.openai_compat import CompletionRequest
from twelve_six.inference.sampling import sample_token


class ContractBackend:
    eos_token_id: int | None = 3
    max_context_tokens: int = 8

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(65 + token_id) for token_id in token_ids if token_id < 3)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [0.0, 3.0, 2.0, 1.0]


class BoolContextBackend(ContractBackend):
    max_context_tokens = True  # type: ignore[assignment]


class BoolPromptTokenBackend(ContractBackend):
    def encode(self, text: str) -> list[int]:
        return [True]  # type: ignore[list-item]


class NegativePromptTokenBackend(ContractBackend):
    def encode(self, text: str) -> list[int]:
        return [-1]


class OutOfRangeEosBackend(ContractBackend):
    eos_token_id = 4


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"max_new_tokens": True}, TypeError, "max_new_tokens"),
        ({"sample": 1}, TypeError, "sample"),
        ({"temperature": True}, TypeError, "temperature"),
        ({"temperature": math.nan}, ValueError, "finite"),
        ({"temperature": math.inf}, ValueError, "finite"),
        ({"top_k": True}, TypeError, "top_k"),
        ({"top_k": 0}, ValueError, "top_k"),
        ({"top_p": True}, TypeError, "top_p"),
        ({"top_p": math.nan}, ValueError, "finite"),
        ({"top_p": math.inf}, ValueError, "finite"),
        ({"seed": True}, TypeError, "seed"),
        ({"stop_token_ids": (True,)}, TypeError, "stop_token_ids"),
        ({"stop_token_ids": (-1,)}, ValueError, "non-negative"),
        ({"stop_strings": (1,)}, TypeError, "stop_strings"),
        ({"stop_strings": ("",)}, ValueError, "must not be empty"),
        ({"strip_stop_strings": 1}, TypeError, "strip_stop_strings"),
    ],
)
def test_generation_config_rejects_ambiguous_or_nonfinite_values(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        GenerationConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"temperature": True}, TypeError, "temperature"),
        ({"temperature": math.nan}, ValueError, "finite"),
        ({"temperature": math.inf}, ValueError, "finite"),
        ({"top_k": True}, TypeError, "top_k"),
        ({"top_p": True}, TypeError, "top_p"),
        ({"top_p": math.nan}, ValueError, "finite"),
    ],
)
def test_sampler_rejects_invalid_scalar_contracts(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        sample_token(
            [0.0, 1.0],
            rng=random.Random(7),
            **kwargs,  # type: ignore[arg-type]
        )


def test_sampler_fails_closed_when_temperature_scaling_overflows() -> None:
    with pytest.raises(ValueError, match="temperature scaling"):
        sample_token(
            [1.0, 0.0],
            rng=random.Random(7),
            temperature=5e-324,
        )


def test_generate_rejects_boolean_context_capability() -> None:
    with pytest.raises(TypeError, match="max_context_tokens"):
        generate(BoolContextBackend(), "x")


@pytest.mark.parametrize("backend", [BoolPromptTokenBackend(), NegativePromptTokenBackend()])
def test_generate_rejects_invalid_encoded_prompt_tokens(backend: ContractBackend) -> None:
    with pytest.raises((TypeError, ValueError), match="prompt token IDs"):
        generate(backend, "x")


def test_generate_rejects_eos_outside_runtime_logits_vocab() -> None:
    with pytest.raises(ValueError, match="eos_token_id 4"):
        generate(OutOfRangeEosBackend(), "x", GenerationConfig(max_new_tokens=1))


def test_generate_rejects_stop_token_outside_runtime_logits_vocab() -> None:
    with pytest.raises(ValueError, match="outside logits vocabulary"):
        generate(
            ContractBackend(),
            "x",
            GenerationConfig(max_new_tokens=1, stop_token_ids=(4,)),
        )


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        ({"prompt": "x", "temperature": True}, TypeError, "temperature"),
        ({"prompt": "x", "temperature": "1.0"}, TypeError, "temperature"),
        ({"prompt": "x", "top_p": True}, TypeError, "top_p"),
        ({"prompt": "x", "top_p": "1.0"}, TypeError, "top_p"),
        ({"prompt": "x", "n": True}, TypeError, "n"),
        ({"prompt": "x", "stream": 0}, TypeError, "stream"),
        ({"prompt": "x", "echo": 0}, TypeError, "echo"),
    ],
)
def test_completion_payload_rejects_json_scalar_coercion(
    payload: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        CompletionRequest.from_payload(payload)


def test_completion_request_direct_construction_is_validated() -> None:
    with pytest.raises(ValueError, match="temperature"):
        CompletionRequest(prompt="x", temperature=math.nan)


def test_valid_seeded_sampling_contract_remains_deterministic() -> None:
    config = GenerationConfig(
        max_new_tokens=2,
        sample=True,
        temperature=0.75,
        top_k=3,
        top_p=0.9,
        seed=42,
    )
    assert generate(ContractBackend(), "x", config) == generate(
        ContractBackend(), "x", config
    )


def test_zero_temperature_completion_still_maps_to_greedy_generation() -> None:
    request = CompletionRequest.from_payload(
        {"prompt": "x", "temperature": 0.0, "top_p": 1.0}
    )
    config = request.generation_config()
    assert config.sample is False
    assert config.temperature == 1.0
