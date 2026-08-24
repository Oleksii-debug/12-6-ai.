from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.openai_compat import CompletionRequest
from twelve_six.inference.sampling import sample_token


class MultiCharacterTokenBackend:
    eos_token_id = None
    max_context_tokens = 4

    def __init__(self, piece: str) -> None:
        self.piece = piece

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.piece for token_id in token_ids if token_id == 1)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [0.0, 10.0]


def test_stop_string_inside_one_decoded_token_stops_and_drops_suffix() -> None:
    backend = MultiCharacterTokenBackend("STOPtail")

    stripped = generate(
        backend,
        "x",
        GenerationConfig(max_new_tokens=2, stop_strings=("STOP",)),
    )
    kept = generate(
        backend,
        "x",
        GenerationConfig(
            max_new_tokens=2,
            stop_strings=("STOP",),
            strip_stop_strings=False,
        ),
    )

    assert stripped.generated_token_ids == kept.generated_token_ids == (1,)
    assert stripped.stop_reason == kept.stop_reason == "stop_string"
    assert stripped.text == ""
    assert kept.text == "STOP"


def test_earliest_text_stop_wins_independent_of_stop_argument_order() -> None:
    backend = MultiCharacterTokenBackend("xxSECONDyyFIRSTzz")
    config = GenerationConfig(
        max_new_tokens=1,
        stop_strings=("FIRST", "SECOND"),
        strip_stop_strings=False,
    )

    result = generate(backend, "x", config)

    assert result.stop_reason == "stop_string"
    assert result.text == "xxSECOND"


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"max_new_tokens": True}, TypeError),
        ({"sample": 1}, TypeError),
        ({"temperature": True}, TypeError),
        ({"temperature": float("nan")}, ValueError),
        ({"temperature": float("inf")}, ValueError),
        ({"top_k": True}, TypeError),
        ({"top_k": 0}, ValueError),
        ({"top_p": True}, TypeError),
        ({"top_p": float("nan")}, ValueError),
        ({"top_p": float("inf")}, ValueError),
        ({"seed": True}, TypeError),
        ({"stop_token_ids": (True,)}, TypeError),
        ({"stop_token_ids": (-1,)}, ValueError),
        ({"stop_strings": (1,)}, TypeError),
        ({"stop_strings": ("",)}, ValueError),
        ({"strip_stop_strings": 1}, TypeError),
    ],
)
def test_generation_config_rejects_ambiguous_or_nonfinite_values(
    kwargs: dict[str, object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        GenerationConfig(**kwargs)  # type: ignore[arg-type]


def test_sampling_is_stable_for_extremely_small_finite_temperature() -> None:
    token = sample_token(
        [1.0e308, 0.0, -float("inf")],
        rng=random.Random(17),
        temperature=5e-324,
    )

    assert token == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": True},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"top_k": True},
        {"top_p": True},
        {"top_p": float("nan")},
    ],
)
def test_sampling_rejects_ambiguous_or_nonfinite_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        sample_token([1.0, 0.0], rng=random.Random(0), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("max_tokens", True, TypeError),
        ("max_tokens", "1", TypeError),
        ("temperature", True, TypeError),
        ("temperature", "0", TypeError),
        ("temperature", float("nan"), ValueError),
        ("temperature", float("inf"), ValueError),
        ("top_p", True, TypeError),
        ("top_p", "1", TypeError),
        ("top_p", float("nan"), ValueError),
        ("seed", True, TypeError),
        ("n", True, TypeError),
        ("stream", 0, TypeError),
        ("echo", 0, TypeError),
    ],
)
def test_completion_request_rejects_json_type_coercion(
    field: str, value: object, error_type: type[Exception]
) -> None:
    payload: dict[str, object] = {"prompt": "raw"}
    payload[field] = value

    with pytest.raises(error_type):
        CompletionRequest.from_payload(payload)


def test_completion_request_accepts_explicit_supported_scalar_types() -> None:
    request = CompletionRequest.from_payload(
        {
            "prompt": "raw",
            "max_tokens": 0,
            "temperature": 0,
            "top_p": 1,
            "seed": -7,
            "n": 1,
            "stream": False,
            "echo": False,
        }
    )

    assert request.max_tokens == 0
    assert request.temperature == 0.0
    assert request.top_p == 1.0
    assert request.seed == -7
    config = request.generation_config()
    assert config.sample is False
    assert config.temperature == 1.0
