from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from twelve_six.inference.parity import compare_backends


class Backend:
    eos_token_id: int | None = None
    max_context_tokens: int = 8

    def __init__(
        self,
        *,
        prompt_tokens: Any = None,
        logits: Any = None,
        decoded: Any = "ok",
        encode_error: Exception | None = None,
        logits_error: Exception | None = None,
        decode_error: Exception | None = None,
    ) -> None:
        self.prompt_tokens = [0] if prompt_tokens is None else prompt_tokens
        self.logits = [2.0, 1.0] if logits is None else logits
        self.decoded = decoded
        self.encode_error = encode_error
        self.logits_error = logits_error
        self.decode_error = decode_error

    def encode(self, text: str) -> list[int]:
        if self.encode_error is not None:
            raise self.encode_error
        return self.prompt_tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        if self.decode_error is not None:
            raise self.decode_error
        return self.decoded

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        if self.logits_error is not None:
            raise self.logits_error
        return self.logits


def _failure_kinds(report) -> list[str]:
    return [failure.kind for failure in report.failures]


def test_finite_equal_backends_with_negative_infinity_mask_pass() -> None:
    reference = Backend(logits=[3.0, -float("inf"), 1.0])
    candidate = Backend(logits=[3.0, -float("inf"), 1.0])

    report = compare_backends(reference, candidate, ["probe"], max_new_tokens=2, atol=0, rtol=0)

    assert report.passed
    assert report.steps_compared == 2
    assert report.max_abs_error == 0.0
    assert report.max_rel_error == 0.0


@pytest.mark.parametrize(
    ("logits", "kind"),
    [
        ([float("inf"), 0.0], "invalid_reference_logits"),
        ([float("nan"), 0.0], "invalid_reference_logits"),
        ([-float("inf"), -float("inf")], "invalid_reference_logits"),
        ([], "invalid_reference_logits"),
        ([True, 0.0], "invalid_reference_logits"),
        (["1.0", 0.0], "invalid_reference_logits"),
    ],
)
def test_invalid_matching_reference_logits_fail_closed_without_sampler_exception(
    logits: Any,
    kind: str,
) -> None:
    report = compare_backends(
        Backend(logits=logits),
        Backend(logits=logits),
        ["probe"],
    )

    assert not report.passed
    assert _failure_kinds(report) == [kind]
    assert report.steps_compared == 0


def test_invalid_candidate_logits_fail_even_when_reference_is_valid() -> None:
    report = compare_backends(
        Backend(logits=[1.0, 0.0]),
        Backend(logits=[float("inf"), 0.0]),
        ["probe"],
    )

    assert not report.passed
    assert _failure_kinds(report) == ["invalid_candidate_logits"]
    assert report.steps_compared == 0


@pytest.mark.parametrize(
    ("atol", "rtol", "error_type"),
    [
        (True, 0.0, TypeError),
        (0.0, True, TypeError),
        ("0", 0.0, TypeError),
        (0.0, "0", TypeError),
        (float("nan"), 0.0, ValueError),
        (0.0, float("inf"), ValueError),
    ],
)
def test_tolerance_contract_rejects_coercion_and_nonfinite_values(
    atol: Any,
    rtol: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        compare_backends(Backend(), Backend(), ["probe"], atol=atol, rtol=rtol)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_max_new_tokens_rejects_bool_float_and_string(value: Any) -> None:
    with pytest.raises(TypeError, match="max_new_tokens"):
        compare_backends(Backend(), Backend(), ["probe"], max_new_tokens=value)


def test_invalid_backend_context_is_structured_failure() -> None:
    reference = Backend()
    reference.max_context_tokens = True  # type: ignore[assignment]

    report = compare_backends(reference, Backend(), ["probe"])

    assert not report.passed
    assert _failure_kinds(report) == ["invalid_reference_context_window"]
    assert report.prompts_compared == 0


def test_invalid_backend_eos_is_structured_failure() -> None:
    candidate = Backend()
    candidate.eos_token_id = True  # type: ignore[assignment]

    report = compare_backends(Backend(), candidate, ["probe"])

    assert not report.passed
    assert _failure_kinds(report) == ["invalid_candidate_eos_token"]


@pytest.mark.parametrize(
    ("prompt_tokens", "kind"),
    [
        ((0,), "invalid_reference_prompt_tokens"),
        ([], "empty_reference_prompt"),
        ([True], "invalid_reference_prompt_tokens"),
        ([-1], "invalid_reference_prompt_tokens"),
    ],
)
def test_reference_prompt_token_contract_fails_closed(prompt_tokens: Any, kind: str) -> None:
    report = compare_backends(
        Backend(prompt_tokens=prompt_tokens),
        Backend(prompt_tokens=prompt_tokens),
        ["probe"],
    )

    assert not report.passed
    assert _failure_kinds(report) == [kind]


def test_prompt_token_outside_runtime_logit_vocabulary_fails_closed() -> None:
    report = compare_backends(
        Backend(prompt_tokens=[4], logits=[3.0, 2.0]),
        Backend(prompt_tokens=[4], logits=[3.0, 2.0]),
        ["probe"],
    )

    assert not report.passed
    assert _failure_kinds(report) == ["input_token_out_of_range"]
    assert report.steps_compared == 0


def test_eos_token_outside_runtime_logit_vocabulary_fails_closed() -> None:
    reference = Backend(logits=[3.0, 2.0])
    candidate = Backend(logits=[3.0, 2.0])
    reference.eos_token_id = candidate.eos_token_id = 2

    report = compare_backends(reference, candidate, ["probe"])

    assert not report.passed
    assert _failure_kinds(report) == ["eos_token_out_of_range"]


def test_backend_exception_is_failure_without_secret_exception_message() -> None:
    secret = "PRIVATE-PROMPT-SHOULD-NOT-APPEAR"
    report = compare_backends(
        Backend(logits_error=ValueError(secret)),
        Backend(),
        [secret],
    )

    assert not report.passed
    failure = report.failures[0]
    assert failure.kind == "reference_next_token_logits_error"
    assert failure.detail == "reference next_token_logits raised ValueError"
    assert secret not in failure.detail


def test_non_string_decode_output_fails_closed() -> None:
    report = compare_backends(
        Backend(decoded=123),
        Backend(decoded=123),
        ["probe"],
        max_new_tokens=1,
    )

    assert not report.passed
    assert _failure_kinds(report) == ["invalid_decoded_text"]


def test_decode_exception_is_structured_failure() -> None:
    report = compare_backends(
        Backend(decode_error=RuntimeError("do not expose this")),
        Backend(),
        ["probe"],
        max_new_tokens=1,
    )

    assert not report.passed
    assert _failure_kinds(report) == ["reference_decode_error"]
    assert "do not expose this" not in report.failures[0].detail


def test_logit_size_mismatch_is_reported_before_vocab_or_greedy_use() -> None:
    report = compare_backends(
        Backend(logits=[2.0, 1.0]),
        Backend(logits=[2.0, 1.0, 0.0]),
        ["probe"],
    )

    assert not report.passed
    assert _failure_kinds(report) == ["logit_mismatch"]
    assert report.steps_compared == 0


def test_non_string_prompts_are_rejected_before_backend_execution() -> None:
    with pytest.raises(TypeError, match="prompts"):
        compare_backends(Backend(), Backend(), ["ok", 7])  # type: ignore[list-item]
