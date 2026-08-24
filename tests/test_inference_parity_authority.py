from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from twelve_six.inference.parity import ParityReport, compare_backends


@dataclass
class _Backend:
    max_context_tokens: Any = 8
    eos_token_id: Any = None
    encoded: Any = field(default_factory=lambda: [1, 2])
    logits: Any = field(default_factory=lambda: [0.1, 0.2, 0.9, -0.3])
    logits_calls: int = 0

    def encode(self, text: str):
        del text
        if isinstance(self.encoded, list):
            return list(self.encoded)
        return self.encoded

    def decode(self, token_ids):
        return ",".join(str(token_id) for token_id in token_ids)

    def next_token_logits(self, input_ids):
        del input_ids
        self.logits_calls += 1
        if isinstance(self.logits, list):
            return list(self.logits)
        return self.logits


def test_identical_backends_require_and_record_real_logit_steps() -> None:
    reference = _Backend()
    candidate = _Backend()

    report = compare_backends(reference, candidate, ["probe"], max_new_tokens=2)

    assert report.passed is True
    assert report.prompts_compared == 1
    assert report.steps_compared == 2
    assert reference.logits_calls == 2
    assert candidate.logits_calls == 2


@pytest.mark.parametrize("value", [0, -1])
def test_zero_or_negative_generation_budget_cannot_create_vacuous_pass(value: int) -> None:
    reference = _Backend()
    candidate = _Backend()

    with pytest.raises(ValueError, match="max_new_tokens must be > 0"):
        compare_backends(reference, candidate, ["probe"], max_new_tokens=value)

    assert reference.logits_calls == 0
    assert candidate.logits_calls == 0


def test_boolean_generation_budget_is_not_an_integer_contract() -> None:
    with pytest.raises(TypeError, match="positive integer"):
        compare_backends(_Backend(), _Backend(), ["probe"], max_new_tokens=True)


@pytest.mark.parametrize("name", ["atol", "rtol"])
@pytest.mark.parametrize("value", [True, "0", float("nan"), float("inf"), -1.0])
def test_invalid_tolerances_fail_before_comparison(name: str, value: object) -> None:
    kwargs = {name: value}
    with pytest.raises((TypeError, ValueError)):
        compare_backends(_Backend(), _Backend(), ["probe"], **kwargs)


def test_invalid_shared_context_contract_cannot_pass_without_steps() -> None:
    reference = _Backend(max_context_tokens=True)
    candidate = _Backend(max_context_tokens=True)

    report = compare_backends(reference, candidate, ["probe"])

    assert report.passed is False
    assert report.steps_compared == 0
    assert report.failures[0].kind == "invalid_reference_context_window"


@pytest.mark.parametrize("eos_token_id", [True, -1, 1.5])
def test_invalid_shared_eos_contract_fails_closed(eos_token_id: object) -> None:
    report = compare_backends(
        _Backend(eos_token_id=eos_token_id),
        _Backend(eos_token_id=eos_token_id),
        ["probe"],
    )

    assert report.passed is False
    assert report.steps_compared == 0
    assert report.failures[0].kind == "invalid_reference_eos_token"


def test_full_context_prompt_is_a_failure_not_zero_step_pass() -> None:
    reference = _Backend(max_context_tokens=2, encoded=[1, 2])
    candidate = _Backend(max_context_tokens=2, encoded=[1, 2])

    report = compare_backends(reference, candidate, ["probe"], max_new_tokens=1)

    assert report.passed is False
    assert report.steps_compared == 0
    assert [failure.kind for failure in report.failures] == ["no_logit_steps"]


def test_empty_logit_vectors_fail_before_greedy_selection() -> None:
    report = compare_backends(
        _Backend(logits=[]),
        _Backend(logits=[]),
        ["probe"],
        max_new_tokens=1,
    )

    assert report.passed is False
    assert report.steps_compared == 1
    assert report.failures[0].kind == "logit_mismatch"
    assert "must not be empty" in report.failures[0].detail


def test_eos_outside_runtime_vocab_fails_closed() -> None:
    report = compare_backends(
        _Backend(eos_token_id=7),
        _Backend(eos_token_id=7),
        ["probe"],
        max_new_tokens=1,
    )

    assert report.passed is False
    assert report.steps_compared == 1
    assert report.failures[0].kind == "eos_token_out_of_vocab"


def test_input_token_outside_runtime_vocab_fails_closed() -> None:
    report = compare_backends(
        _Backend(encoded=[1, 8]),
        _Backend(encoded=[1, 8]),
        ["probe"],
        max_new_tokens=1,
    )

    assert report.passed is False
    assert report.failures[0].kind == "input_token_out_of_vocab"


@pytest.mark.parametrize("encoded", [(1, 2), [1, True], [1, -1]])
def test_malformed_encoded_prompt_contract_fails_closed(encoded: object) -> None:
    report = compare_backends(
        _Backend(encoded=encoded),
        _Backend(encoded=encoded),
        ["probe"],
    )

    assert report.passed is False
    assert report.steps_compared == 0
    assert report.failures[0].kind == "invalid_reference_encoded_prompt"


def test_non_string_probe_is_rejected_before_backend_call() -> None:
    reference = _Backend()
    candidate = _Backend()

    with pytest.raises(TypeError, match="prompt at index 0 must be a string"):
        compare_backends(reference, candidate, [123])  # type: ignore[list-item]

    assert reference.logits_calls == 0
    assert candidate.logits_calls == 0


def test_report_object_itself_cannot_claim_pass_without_numerical_steps() -> None:
    report = ParityReport(
        prompts_compared=1,
        steps_compared=0,
        max_abs_error=0.0,
        max_rel_error=0.0,
        max_new_tokens=1,
        atol=0.0,
        rtol=0.0,
        failures=(),
    )

    assert report.passed is False
    assert report.to_dict()["passed"] is False
