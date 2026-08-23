from collections.abc import Sequence

from twelve_six.inference.parity import PARITY_SCHEMA, compare_backends


class ParityBackend:
    eos_token_id = None
    max_context_tokens = 6

    def __init__(self, *, delta: float = 0.0) -> None:
        self.delta = delta

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(str(token_id) for token_id in token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [0.0, 2.0 + self.delta, 1.0]


class DivergentBackend(ParityBackend):
    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [0.0, 1.0, 2.0]


class EncodingMismatchBackend(ParityBackend):
    def encode(self, text: str) -> list[int]:
        return [1] if text else []


def test_identical_backends_pass() -> None:
    report = compare_backends(ParityBackend(), ParityBackend(), ["x"], max_new_tokens=3)
    assert report.passed
    assert report.steps_compared == 3
    assert report.max_abs_error == 0.0
    assert report.failures == ()


def test_small_logit_delta_within_tolerance_passes() -> None:
    report = compare_backends(
        ParityBackend(),
        ParityBackend(delta=1e-7),
        ["x"],
        max_new_tokens=2,
        atol=1e-6,
        rtol=0.0,
    )
    assert report.passed
    assert 0 < report.max_abs_error < 1e-6


def test_logit_delta_outside_tolerance_fails() -> None:
    report = compare_backends(
        ParityBackend(),
        ParityBackend(delta=1e-3),
        ["x"],
        max_new_tokens=2,
        atol=1e-6,
        rtol=0.0,
    )
    assert not report.passed
    assert report.failures[0].kind == "logit_mismatch"
    assert report.failures[0].prompt_index == 0
    assert report.failures[0].step_index == 0


def test_greedy_divergence_fails_even_with_wide_tolerance() -> None:
    report = compare_backends(
        ParityBackend(),
        DivergentBackend(),
        ["x"],
        max_new_tokens=1,
        atol=10.0,
        rtol=0.0,
    )
    assert not report.passed
    assert report.failures[0].kind == "greedy_token_mismatch"


def test_encoding_mismatch_is_reported_without_prompt_text() -> None:
    report = compare_backends(ParityBackend(), EncodingMismatchBackend(), ["secret prompt"])
    assert not report.passed
    assert report.failures[0].kind == "encoded_prompt_mismatch"
    assert "secret prompt" not in report.failures[0].detail


def test_context_window_mismatch_fails_contract() -> None:
    candidate = ParityBackend()
    candidate.max_context_tokens = 7
    report = compare_backends(ParityBackend(), candidate, ["x"])
    assert not report.passed
    assert report.prompts_compared == 0
    assert report.failures[0].kind == "context_window_mismatch"


def test_report_serializes_evidence_parameters() -> None:
    report = compare_backends(
        ParityBackend(),
        ParityBackend(),
        ["x"],
        max_new_tokens=1,
        atol=2e-6,
        rtol=3e-5,
    )
    payload = report.to_dict()
    assert payload["schema"] == PARITY_SCHEMA
    assert payload["passed"] is True
    assert payload["prompts_compared"] == 1
    assert payload["max_new_tokens"] == 1
    assert payload["atol"] == 2e-6
    assert payload["rtol"] == 3e-5
    assert payload["failures"] == []
