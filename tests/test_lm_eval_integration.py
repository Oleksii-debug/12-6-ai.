from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from twelve_six.integrations import lm_eval as integration


@dataclass(frozen=True)
class Request:
    args: tuple[object, ...]


class ToyBackend:
    eos_token_id = None
    max_context_tokens = 4

    def encode(self, text: str) -> list[int]:
        table = {"a": 0, "b": 1, "c": 2}
        return [table[char] for char in text]

    def decode(self, token_ids) -> str:
        table = {0: "a", 1: "b", 2: "c"}
        return "".join(table[token_id] for token_id in token_ids)

    def next_token_logits(self, input_ids):
        transitions = {
            0: (0.0, 2.0, 0.0),
            1: (0.0, 0.0, 2.0),
            2: (2.0, 0.0, 0.0),
        }
        return transitions[input_ids[-1]]


def _step_logprob() -> float:
    return 2.0 - math.log(math.exp(2.0) + 2.0)


def _core() -> integration.TwelveSixHarnessCore:
    return integration.TwelveSixHarnessCore(
        ToyBackend(),
        default_max_gen_toks=3,
        require_s0_byte_identity=False,
    )


def test_component_manifest_pins_reviewed_lm_eval_release() -> None:
    manifest = integration.component_manifest()
    assert manifest["distribution"] == "lm-eval"
    assert manifest["version"] == "0.4.12"
    assert (
        manifest["wheel_sha256"]
        == "02971ff68284dd14cfa7fce9310a58452c4162e8d413ba96aa7988a0ff9352ef"
    )
    assert manifest["foreign_pretrained_weights"] is False
    assert manifest["no_bos_policy"] == "first_token_unscored"


def test_loglikelihood_scores_only_continuation_tokens() -> None:
    score, greedy = _core().loglikelihood([Request(("a", "bc"))])[0]
    assert score == pytest.approx(2 * _step_logprob())
    assert greedy is True


def test_empty_context_uses_first_token_as_unscored_context() -> None:
    score, greedy = _core().loglikelihood([Request(("", "ab"))])[0]
    assert score == pytest.approx(_step_logprob())
    assert greedy is True


def test_rolling_scores_every_token_after_first_without_invented_bos() -> None:
    score = _core().loglikelihood_rolling([Request(("abc",))])[0]
    assert score == pytest.approx(2 * _step_logprob())


def test_generate_until_delegates_to_first_party_generation_contract() -> None:
    output = _core().generate_until(
        [Request(("a", {"until": ["c"], "max_gen_toks": 3}))]
    )
    assert output == ["b"]


def test_generation_rejects_unknown_harness_kwargs() -> None:
    with pytest.raises(ValueError, match="unsupported lm-eval generation kwargs"):
        _core().generate_until([Request(("a", {"beam_size": 4}))])


def test_exact_version_guard_rejects_other_lm_eval_releases(monkeypatch) -> None:
    monkeypatch.setattr(integration.metadata, "version", lambda _: "0.4.11")
    with pytest.raises(RuntimeError, match="expected exactly 0.4.12"):
        integration.require_lm_eval_version()
