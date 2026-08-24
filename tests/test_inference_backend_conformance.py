from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

import twelve_six.inference.conformance as conformance
from twelve_six.inference.conformance import (
    AUTHORITY,
    SCHEMA_VERSION,
    run_backend_conformance,
    validate_conformance_report,
)


class GoodBackend:
    eos_token_id = None
    max_context_tokens = 8

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 4 for character in text]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(chr(ord("A") + token_id) for token_id in token_ids)

    def next_token_logits(self, input_ids: list[int] | tuple[int, ...]) -> list[float]:
        offset = float(sum(input_ids)) / 100.0
        return [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.4 + offset]


class BoolContextBackend(GoodBackend):
    max_context_tokens = True


class BoolTokenBackend(GoodBackend):
    def encode(self, text: str) -> list[int]:
        return [True]


class NonFiniteBackend(GoodBackend):
    def next_token_logits(self, input_ids: list[int] | tuple[int, ...]) -> list[float]:
        return [0.0, math.nan, 1.0, 2.0]


class NondeterministicBackend(GoodBackend):
    def __init__(self) -> None:
        self.calls = 0

    def next_token_logits(self, input_ids: list[int] | tuple[int, ...]) -> list[float]:
        self.calls += 1
        return [0.0, 1.0, 2.0, 3.0 + self.calls]


class WidthDriftBackend(GoodBackend):
    def next_token_logits(self, input_ids: list[int] | tuple[int, ...]) -> list[float]:
        if sum(input_ids) % 2:
            return [0.0, 1.0, 2.0, 3.0, 4.0]
        return [0.0, 1.0, 2.0, 3.0]


class InvalidEosBackend(GoodBackend):
    eos_token_id = 4


class FullContextBackend(GoodBackend):
    max_context_tokens = 1

    def encode(self, text: str) -> list[int]:
        return [0]


def test_backend_conformance_emits_tamper_evident_privacy_safe_report() -> None:
    report = run_backend_conformance(GoodBackend(), ("ab", "ba"))

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["authority"] == AUTHORITY
    assert report["inferred_vocab_size"] == 4
    assert report["max_context_tokens"] == 8
    assert report["probe_count"] == 2
    assert report["conformance_pass"] is True
    assert report["parity_proven"] is False
    assert report["checkpoint_identity_proven"] is False
    assert report["promotion_authority"] is False
    assert all(report["checks"].values())
    assert report["generation_probe"]["executed"] is True
    assert "ab" not in json.dumps(report, ensure_ascii=False)
    assert "ba" not in json.dumps(report, ensure_ascii=False)
    validate_conformance_report(report)


def test_backend_conformance_rejects_bool_context() -> None:
    with pytest.raises(TypeError, match="max_context_tokens"):
        run_backend_conformance(BoolContextBackend(), ("a",))


def test_backend_conformance_rejects_bool_token_id() -> None:
    with pytest.raises(TypeError, match="integer token ID"):
        run_backend_conformance(BoolTokenBackend(), ("a",))


def test_backend_conformance_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="finite"):
        run_backend_conformance(NonFiniteBackend(), ("a",))


def test_backend_conformance_rejects_repeat_drift() -> None:
    with pytest.raises(ValueError, match="not repeatable"):
        run_backend_conformance(NondeterministicBackend(), ("a",))


def test_backend_conformance_can_bound_small_repeat_delta() -> None:
    backend = NondeterministicBackend()
    report = run_backend_conformance(backend, ("a",), repeat_atol=2.0)
    assert report["probes"][0]["repeat_max_abs_delta"] == 1.0


def test_backend_conformance_rejects_vocab_width_drift() -> None:
    with pytest.raises(ValueError, match="vocabulary width changed across probes"):
        run_backend_conformance(WidthDriftBackend(), ("a", "b"))


def test_backend_conformance_rejects_eos_outside_vocab() -> None:
    with pytest.raises(ValueError, match="eos_token_id is outside"):
        run_backend_conformance(InvalidEosBackend(), ("a",))


def test_backend_conformance_requires_one_real_generation_step() -> None:
    with pytest.raises(ValueError, match="leaves room"):
        run_backend_conformance(FullContextBackend(), ("a",))


def test_conformance_report_validator_rejects_rehashed_overclaim() -> None:
    report = run_backend_conformance(GoodBackend(), ("a",))
    tampered = copy.deepcopy(report)
    tampered["parity_proven"] = True
    unhashed = dict(tampered)
    unhashed.pop("report_sha256")
    tampered["report_sha256"] = conformance._canonical_sha256(unhashed)
    with pytest.raises(ValueError, match="may not claim parity"):
        validate_conformance_report(tampered)


def test_conformance_cli_uses_existing_loader_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(conformance, "load_backend", lambda loader, path: GoodBackend())

    assert (
        conformance.main(
            [
                str(checkpoint),
                "--backend-loader",
                "fixture:load",
                "--prompt",
                "ab",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["conformance_pass"] is True
    assert payload["probe_count"] == 1
