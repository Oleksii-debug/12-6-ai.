from __future__ import annotations

import math
from types import MappingProxyType

import pytest

from twelve_six.tokenization.base import TokenizerIdentity
from twelve_six.tokenization.byte import ByteTokenizer
from twelve_six.tokenization.calibration import (
    MEASUREMENT_STATUS,
    TokenizerCalibrationError,
    bits_per_byte_from_nll_nats,
    calibrate_tokenizer_efficiency,
)


SAMPLES = {
    "uk": ["Привіт, світе!", "Модель навчається на точних даних."],
    "en": ["Hello, world!", "Measure the tokenizer in stable units."],
    "code": ["def f(x):\n    return x + 1\n", "items = [x * 2 for x in range(4)]\n"],
}


class CodepointTokenizer:
    pad_id = None
    bos_id = None
    eos_id = None
    vocab_size = 0x110000
    version = "test-codepoint-v1"

    @property
    def identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(
            version=self.version,
            config_sha256="1" * 64,
            vocab_sha256="2" * 64,
            vocab_size=self.vocab_size,
            normalization="none",
            encoding="unicode-codepoint",
            special_tokens=MappingProxyType({}),
        )

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        assert not add_bos and not add_eos
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        del skip_special_tokens, errors
        return "".join(chr(token_id) for token_id in token_ids)


class BrokenRoundtripTokenizer(CodepointTokenizer):
    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        del token_ids, skip_special_tokens, errors
        return "corrupted"


def test_byte_tokenizer_reports_exact_one_byte_per_token() -> None:
    report = calibrate_tokenizer_efficiency(ByteTokenizer(), SAMPLES, context_tokens=1024)

    assert report.status == MEASUREMENT_STATUS
    assert report.tokenizer_identity["version"] == "s0-byte-v1"
    assert report.aggregate.bytes_per_token == pytest.approx(1.0)
    assert report.aggregate.tokens_per_byte == pytest.approx(1.0)
    assert report.aggregate.estimated_utf8_bytes_per_context == pytest.approx(1024.0)
    assert report.aggregate.roundtrip_exact is True
    assert len(report.report_sha256) == 64
    for item in report.strata:
        assert item.bytes_per_token == pytest.approx(1.0)


def test_multibyte_language_geometry_is_not_mistaken_for_token_budget() -> None:
    report = calibrate_tokenizer_efficiency(CodepointTokenizer(), SAMPLES, context_tokens=1024)
    by_name = {item.stratum: item for item in report.strata}

    assert by_name["uk"].bytes_per_token > by_name["en"].bytes_per_token
    assert by_name["uk"].estimated_utf8_bytes_per_context > 1024.0
    assert report.status == "MEASUREMENT_ONLY_NOT_TRAINING_AUTHORIZATION"


def test_sample_identity_is_key_order_independent_but_content_bound() -> None:
    reordered = {"code": SAMPLES["code"], "en": SAMPLES["en"], "uk": SAMPLES["uk"]}
    first = calibrate_tokenizer_efficiency(ByteTokenizer(), SAMPLES, context_tokens=1024)
    second = calibrate_tokenizer_efficiency(ByteTokenizer(), reordered, context_tokens=1024)
    changed = {**SAMPLES, "en": [*SAMPLES["en"], "new exact sample"]}
    third = calibrate_tokenizer_efficiency(ByteTokenizer(), changed, context_tokens=1024)

    assert first.corpus_sha256 == second.corpus_sha256
    assert first.corpus_sha256 != third.corpus_sha256


def test_missing_required_stratum_fails_closed() -> None:
    with pytest.raises(TokenizerCalibrationError, match="missing required strata: code"):
        calibrate_tokenizer_efficiency(
            ByteTokenizer(), {"uk": ["текст"], "en": ["text"]}, context_tokens=1024
        )


def test_roundtrip_corruption_fails_closed() -> None:
    with pytest.raises(TokenizerCalibrationError, match="roundtrip mismatch"):
        calibrate_tokenizer_efficiency(BrokenRoundtripTokenizer(), SAMPLES, context_tokens=1024)


def test_bits_per_byte_uses_utf8_bytes_not_token_count() -> None:
    byte_count = 137
    nll_nats = byte_count * math.log(2.0)
    assert bits_per_byte_from_nll_nats(nll_nats, byte_count) == pytest.approx(1.0)


@pytest.mark.parametrize("bad_bytes", [0, -1, True])
def test_bits_per_byte_rejects_invalid_byte_counts(bad_bytes) -> None:
    with pytest.raises(TokenizerCalibrationError):
        bits_per_byte_from_nll_nats(1.0, bad_bytes)


@pytest.mark.parametrize("bad_nll", [-1.0, float("inf"), float("nan"), True])
def test_bits_per_byte_rejects_invalid_nll(bad_nll) -> None:
    with pytest.raises(TokenizerCalibrationError):
        bits_per_byte_from_nll_nats(bad_nll, 1)
