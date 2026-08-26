"""Tokenizer-efficiency calibration in tokenizer-neutral UTF-8 units.

This module measures tokenization geometry only. It does not fit a tokenizer, mutate a
corpus, authorize training, or convert external tokens-per-parameter ratios into byte
loss-position budgets.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .base import TokenizerProtocol

SCHEMA_VERSION = "12-6.tokenizer-efficiency-calibration.v1"
MEASUREMENT_STATUS = "MEASUREMENT_ONLY_NOT_TRAINING_AUTHORIZATION"
DEFAULT_REQUIRED_STRATA = ("uk", "en", "code")


class TokenizerCalibrationError(ValueError):
    """Raised when a calibration input or tokenizer fails a fail-closed invariant."""


@dataclass(frozen=True)
class StratumEfficiency:
    stratum: str
    sample_count: int
    utf8_bytes: int
    unicode_codepoints: int
    token_count: int
    bytes_per_token: float
    tokens_per_byte: float
    codepoints_per_token: float
    context_tokens: int
    estimated_utf8_bytes_per_context: float
    roundtrip_exact: bool


@dataclass(frozen=True)
class TokenizerCalibrationReport:
    schema_version: str
    status: str
    corpus_sha256: str
    tokenizer_identity: Mapping[str, object]
    context_tokens: int
    required_strata: tuple[str, ...]
    strata: tuple[StratumEfficiency, ...]
    aggregate: StratumEfficiency

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "corpus_sha256": self.corpus_sha256,
            "tokenizer_identity": dict(self.tokenizer_identity),
            "context_tokens": self.context_tokens,
            "required_strata": list(self.required_strata),
            "strata": [asdict(item) for item in self.strata],
            "aggregate": asdict(self.aggregate),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def report_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def bits_per_byte_from_nll_nats(total_nll_nats: float, utf8_bytes: int) -> float:
    """Convert total natural-log NLL over an exact text payload to bits per UTF-8 byte."""
    if not isinstance(utf8_bytes, int) or isinstance(utf8_bytes, bool) or utf8_bytes <= 0:
        raise TokenizerCalibrationError("utf8_bytes must be a positive integer")
    if not isinstance(total_nll_nats, (int, float)) or isinstance(total_nll_nats, bool):
        raise TokenizerCalibrationError("total_nll_nats must be numeric")
    value = float(total_nll_nats)
    if not math.isfinite(value) or value < 0.0:
        raise TokenizerCalibrationError("total_nll_nats must be finite and non-negative")
    return value / (math.log(2.0) * utf8_bytes)


def _canonical_samples(samples: Mapping[str, Sequence[str]]) -> tuple[dict[str, list[str]], str]:
    normalized: dict[str, list[str]] = {}
    for stratum, texts in samples.items():
        if not isinstance(stratum, str) or not stratum.strip():
            raise TokenizerCalibrationError("stratum names must be non-empty strings")
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TokenizerCalibrationError(f"stratum {stratum!r} must contain a text sequence")
        materialized: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TokenizerCalibrationError(
                    f"stratum {stratum!r} sample {index} must be str"
                )
            if not text:
                raise TokenizerCalibrationError(
                    f"stratum {stratum!r} sample {index} must not be empty"
                )
            materialized.append(text)
        if not materialized:
            raise TokenizerCalibrationError(f"stratum {stratum!r} must not be empty")
        normalized[stratum] = materialized

    payload = {
        "schema_version": 1,
        "encoding": "utf-8",
        "strata": {key: normalized[key] for key in sorted(normalized)},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    identity = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return normalized, identity


def _measure_stratum(
    tokenizer: TokenizerProtocol,
    stratum: str,
    texts: Sequence[str],
    *,
    context_tokens: int,
) -> StratumEfficiency:
    utf8_bytes = 0
    unicode_codepoints = 0
    token_count = 0

    for index, text in enumerate(texts):
        encoded = tokenizer.encode(text, add_bos=False, add_eos=False)
        if not encoded:
            raise TokenizerCalibrationError(
                f"tokenizer emitted zero tokens for non-empty {stratum!r} sample {index}"
            )
        decoded = tokenizer.decode(encoded, skip_special_tokens=True, errors="strict")
        if decoded != text:
            raise TokenizerCalibrationError(
                f"tokenizer roundtrip mismatch for {stratum!r} sample {index}"
            )
        utf8_bytes += len(text.encode("utf-8"))
        unicode_codepoints += len(text)
        token_count += len(encoded)

    if utf8_bytes <= 0 or token_count <= 0:
        raise TokenizerCalibrationError(f"stratum {stratum!r} has no measurable payload")

    bytes_per_token = utf8_bytes / token_count
    return StratumEfficiency(
        stratum=stratum,
        sample_count=len(texts),
        utf8_bytes=utf8_bytes,
        unicode_codepoints=unicode_codepoints,
        token_count=token_count,
        bytes_per_token=bytes_per_token,
        tokens_per_byte=token_count / utf8_bytes,
        codepoints_per_token=unicode_codepoints / token_count,
        context_tokens=context_tokens,
        estimated_utf8_bytes_per_context=bytes_per_token * context_tokens,
        roundtrip_exact=True,
    )


def calibrate_tokenizer_efficiency(
    tokenizer: TokenizerProtocol,
    samples: Mapping[str, Sequence[str]],
    *,
    context_tokens: int,
    required_strata: Sequence[str] = DEFAULT_REQUIRED_STRATA,
) -> TokenizerCalibrationReport:
    """Measure one tokenizer on an exact multilingual/code sample set.

    The output binds the tokenizer identity and exact sample payload hash. Metrics are
    descriptive and scale-neutral; callers must not treat them as a training budget.
    """
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool):
        raise TokenizerCalibrationError("context_tokens must be an integer")
    if context_tokens <= 0:
        raise TokenizerCalibrationError("context_tokens must be positive")

    required = tuple(required_strata)
    if not required or any(not isinstance(item, str) or not item for item in required):
        raise TokenizerCalibrationError("required_strata must contain non-empty strings")
    if len(set(required)) != len(required):
        raise TokenizerCalibrationError("required_strata must not contain duplicates")

    normalized, corpus_sha256 = _canonical_samples(samples)
    missing = [item for item in required if item not in normalized]
    if missing:
        raise TokenizerCalibrationError(f"missing required strata: {', '.join(missing)}")

    measured = tuple(
        _measure_stratum(tokenizer, stratum, normalized[stratum], context_tokens=context_tokens)
        for stratum in sorted(normalized)
    )
    aggregate = StratumEfficiency(
        stratum="aggregate",
        sample_count=sum(item.sample_count for item in measured),
        utf8_bytes=sum(item.utf8_bytes for item in measured),
        unicode_codepoints=sum(item.unicode_codepoints for item in measured),
        token_count=sum(item.token_count for item in measured),
        bytes_per_token=0.0,
        tokens_per_byte=0.0,
        codepoints_per_token=0.0,
        context_tokens=context_tokens,
        estimated_utf8_bytes_per_context=0.0,
        roundtrip_exact=all(item.roundtrip_exact for item in measured),
    )
    aggregate = StratumEfficiency(
        **{
            **asdict(aggregate),
            "bytes_per_token": aggregate.utf8_bytes / aggregate.token_count,
            "tokens_per_byte": aggregate.token_count / aggregate.utf8_bytes,
            "codepoints_per_token": aggregate.unicode_codepoints / aggregate.token_count,
            "estimated_utf8_bytes_per_context": (
                aggregate.utf8_bytes / aggregate.token_count * context_tokens
            ),
        }
    )

    return TokenizerCalibrationReport(
        schema_version=SCHEMA_VERSION,
        status=MEASUREMENT_STATUS,
        corpus_sha256=corpus_sha256,
        tokenizer_identity=tokenizer.identity.to_dict(),
        context_tokens=context_tokens,
        required_strata=required,
        strata=measured,
        aggregate=aggregate,
    )
