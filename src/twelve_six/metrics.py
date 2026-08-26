"""Tokenizer-agnostic language-model metrics.

The primary primitive here is bits per byte (BPB).  BPB normalizes negative
log-likelihood by the number of raw bytes represented by the scored targets,
so it remains meaningful when tokenizer segmentation or vocabulary size changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, log
from numbers import Integral, Real

_LN_2 = log(2.0)


def _checked_nll(value: Real, *, field: str = "negative log-likelihood") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _checked_byte_count(value: Integral, *, field: str = "UTF-8 byte count") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _checked_token_count(value: Integral, *, field: str = "predicted token count") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class BPBTotals:
    """Additive sufficient statistics for exact BPB aggregation.

    ``total_nll_nats`` and ``total_utf8_bytes`` must describe the same scored
    target coverage. ``predicted_tokens`` counts only targets with positive raw
    byte length; zero-byte special tokens are intentionally excluded.
    """

    total_nll_nats: float
    total_utf8_bytes: int
    predicted_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "total_nll_nats",
            _checked_nll(self.total_nll_nats, field="total_nll_nats"),
        )
        object.__setattr__(
            self,
            "total_utf8_bytes",
            _checked_byte_count(self.total_utf8_bytes, field="total_utf8_bytes"),
        )
        object.__setattr__(
            self,
            "predicted_tokens",
            _checked_token_count(self.predicted_tokens, field="predicted_tokens"),
        )
        if self.total_utf8_bytes == 0 and self.total_nll_nats != 0.0:
            raise ValueError("non-zero NLL cannot have zero scored bytes")
        if self.total_utf8_bytes == 0 and self.predicted_tokens != 0:
            raise ValueError("predicted_tokens must be zero when scored bytes are zero")
        if self.total_utf8_bytes > 0 and self.predicted_tokens == 0:
            raise ValueError("positive scored bytes require at least one predicted token")

    @property
    def bits_per_byte(self) -> float:
        """Return total NLL in bits divided by scored raw bytes."""

        if self.total_utf8_bytes <= 0:
            raise ValueError("bits_per_byte is undefined for zero scored bytes")
        return self.total_nll_nats / (_LN_2 * self.total_utf8_bytes)

    @property
    def mean_nll_nats_per_token(self) -> float:
        """Return mean NLL for the included positive-byte target tokens."""

        if self.predicted_tokens <= 0:
            raise ValueError("mean NLL is undefined for zero predicted tokens")
        return self.total_nll_nats / self.predicted_tokens

    @property
    def bytes_per_token(self) -> float:
        """Return scored raw bytes per included predicted token."""

        if self.predicted_tokens <= 0:
            raise ValueError("bytes_per_token is undefined for zero predicted tokens")
        return self.total_utf8_bytes / self.predicted_tokens

    def __add__(self, other: object) -> BPBTotals:
        if not isinstance(other, BPBTotals):
            return NotImplemented
        return BPBTotals(
            total_nll_nats=self.total_nll_nats + other.total_nll_nats,
            total_utf8_bytes=self.total_utf8_bytes + other.total_utf8_bytes,
            predicted_tokens=self.predicted_tokens + other.predicted_tokens,
        )


def bpb_from_aggregate(
    *,
    total_nll_nats: Real,
    total_utf8_bytes: Integral,
    predicted_tokens: Integral,
) -> BPBTotals:
    """Build validated BPB totals from already-reduced evaluation statistics."""

    return BPBTotals(
        total_nll_nats=float(total_nll_nats),
        total_utf8_bytes=int(total_utf8_bytes),
        predicted_tokens=int(predicted_tokens),
    )


def bpb_from_token_nll(
    nll_nats: Iterable[Real],
    token_byte_lengths: Iterable[Integral],
) -> BPBTotals:
    """Accumulate BPB statistics from per-target NLL and raw byte lengths.

    ``token_byte_lengths`` must come from the tokenizer's raw token-byte mapping,
    not from independently decoding each token to Unicode text. Some tokenizers
    contain pieces that are not valid standalone UTF-8 strings even though the
    combined sequence is valid.

    A byte length of zero marks a special/control token. Its NLL is excluded from
    both the BPB numerator and denominator so tokenizer-specific specials cannot
    alter the metric. The two iterables must have exactly equal length.
    """

    total_nll = 0.0
    total_bytes = 0
    predicted_tokens = 0

    try:
        pairs = zip(nll_nats, token_byte_lengths, strict=True)
        for index, (nll_value, byte_value) in enumerate(pairs):
            nll = _checked_nll(nll_value, field=f"nll_nats[{index}]")
            byte_count = _checked_byte_count(
                byte_value,
                field=f"token_byte_lengths[{index}]",
            )
            if byte_count == 0:
                continue
            total_nll += nll
            total_bytes += byte_count
            predicted_tokens += 1
    except ValueError as exc:
        if "zip() argument" in str(exc):
            raise ValueError("nll_nats and token_byte_lengths must have equal length") from exc
        raise

    return BPBTotals(
        total_nll_nats=total_nll,
        total_utf8_bytes=total_bytes,
        predicted_tokens=predicted_tokens,
    )


def merge_bpb_totals(parts: Iterable[BPBTotals]) -> BPBTotals:
    """Merge shard/rank BPB sufficient statistics without averaging averages."""

    total = BPBTotals(total_nll_nats=0.0, total_utf8_bytes=0, predicted_tokens=0)
    for index, part in enumerate(parts):
        if not isinstance(part, BPBTotals):
            raise TypeError(f"parts[{index}] must be BPBTotals")
        total = total + part
    return total
