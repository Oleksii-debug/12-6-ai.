"""Tokenizer-agnostic bits-per-byte (BPB) evaluation primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import zip_longest
from typing import Iterable

LN_2 = math.log(2.0)
_MISSING = object()


def _validate_nll(nll_nats: float) -> float:
    if isinstance(nll_nats, bool) or not isinstance(nll_nats, (int, float)):
        raise TypeError("nll_nats must be a real number")
    value = float(nll_nats)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("nll_nats must be finite and non-negative")
    return value


def _validate_byte_count(raw_byte_count: int) -> int:
    if isinstance(raw_byte_count, bool) or not isinstance(raw_byte_count, int):
        raise TypeError("raw_byte_count must be an integer")
    if raw_byte_count < 0:
        raise ValueError("raw_byte_count must be non-negative")
    return raw_byte_count


def token_bits_per_byte(nll_nats: float, raw_byte_length: int) -> float | None:
    """Return one token's BPB, or ``None`` for a zero-byte special token."""
    nll = _validate_nll(nll_nats)
    byte_length = _validate_byte_count(raw_byte_length)
    if byte_length == 0:
        return None
    return nll / (LN_2 * byte_length)


def bits_per_byte(total_nll_nats: float, raw_byte_count: int) -> float:
    """Convert aggregate NLL in nats to bits per raw source byte."""
    nll = _validate_nll(total_nll_nats)
    byte_count = _validate_byte_count(raw_byte_count)
    if byte_count == 0:
        raise ValueError("bits-per-byte is undefined with zero scored bytes")
    return nll / (LN_2 * byte_count)


@dataclass(frozen=True)
class BpbTotals:
    """Mergeable sufficient statistics for exact aggregate BPB accounting."""

    nll_nats: float = 0.0
    byte_count: int = 0
    scored_tokens: int = 0
    excluded_zero_byte_tokens: int = 0

    def __post_init__(self) -> None:
        _validate_nll(self.nll_nats)
        _validate_byte_count(self.byte_count)
        _validate_byte_count(self.scored_tokens)
        _validate_byte_count(self.excluded_zero_byte_tokens)
        if self.byte_count == 0 and self.nll_nats != 0.0:
            raise ValueError("nonzero NLL cannot be bound to zero scored bytes")
        if self.byte_count > 0 and self.scored_tokens == 0:
            raise ValueError("positive scored bytes require at least one scored token")

    @property
    def total_bits(self) -> float:
        return self.nll_nats / LN_2

    @property
    def bpb(self) -> float:
        return bits_per_byte(self.nll_nats, self.byte_count)

    def add_token(self, nll_nats: float, raw_byte_length: int) -> BpbTotals:
        """Return totals with one token added under zero-byte exclusion semantics."""
        nll = _validate_nll(nll_nats)
        byte_length = _validate_byte_count(raw_byte_length)
        if byte_length == 0:
            return BpbTotals(
                nll_nats=self.nll_nats,
                byte_count=self.byte_count,
                scored_tokens=self.scored_tokens,
                excluded_zero_byte_tokens=self.excluded_zero_byte_tokens + 1,
            )
        return BpbTotals(
            nll_nats=self.nll_nats + nll,
            byte_count=self.byte_count + byte_length,
            scored_tokens=self.scored_tokens + 1,
            excluded_zero_byte_tokens=self.excluded_zero_byte_tokens,
        )

    def merge(self, other: BpbTotals) -> BpbTotals:
        """Merge shard/process totals without averaging per-shard BPB values."""
        if not isinstance(other, BpbTotals):
            raise TypeError("other must be BpbTotals")
        return BpbTotals(
            nll_nats=self.nll_nats + other.nll_nats,
            byte_count=self.byte_count + other.byte_count,
            scored_tokens=self.scored_tokens + other.scored_tokens,
            excluded_zero_byte_tokens=(
                self.excluded_zero_byte_tokens + other.excluded_zero_byte_tokens
            ),
        )

    def as_dict(self) -> dict[str, float | int | None]:
        """Return JSON-friendly sufficient statistics and derived BPB."""
        return {
            "nll_nats": self.nll_nats,
            "byte_count": self.byte_count,
            "scored_tokens": self.scored_tokens,
            "excluded_zero_byte_tokens": self.excluded_zero_byte_tokens,
            "bits_per_byte": self.bpb if self.byte_count else None,
        }


def accumulate_bpb(
    nll_nats: Iterable[float], raw_byte_lengths: Iterable[int]
) -> BpbTotals:
    """Accumulate token losses and raw token-byte lengths with strict pairing."""
    totals = BpbTotals()
    for index, (nll, byte_length) in enumerate(
        zip_longest(nll_nats, raw_byte_lengths, fillvalue=_MISSING)
    ):
        if nll is _MISSING or byte_length is _MISSING:
            raise ValueError(f"loss/byte-length sequences differ at token index {index}")
        totals = totals.add_token(nll, byte_length)
    return totals
