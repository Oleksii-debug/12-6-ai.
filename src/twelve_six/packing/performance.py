"""Objective-preserving performance helpers around canonical D04 packing.

This module does not define a new packing identity.  It derives source-offset maps
for D04 document-isolated windows and may remove only an all-padding suffix from
an already-formed batch.  The valid causal-pair trace is therefore unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .core import PackedCausalExample, collate_rows


@dataclass(frozen=True, slots=True)
class PackedSourceSpan:
    """Compact map from one D04 packed block to one source-document token span."""

    record_id: str
    source_start: int
    source_end: int
    sequence_length: int

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("source span must be non-empty and non-negative")
        if self.actual_length > self.sequence_length:
            raise ValueError("source span exceeds packed sequence length")
        if self.actual_length < 2:
            raise ValueError("D04 emits only spans with at least two actual tokens")

    @property
    def actual_length(self) -> int:
        return self.source_end - self.source_start

    @property
    def padding_length(self) -> int:
        return self.sequence_length - self.actual_length

    @property
    def optimized_pairs(self) -> int:
        return self.actual_length - 1

    def source_offset_for_packed_position(self, packed_position: int) -> int | None:
        if not 0 <= packed_position < self.sequence_length:
            raise IndexError("packed_position outside block")
        if packed_position >= self.actual_length:
            return None
        return self.source_start + packed_position


def document_window_spans(
    record_id: str,
    token_count: int,
    *,
    sequence_length: int,
) -> tuple[PackedSourceSpan, ...]:
    """Return the exact compact provenance layout used by isolated D04 windows."""
    if token_count < 2:
        return ()
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")

    spans: list[PackedSourceSpan] = []
    start = 0
    while token_count - start >= sequence_length:
        spans.append(
            PackedSourceSpan(
                record_id=record_id,
                source_start=start,
                source_end=start + sequence_length,
                sequence_length=sequence_length,
            )
        )
        start += sequence_length - 1
    if token_count - start >= 2:
        spans.append(
            PackedSourceSpan(
                record_id=record_id,
                source_start=start,
                source_end=token_count,
                sequence_length=sequence_length,
            )
        )
    return tuple(spans)


def valid_causal_pairs(span: PackedSourceSpan) -> tuple[tuple[int, int], ...]:
    """Return source-token offset pairs optimized by one packed block."""
    return tuple(
        (offset, offset + 1)
        for offset in range(span.source_start, span.source_end - 1)
    )


def right_trim_width(examples: Sequence[PackedCausalExample]) -> int:
    """Find the smallest common time width that removes only right padding."""
    if not examples:
        raise ValueError("examples must not be empty")
    widths = []
    for example in examples:
        actual = sum(example.attention_mask)
        if actual < 2:
            raise ValueError("packed example must contain at least two actual tokens")
        if tuple(example.attention_mask[:actual]) != (1,) * actual:
            raise ValueError("right trimming requires a contiguous unpadded prefix")
        if any(example.attention_mask[actual:]):
            raise ValueError("right trimming cannot remove interior padding")
        widths.append(actual)
    return max(widths)


def collate_right_trimmed_rows(
    examples: Sequence[PackedCausalExample],
    *,
    target_mode: str = "labels",
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Collate in incumbent order and remove only the batch-wide padded suffix."""
    width = right_trim_width(examples)
    rows = collate_rows(examples, target_mode=target_mode)
    return {
        key: tuple(tuple(row[:width]) for row in value)
        for key, value in rows.items()
    }
