"""Tokenizer-aware raw-source context-span calibration.

This module measures how many raw source bytes are reachable through a fixed
token-context budget without crossing document boundaries. The result is a
mechanics calibration primitive, not evidence that any context length is
scientifically optimal for training.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral


def _checked_non_negative_int(value: Integral, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _checked_context_tokens(value: Integral) -> int:
    result = _checked_non_negative_int(value, field="context_tokens")
    if result == 0:
        raise ValueError("context_tokens must be positive")
    return result


def _materialize_token_byte_lengths(values: Iterable[Integral]) -> tuple[int, ...]:
    result: list[int] = []
    for index, value in enumerate(values):
        result.append(
            _checked_non_negative_int(value, field=f"token_byte_lengths[{index}]")
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ContextSpanTotals:
    """Additive source-span statistics for one token-context budget."""

    context_tokens: int
    target_positions: int
    total_context_bytes: int
    total_context_token_slots: int
    saturated_target_positions: int
    min_context_bytes: int | None
    max_context_bytes: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_tokens", _checked_context_tokens(self.context_tokens))
        for field_name in (
            "target_positions",
            "total_context_bytes",
            "total_context_token_slots",
            "saturated_target_positions",
        ):
            object.__setattr__(
                self,
                field_name,
                _checked_non_negative_int(getattr(self, field_name), field=field_name),
            )

        if self.saturated_target_positions > self.target_positions:
            raise ValueError("saturated_target_positions cannot exceed target_positions")
        if self.target_positions == 0:
            if self.total_context_bytes != 0 or self.total_context_token_slots != 0:
                raise ValueError("empty calibration cannot contain context totals")
            if self.min_context_bytes is not None or self.max_context_bytes is not None:
                raise ValueError("empty calibration cannot contain min/max context bytes")
            return

        if self.total_context_token_slots < self.target_positions:
            raise ValueError("every included target must have at least one context token slot")
        if self.total_context_token_slots > self.context_tokens * self.target_positions:
            raise ValueError("context token slots exceed the declared context budget")

        if self.min_context_bytes is None or self.max_context_bytes is None:
            raise ValueError("non-empty calibration requires min/max context bytes")
        min_bytes = _checked_non_negative_int(
            self.min_context_bytes,
            field="min_context_bytes",
        )
        max_bytes = _checked_non_negative_int(
            self.max_context_bytes,
            field="max_context_bytes",
        )
        if min_bytes > max_bytes:
            raise ValueError("min_context_bytes cannot exceed max_context_bytes")
        if not min_bytes * self.target_positions <= self.total_context_bytes:
            raise ValueError("total_context_bytes is below the declared minimum")
        if not self.total_context_bytes <= max_bytes * self.target_positions:
            raise ValueError("total_context_bytes is above the declared maximum")
        object.__setattr__(self, "min_context_bytes", min_bytes)
        object.__setattr__(self, "max_context_bytes", max_bytes)

    @property
    def mean_context_bytes(self) -> float:
        """Mean raw-source bytes reachable before an included target."""

        if self.target_positions == 0:
            raise ValueError("mean_context_bytes is undefined for zero target positions")
        return self.total_context_bytes / self.target_positions

    @property
    def mean_context_token_slots(self) -> float:
        """Mean token slots actually available before an included target."""

        if self.target_positions == 0:
            raise ValueError("mean_context_token_slots is undefined for zero target positions")
        return self.total_context_token_slots / self.target_positions

    @property
    def bytes_per_context_token_slot(self) -> float:
        """Raw-source bytes represented per occupied context token slot."""

        if self.total_context_token_slots == 0:
            raise ValueError("bytes_per_context_token_slot is undefined for zero token slots")
        return self.total_context_bytes / self.total_context_token_slots

    @property
    def saturated_target_fraction(self) -> float:
        """Fraction of targets that have the full token-context budget available."""

        if self.target_positions == 0:
            raise ValueError("saturated_target_fraction is undefined for zero target positions")
        return self.saturated_target_positions / self.target_positions

    def __add__(self, other: object) -> ContextSpanTotals:
        if not isinstance(other, ContextSpanTotals):
            return NotImplemented
        if self.context_tokens != other.context_tokens:
            raise ValueError("cannot merge different context token budgets")

        if self.target_positions == 0:
            min_bytes = other.min_context_bytes
            max_bytes = other.max_context_bytes
        elif other.target_positions == 0:
            min_bytes = self.min_context_bytes
            max_bytes = self.max_context_bytes
        else:
            if self.min_context_bytes is None or other.min_context_bytes is None:
                raise ValueError("non-empty calibration is missing min_context_bytes")
            if self.max_context_bytes is None or other.max_context_bytes is None:
                raise ValueError("non-empty calibration is missing max_context_bytes")
            min_bytes = min(self.min_context_bytes, other.min_context_bytes)
            max_bytes = max(self.max_context_bytes, other.max_context_bytes)

        return ContextSpanTotals(
            context_tokens=self.context_tokens,
            target_positions=self.target_positions + other.target_positions,
            total_context_bytes=self.total_context_bytes + other.total_context_bytes,
            total_context_token_slots=(
                self.total_context_token_slots + other.total_context_token_slots
            ),
            saturated_target_positions=(
                self.saturated_target_positions + other.saturated_target_positions
            ),
            min_context_bytes=min_bytes,
            max_context_bytes=max_bytes,
        )


def calibrate_document_context_span(
    token_byte_lengths: Iterable[Integral],
    *,
    context_tokens: Integral,
) -> ContextSpanTotals:
    """Measure raw-byte context spans for one document.

    Positive-byte tokens are source-bearing targets. Zero-byte special/control
    tokens are not scored as targets, but they still occupy context token slots.
    This prevents tokenizer-specific control tokens from falsely increasing the
    measured raw-source span.

    The first source-bearing token is excluded until at least one preceding token
    slot exists. No context is borrowed from another document.
    """

    budget = _checked_context_tokens(context_tokens)
    lengths = _materialize_token_byte_lengths(token_byte_lengths)
    if len(lengths) < 2:
        return ContextSpanTotals(budget, 0, 0, 0, 0, None, None)

    prefix_bytes = [0]
    for byte_count in lengths:
        prefix_bytes.append(prefix_bytes[-1] + byte_count)

    target_positions = 0
    total_context_bytes = 0
    total_context_token_slots = 0
    saturated_target_positions = 0
    min_context_bytes: int | None = None
    max_context_bytes: int | None = None

    for target_index in range(1, len(lengths)):
        if lengths[target_index] == 0:
            continue

        start = max(0, target_index - budget)
        context_bytes = prefix_bytes[target_index] - prefix_bytes[start]
        context_slots = target_index - start

        target_positions += 1
        total_context_bytes += context_bytes
        total_context_token_slots += context_slots
        if context_slots == budget:
            saturated_target_positions += 1
        min_context_bytes = (
            context_bytes
            if min_context_bytes is None
            else min(min_context_bytes, context_bytes)
        )
        max_context_bytes = (
            context_bytes
            if max_context_bytes is None
            else max(max_context_bytes, context_bytes)
        )

    return ContextSpanTotals(
        context_tokens=budget,
        target_positions=target_positions,
        total_context_bytes=total_context_bytes,
        total_context_token_slots=total_context_token_slots,
        saturated_target_positions=saturated_target_positions,
        min_context_bytes=min_context_bytes,
        max_context_bytes=max_context_bytes,
    )


def calibrate_context_grid(
    documents: Iterable[Iterable[Integral]],
    *,
    context_lengths: Iterable[Integral],
) -> dict[int, ContextSpanTotals]:
    """Calibrate several token-context budgets over document-isolated inputs."""

    budgets: list[int] = []
    seen: set[int] = set()
    for value in context_lengths:
        budget = _checked_context_tokens(value)
        if budget in seen:
            raise ValueError(f"duplicate context length: {budget}")
        seen.add(budget)
        budgets.append(budget)

    if not budgets:
        raise ValueError("context_lengths must not be empty")

    materialized_documents = [
        _materialize_token_byte_lengths(document) for document in documents
    ]
    totals = {
        budget: ContextSpanTotals(budget, 0, 0, 0, 0, None, None)
        for budget in budgets
    }

    for document in materialized_documents:
        for budget in budgets:
            totals[budget] = totals[budget] + calibrate_document_context_span(
                document,
                context_tokens=budget,
            )
    return totals
