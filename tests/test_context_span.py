from __future__ import annotations

import pytest

from twelve_six.context_span import (
    ContextSpanTotals,
    calibrate_context_grid,
    calibrate_document_context_span,
)


def test_fixed_byte_tokens_have_expected_causal_source_span() -> None:
    totals = calibrate_document_context_span([1, 1, 1, 1], context_tokens=2)

    assert totals.target_positions == 3
    assert totals.total_context_bytes == 5
    assert totals.total_context_token_slots == 5
    assert totals.saturated_target_positions == 2
    assert totals.min_context_bytes == 1
    assert totals.max_context_bytes == 2
    assert totals.mean_context_bytes == pytest.approx(5 / 3)
    assert totals.bytes_per_context_token_slot == pytest.approx(1.0)
    assert totals.saturated_target_fraction == pytest.approx(2 / 3)


def test_variable_token_bytes_change_reachable_source_span() -> None:
    totals = calibrate_document_context_span([2, 4, 1, 3], context_tokens=2)

    assert totals.target_positions == 3
    assert totals.total_context_bytes == 13
    assert totals.mean_context_bytes == pytest.approx(13 / 3)
    assert totals.min_context_bytes == 2
    assert totals.max_context_bytes == 6


def test_document_boundaries_are_never_crossed() -> None:
    grid = calibrate_context_grid(
        [[5, 5], [5, 5]],
        context_lengths=[4],
    )
    totals = grid[4]

    assert totals.target_positions == 2
    assert totals.total_context_bytes == 10
    assert totals.total_context_token_slots == 2
    assert totals.saturated_target_positions == 0
    assert totals.max_context_bytes == 5


def test_zero_byte_special_occupies_slot_but_is_not_target() -> None:
    totals = calibrate_document_context_span([1, 0, 3], context_tokens=2)

    assert totals.target_positions == 1
    assert totals.total_context_bytes == 1
    assert totals.total_context_token_slots == 2
    assert totals.saturated_target_positions == 1


def test_grid_supports_multiple_context_budgets_without_average_of_averages() -> None:
    grid = calibrate_context_grid(
        [[1, 1, 1, 1], [2, 2, 2]],
        context_lengths=[1, 2],
    )

    assert list(grid) == [1, 2]
    assert grid[1].target_positions == 5
    assert grid[1].total_context_bytes == 7
    assert grid[2].target_positions == 5
    assert grid[2].total_context_bytes == 11


def test_empty_calibration_is_mergeable_but_not_scoreable() -> None:
    empty = calibrate_context_grid([], context_lengths=[8])[8]
    assert empty == ContextSpanTotals(8, 0, 0, 0, 0, None, None)

    with pytest.raises(ValueError, match="undefined"):
        _ = empty.mean_context_bytes


@pytest.mark.parametrize("bad_context", [0, -1, 1.5, True])
def test_invalid_context_budget_fails_closed(bad_context: object) -> None:
    expected = TypeError if bad_context in {1.5, True} else ValueError
    with pytest.raises(expected):
        calibrate_document_context_span([1, 1], context_tokens=bad_context)


@pytest.mark.parametrize("bad_length", [-1, 1.5, True])
def test_invalid_token_byte_length_fails_closed(bad_length: object) -> None:
    expected = TypeError if bad_length in {1.5, True} else ValueError
    with pytest.raises(expected):
        calibrate_document_context_span([1, bad_length], context_tokens=1)


def test_duplicate_or_empty_context_grid_fails_closed() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_context_grid([[1, 1]], context_lengths=[])
    with pytest.raises(ValueError, match="duplicate"):
        calibrate_context_grid([[1, 1]], context_lengths=[2, 2])


def test_context_totals_reject_inconsistent_aggregates() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ContextSpanTotals(2, 1, 1, 1, 2, 1, 1)
    with pytest.raises(ValueError, match="empty calibration"):
        ContextSpanTotals(2, 0, 1, 0, 0, None, None)
