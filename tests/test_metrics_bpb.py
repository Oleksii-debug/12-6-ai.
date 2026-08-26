from __future__ import annotations

import math

import pytest

from twelve_six.metrics import BPBTotals, bpb_from_aggregate, bpb_from_token_nll, merge_bpb_totals


def test_known_bpb_conversion_from_nats() -> None:
    totals = bpb_from_aggregate(
        total_nll_nats=3.0 * math.log(2.0),
        total_utf8_bytes=3,
        predicted_tokens=2,
    )
    assert totals.bits_per_byte == pytest.approx(1.0)
    assert totals.mean_nll_nats_per_token == pytest.approx(1.5 * math.log(2.0))
    assert totals.bytes_per_token == pytest.approx(1.5)


def test_per_token_accumulator_excludes_zero_byte_special_tokens() -> None:
    totals = bpb_from_token_nll(
        [100.0, math.log(2.0), 2.0 * math.log(2.0), 100.0],
        [0, 1, 2, 0],
    )
    assert totals.total_nll_nats == pytest.approx(3.0 * math.log(2.0))
    assert totals.total_utf8_bytes == 3
    assert totals.predicted_tokens == 2
    assert totals.bits_per_byte == pytest.approx(1.0)


def test_equal_sequence_probability_is_invariant_to_token_segmentation() -> None:
    byte_tokenization = bpb_from_token_nll(
        [0.5 * math.log(2.0)] * 4,
        [1, 1, 1, 1],
    )
    merged_tokenization = bpb_from_token_nll(
        [math.log(2.0), math.log(2.0)],
        [2, 2],
    )
    assert byte_tokenization.bits_per_byte == pytest.approx(0.5)
    assert merged_tokenization.bits_per_byte == pytest.approx(0.5)
    assert byte_tokenization.bits_per_byte == pytest.approx(merged_tokenization.bits_per_byte)


def test_shard_merge_matches_monolithic_accumulation() -> None:
    left = bpb_from_token_nll([math.log(2.0), 2.0 * math.log(2.0)], [1, 2])
    right = bpb_from_token_nll([0.5 * math.log(2.0), math.log(2.0)], [1, 3])
    merged = merge_bpb_totals([left, right])
    monolithic = bpb_from_token_nll(
        [math.log(2.0), 2.0 * math.log(2.0), 0.5 * math.log(2.0), math.log(2.0)],
        [1, 2, 1, 3],
    )
    assert merged == monolithic
    assert merged.bits_per_byte == pytest.approx(monolithic.bits_per_byte)


def test_mismatched_per_token_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="equal length"):
        bpb_from_token_nll([1.0, 2.0], [1])


@pytest.mark.parametrize("bad_nll", [-1.0, float("nan"), float("inf")])
def test_invalid_nll_fails_closed(bad_nll: float) -> None:
    with pytest.raises(ValueError):
        bpb_from_token_nll([bad_nll], [1])


@pytest.mark.parametrize("bad_bytes", [-1, 1.5, True])
def test_invalid_token_byte_length_fails_closed(bad_bytes: object) -> None:
    expected = TypeError if bad_bytes in {1.5, True} else ValueError
    with pytest.raises(expected):
        bpb_from_token_nll([1.0], [bad_bytes])


def test_bool_aggregate_inputs_are_rejected_not_coerced() -> None:
    with pytest.raises(TypeError):
        bpb_from_aggregate(total_nll_nats=True, total_utf8_bytes=1, predicted_tokens=1)
    with pytest.raises(TypeError):
        bpb_from_aggregate(total_nll_nats=1.0, total_utf8_bytes=True, predicted_tokens=1)
    with pytest.raises(TypeError):
        bpb_from_aggregate(total_nll_nats=1.0, total_utf8_bytes=1, predicted_tokens=True)


def test_zero_byte_totals_are_mergeable_but_not_scoreable() -> None:
    empty = merge_bpb_totals([])
    assert empty == BPBTotals(0.0, 0, 0)
    with pytest.raises(ValueError, match="undefined"):
        _ = empty.bits_per_byte


def test_inconsistent_aggregate_coverage_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-zero NLL"):
        BPBTotals(total_nll_nats=1.0, total_utf8_bytes=0, predicted_tokens=0)
    with pytest.raises(ValueError, match="positive scored bytes"):
        BPBTotals(total_nll_nats=0.0, total_utf8_bytes=1, predicted_tokens=0)
