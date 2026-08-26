from __future__ import annotations

import math

import pytest

from twelve_six.bpb import BpbTotals, accumulate_bpb, bits_per_byte, token_bits_per_byte


def test_nats_to_bits_conversion() -> None:
    assert bits_per_byte(math.log(2.0), 1) == pytest.approx(1.0)
    assert bits_per_byte(8.0 * math.log(2.0), 4) == pytest.approx(2.0)


def test_per_token_bpb_and_zero_byte_special_exclusion() -> None:
    assert token_bits_per_byte(2.0 * math.log(2.0), 2) == pytest.approx(1.0)
    assert token_bits_per_byte(100.0, 0) is None


def test_zero_byte_special_token_does_not_pollute_aggregate_nll() -> None:
    totals = accumulate_bpb([100.0, math.log(2.0)], [0, 1])
    assert totals.nll_nats == pytest.approx(math.log(2.0))
    assert totals.byte_count == 1
    assert totals.scored_tokens == 1
    assert totals.excluded_zero_byte_tokens == 1
    assert totals.bpb == pytest.approx(1.0)


def test_equivalent_sequence_probability_is_segmentation_invariant() -> None:
    one_token = accumulate_bpb([4.0 * math.log(2.0)], [4])
    two_tokens = accumulate_bpb(
        [1.5 * math.log(2.0), 2.5 * math.log(2.0)],
        [2, 2],
    )
    assert one_token.nll_nats == pytest.approx(two_tokens.nll_nats)
    assert one_token.byte_count == two_tokens.byte_count == 4
    assert one_token.bpb == pytest.approx(two_tokens.bpb)


def test_merge_matches_single_pass_accumulation() -> None:
    left = accumulate_bpb([math.log(2.0), 9.0], [1, 0])
    right = accumulate_bpb([2.0 * math.log(2.0), 3.0 * math.log(2.0)], [1, 2])
    merged = left.merge(right)
    direct = accumulate_bpb(
        [math.log(2.0), 9.0, 2.0 * math.log(2.0), 3.0 * math.log(2.0)],
        [1, 0, 1, 2],
    )
    assert merged.nll_nats == pytest.approx(direct.nll_nats)
    assert merged.byte_count == direct.byte_count
    assert merged.scored_tokens == direct.scored_tokens
    assert merged.excluded_zero_byte_tokens == direct.excluded_zero_byte_tokens
    assert merged.bpb == pytest.approx(direct.bpb)


def test_merge_uses_totals_not_unweighted_shard_average() -> None:
    small = accumulate_bpb([math.log(2.0)], [1])
    large = accumulate_bpb([18.0 * math.log(2.0)], [9])
    merged = small.merge(large)
    assert small.bpb == pytest.approx(1.0)
    assert large.bpb == pytest.approx(2.0)
    assert merged.bpb == pytest.approx(1.9)


@pytest.mark.parametrize("bad_nll", [-1.0, math.inf, -math.inf, math.nan, True, "1"])
def test_invalid_nll_fails_closed(bad_nll: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        token_bits_per_byte(bad_nll, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_bytes", [-1, 1.5, True, "1"])
def test_invalid_byte_length_fails_closed(bad_bytes: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        token_bits_per_byte(1.0, bad_bytes)  # type: ignore[arg-type]


def test_zero_scored_bytes_have_no_defined_aggregate_bpb() -> None:
    totals = accumulate_bpb([1.0, 2.0], [0, 0])
    assert totals.as_dict()["bits_per_byte"] is None
    with pytest.raises(ValueError, match="zero scored bytes"):
        _ = totals.bpb


def test_mismatched_token_sequences_fail_closed() -> None:
    with pytest.raises(ValueError, match="differ at token index 1"):
        accumulate_bpb([1.0, 2.0], [1])


def test_invalid_preaggregated_totals_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonzero NLL"):
        BpbTotals(nll_nats=1.0, byte_count=0, scored_tokens=0)
    with pytest.raises(ValueError, match="scored token"):
        BpbTotals(nll_nats=0.0, byte_count=1, scored_tokens=0)
