from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.distributed.ddp_training import (
    build_evidence,
    ddp_token_weighted_loss_scale,
    run_canonical_cpu_ddp_probe,
)


def test_ddp_token_weighted_scale_reconstructs_global_mean() -> None:
    # DDP averages rank gradients. For 15 vs 11 valid tokens, local division is wrong;
    # these unequal scales make the DDP average equal the 26-token global mean.
    rank0 = ddp_token_weighted_loss_scale(15, 26, 2)
    rank1 = ddp_token_weighted_loss_scale(11, 26, 2)
    assert rank0 == pytest.approx(30 / 26)
    assert rank1 == pytest.approx(22 / 26)
    assert (rank0 + rank1) / 2 == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("local_tokens", "global_tokens", "world_size", "error"),
    [
        (0, 2, 2, ValueError),
        (3, 2, 2, ValueError),
        (1, 2, 0, ValueError),
        (True, 2, 2, TypeError),
    ],
)
def test_ddp_token_weighted_scale_fails_closed(
    local_tokens: int,
    global_tokens: int,
    world_size: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ddp_token_weighted_loss_scale(local_tokens, global_tokens, world_size)


def test_real_canonical_s0_two_rank_ddp_matches_global_reference() -> None:
    result = run_canonical_cpu_ddp_probe(
        Path("configs/stages/s0_10k.json"),
        world_size=2,
        sequence_length=12,
        seed=20260825,
        timeout_seconds=60.0,
    )
    assert result.passed
    assert result.parameter_count == 10_140
    assert result.local_valid_tokens == (11, 9)
    assert result.global_valid_tokens == 20
    assert result.max_cross_rank_parameter_diff == 0.0
    assert result.max_reference_parameter_diff <= result.reference_tolerance
    assert result.max_parameter_change > 0.0


def test_evidence_is_source_bound_and_self_hashed() -> None:
    result = run_canonical_cpu_ddp_probe(
        Path("configs/stages/s0_10k.json"),
        world_size=2,
        sequence_length=8,
        seed=7,
        timeout_seconds=60.0,
    )
    evidence = build_evidence(result, source_sha="a" * 40)
    expected = evidence.pop("report_sha256")
    encoded = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert evidence["canonical_base"] == "random_init"
    assert evidence["paid_compute"] is False
    assert evidence["promotion_authority"] is False
    assert evidence["execution"]["passed"] is True

    with pytest.raises(ValueError, match="40-hex"):
        build_evidence(result, source_sha="deadbeef")
