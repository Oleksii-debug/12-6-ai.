from __future__ import annotations

import math

from twelve_six.training.precision_learning import (
    TOLERANCES,
    compare_precision_profiles,
    loss_to_bpb,
)


def _profile(*, validation_offset: float = 0.0, dynamics_scale: float = 1.0) -> dict:
    return {
        "optimized_tokens": 100_128,
        "optimizer_steps": 4,
        "curve": [
            {
                "requested_tokens": 0,
                "actual_tokens": 0,
                "train": {"bpb": 8.0},
                "validation": {"bpb": 8.1 + validation_offset},
            },
            {
                "requested_tokens": 100_000,
                "actual_tokens": 100_128,
                "train": {"bpb": 4.0 + validation_offset / 2.0},
                "validation": {"bpb": 4.2 + validation_offset},
            },
        ],
        "steps": [
            {
                "tokens_seen": tokens,
                "gradient_norm": gradient * dynamics_scale,
                "update_norm": update * dynamics_scale,
            }
            for tokens, gradient, update in (
                (25_032, 2.0, 0.50),
                (50_064, 1.8, 0.45),
                (75_096, 1.6, 0.40),
                (100_128, 1.4, 0.35),
            )
        ],
        "finite_state": {"all_finite": True},
        "checkpoint": {"post_reload_logits_max_abs": 0.0},
    }


def test_loss_to_bpb_uses_base_two_units() -> None:
    assert math.isclose(loss_to_bpb(math.log(2.0)), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_compare_precision_profiles_accepts_bounded_non_bitwise_differences() -> None:
    fp32 = _profile()
    bf16 = _profile(validation_offset=0.02, dynamics_scale=1.03)

    comparison = compare_precision_profiles(fp32, bf16)

    assert comparison["within_tolerance"] is True
    assert comparison["cross_precision_bitwise_equality_required"] is False
    assert comparison["metrics"]["final_validation_bpb_abs"] == 0.02


def test_compare_precision_profiles_surfaces_learning_divergence() -> None:
    fp32 = _profile()
    bf16 = _profile(
        validation_offset=TOLERANCES["final_validation_bpb_abs"] + 0.02,
        dynamics_scale=1.02,
    )

    comparison = compare_precision_profiles(fp32, bf16)

    assert comparison["within_tolerance"] is False
    assert comparison["checks"]["final_validation_bpb_abs"] is False
