from __future__ import annotations

import pytest

from twelve_six.training.observability_decision import (
    target_hardware_paid_compute_decision_support,
)


def _summary(
    *,
    device_type: str,
    timing_mode: str,
    world_size: int = 1,
    tokens_per_second: float = 1000.0,
) -> dict[str, object]:
    return {
        "run_identity_sha256": "a" * 64,
        "rank": {"rank": 0, "local_rank": 0, "world_size": world_size},
        "device": {"type": device_type, "step_timing_mode": timing_mode},
        "throughput": {"train_tokens_per_second": tokens_per_second},
    }


def test_cpu_never_opens_paid_gpu_capacity_gate() -> None:
    decision = target_hardware_paid_compute_decision_support(
        _summary(device_type="cpu", timing_mode="CPU_WALL")
    )

    assert decision["euro_2000_gate"] == "BLOCKED_PENDING_TARGET_GPU_CALIBRATION"
    assert (
        decision["euro_10000_gate"]
        == "BLOCKED_PENDING_TARGET_GPU_AND_DISTRIBUTED_CALIBRATION"
    )
    assert decision["measured_global_tokens_per_second"] is None


def test_cuda_host_enqueue_timing_is_not_cost_usable() -> None:
    decision = target_hardware_paid_compute_decision_support(
        _summary(device_type="cuda", timing_mode="CUDA_HOST_ENQUEUE_WALL")
    )

    assert (
        decision["euro_2000_gate"]
        == "BLOCKED_PENDING_SYNCHRONIZED_TARGET_GPU_CALIBRATION"
    )
    assert decision["throughput_source"] == "NOT_COST_USABLE_CUDA_HOST_ENQUEUE_TIMING"
    assert decision["measured_global_tokens_per_second"] is None


def test_synchronized_single_gpu_is_cost_usable_but_not_ten_k_scaling_evidence() -> None:
    decision = target_hardware_paid_compute_decision_support(
        _summary(
            device_type="cuda",
            timing_mode="CUDA_SYNCHRONIZED_WALL",
            tokens_per_second=1234.5,
        )
    )

    assert decision["measured_global_tokens_per_second"] == pytest.approx(1234.5)
    assert decision["euro_2000_gate"] == "REQUIRES_TOKEN_BUDGET_AND_PROVIDER_PRICE"
    assert decision["euro_10000_gate"] == "BLOCKED_PENDING_MULTI_GPU_SCALING_EVIDENCE"


def test_multi_rank_local_summary_is_not_global_throughput() -> None:
    decision = target_hardware_paid_compute_decision_support(
        _summary(
            device_type="cuda",
            timing_mode="CUDA_SYNCHRONIZED_WALL",
            world_size=4,
        )
    )

    assert decision["euro_2000_gate"] == "BLOCKED_PENDING_FULL_RANK_THROUGHPUT_AGGREGATE"
    assert decision["throughput_source"] == "NOT_COST_USABLE_RANK_LOCAL_ONLY"


def test_full_rank_critical_path_aggregate_can_feed_cost_formula() -> None:
    local = _summary(
        device_type="cuda",
        timing_mode="CUDA_SYNCHRONIZED_WALL",
        world_size=2,
        tokens_per_second=900.0,
    )
    aggregate = {
        "run_identity_sha256": "a" * 64,
        "world_size": 2,
        "global_train_tokens_per_second": 1700.0,
        "critical_path_training_seconds": 100.0,
        "rank_time_skew_ratio": 1.08,
    }

    decision = target_hardware_paid_compute_decision_support(
        local,
        distributed_aggregate=aggregate,
    )

    assert decision["throughput_source"] == "FULL_RANK_CRITICAL_PATH_AGGREGATE"
    assert decision["measured_global_tokens_per_second"] == pytest.approx(1700.0)
    assert decision["euro_2000_gate"] == "REQUIRES_TOKEN_BUDGET_AND_PROVIDER_PRICE"
    assert (
        decision["euro_10000_gate"]
        == "REQUIRES_TOKEN_BUDGET_PROVIDER_PRICE_AND_STABILITY_GATE"
    )
    assert decision["telemetry_alone_authorizes_spend"] is False


def test_distributed_aggregate_must_match_exact_run_identity() -> None:
    local = _summary(
        device_type="cuda",
        timing_mode="CUDA_SYNCHRONIZED_WALL",
        world_size=2,
    )
    aggregate = {
        "run_identity_sha256": "b" * 64,
        "world_size": 2,
        "global_train_tokens_per_second": 1700.0,
        "critical_path_training_seconds": 100.0,
    }

    with pytest.raises(ValueError, match="run identity"):
        target_hardware_paid_compute_decision_support(
            local,
            distributed_aggregate=aggregate,
        )
