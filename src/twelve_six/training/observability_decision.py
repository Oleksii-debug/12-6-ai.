"""Strict paid-compute decision gates for measured training observability.

This module deliberately refuses to turn host-enqueue CUDA timings or rank-local
throughput into paid-run capacity evidence. It consumes observability summaries;
it never authorizes spend by itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _positive_finite(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def target_hardware_paid_compute_decision_support(
    local_summary: Mapping[str, Any],
    *,
    distributed_aggregate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fail-closed €2k/€10k evidence gates for a measured target run.

    A CUDA result is cost-usable only when step timing is synchronized. Multi-rank
    cost projection additionally requires a full-rank aggregate whose throughput
    uses the slowest-rank critical path.
    """
    rank = local_summary.get("rank")
    device = local_summary.get("device")
    throughput = local_summary.get("throughput")
    identity_sha256 = local_summary.get("run_identity_sha256")
    if not isinstance(rank, Mapping) or not isinstance(device, Mapping):
        raise ValueError("summary missing rank/device metadata")
    if not isinstance(throughput, Mapping):
        raise ValueError("summary missing throughput metadata")
    if not isinstance(identity_sha256, str) or not identity_sha256:
        raise ValueError("summary missing run identity SHA-256")

    device_type = str(device.get("type"))
    timing_mode = str(device.get("step_timing_mode"))
    world_size = int(rank.get("world_size", 1))
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    common: dict[str, Any] = {
        "run_identity_sha256": identity_sha256,
        "world_size": world_size,
        "device_type": device_type,
        "step_timing_mode": timing_mode,
        "telemetry_alone_authorizes_spend": False,
        "cost_projection_formula": (
            "projected_cost_eur = target_training_tokens / measured_global_tokens_per_second "
            "/ 3600 * measured_or_quoted_eur_per_gpu_hour * gpu_count"
        ),
    }

    if device_type != "cuda":
        return {
            **common,
            "throughput_source": "NOT_COST_USABLE_CPU_ONLY",
            "measured_global_tokens_per_second": None,
            "euro_2000_gate": "BLOCKED_PENDING_TARGET_GPU_CALIBRATION",
            "euro_10000_gate": "BLOCKED_PENDING_TARGET_GPU_AND_DISTRIBUTED_CALIBRATION",
            "reason": "CPU throughput cannot price a paid GPU run",
        }

    if timing_mode != "CUDA_SYNCHRONIZED_WALL":
        return {
            **common,
            "throughput_source": "NOT_COST_USABLE_CUDA_HOST_ENQUEUE_TIMING",
            "measured_global_tokens_per_second": None,
            "euro_2000_gate": "BLOCKED_PENDING_SYNCHRONIZED_TARGET_GPU_CALIBRATION",
            "euro_10000_gate": "BLOCKED_PENDING_SYNCHRONIZED_GPU_AND_DISTRIBUTED_CALIBRATION",
            "reason": (
                "CUDA host-enqueue wall time does not measure completed device work; "
                "rerun with synchronized step timing"
            ),
        }

    local_tps = _positive_finite(
        throughput.get("train_tokens_per_second"),
        "throughput.train_tokens_per_second",
    )
    if world_size == 1:
        return {
            **common,
            "throughput_source": "SYNCHRONIZED_SINGLE_GPU_LOCAL_SUMMARY",
            "measured_global_tokens_per_second": local_tps,
            "euro_2000_gate": "REQUIRES_TOKEN_BUDGET_AND_PROVIDER_PRICE",
            "euro_10000_gate": "BLOCKED_PENDING_MULTI_GPU_SCALING_EVIDENCE",
            "reason": (
                "synchronized single-GPU throughput can price only a bounded single-GPU run; "
                "€10k scale requires multi-rank evidence"
            ),
        }

    if distributed_aggregate is None:
        return {
            **common,
            "throughput_source": "NOT_COST_USABLE_RANK_LOCAL_ONLY",
            "measured_global_tokens_per_second": None,
            "euro_2000_gate": "BLOCKED_PENDING_FULL_RANK_THROUGHPUT_AGGREGATE",
            "euro_10000_gate": "BLOCKED_PENDING_FULL_RANK_THROUGHPUT_AND_STABILITY_EVIDENCE",
            "reason": (
                "rank-local throughput cannot price synchronous distributed training; "
                "aggregate all ranks over the critical-path wall time"
            ),
        }

    aggregate_identity = distributed_aggregate.get("run_identity_sha256")
    aggregate_world_size = distributed_aggregate.get("world_size")
    if aggregate_identity != identity_sha256:
        raise ValueError("distributed aggregate run identity does not match local summary")
    if aggregate_world_size != world_size:
        raise ValueError("distributed aggregate world size does not match local summary")
    global_tps = _positive_finite(
        distributed_aggregate.get("global_train_tokens_per_second"),
        "distributed_aggregate.global_train_tokens_per_second",
    )
    critical_path = _positive_finite(
        distributed_aggregate.get("critical_path_training_seconds"),
        "distributed_aggregate.critical_path_training_seconds",
    )

    return {
        **common,
        "throughput_source": "FULL_RANK_CRITICAL_PATH_AGGREGATE",
        "measured_global_tokens_per_second": global_tps,
        "critical_path_training_seconds": critical_path,
        "rank_time_skew_ratio": distributed_aggregate.get("rank_time_skew_ratio"),
        "euro_2000_gate": "REQUIRES_TOKEN_BUDGET_AND_PROVIDER_PRICE",
        "euro_10000_gate": "REQUIRES_TOKEN_BUDGET_PROVIDER_PRICE_AND_STABILITY_GATE",
        "reason": (
            "synchronized full-rank throughput is cost-usable; budget, provider price, "
            "memory headroom, checkpoint overhead, and stability remain external gates"
        ),
    }
