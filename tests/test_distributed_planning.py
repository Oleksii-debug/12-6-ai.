from __future__ import annotations

import pytest

from twelve_six.distributed import (
    HardwareProfile,
    ModelScaleSpec,
    ParallelPlan,
    build_torchrun_command,
    estimate_training_memory,
    validate_topology,
)


def s0_model() -> ModelScaleSpec:
    return ModelScaleSpec(
        total_parameters=10_000,
        hidden_size=20,
        num_layers=1,
        num_attention_heads=2,
        sequence_length=128,
        micro_batch_size=2,
    )


def test_s0_single_device_plan_stays_simple() -> None:
    report = validate_topology(HardwareProfile(), s0_model(), ParallelPlan())
    assert report.valid
    assert report.world_size == 1
    assert "distributed runtime is unnecessary for S0" in report.notes[0]


def test_topology_world_size_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="world_size"):
        validate_topology(
            HardwareProfile(nodes=1, accelerators_per_node=2),
            s0_model(),
            ParallelPlan(),
        )


def test_tensor_parallel_requires_head_divisibility() -> None:
    model = ModelScaleSpec(1_000_000, 128, 4, 6, 256)
    with pytest.raises(ValueError, match="attention_heads"):
        validate_topology(
            HardwareProfile(nodes=1, accelerators_per_node=4),
            model,
            ParallelPlan(tensor_parallel=4),
        )


def test_memory_estimate_reports_transparent_breakdown() -> None:
    estimate = estimate_training_memory(s0_model(), ParallelPlan())
    assert estimate.parameter_bytes_per_rank == 20_000
    assert estimate.gradient_bytes_per_rank == 20_000
    assert estimate.optimizer_bytes_per_rank == 80_000
    assert estimate.master_weight_bytes_per_rank == 40_000
    assert estimate.activation_bytes_per_rank > 0
    assert estimate.total_bytes_per_rank == sum(
        (
            estimate.parameter_bytes_per_rank,
            estimate.gradient_bytes_per_rank,
            estimate.optimizer_bytes_per_rank,
            estimate.master_weight_bytes_per_rank,
            estimate.activation_bytes_per_rank,
        )
    )


def test_fsdp_style_sharding_reduces_model_state_estimate() -> None:
    model = ModelScaleSpec(1_000_000, 128, 4, 4, 256)
    unsharded = estimate_training_memory(model, ParallelPlan(data_parallel=4))
    sharded = estimate_training_memory(
        model,
        ParallelPlan(data_parallel=4, shard_model_state_across_data_parallel=True),
    )
    assert sharded.parameter_bytes_per_rank == unsharded.parameter_bytes_per_rank // 4


def test_torchrun_builder_only_constructs_command() -> None:
    command = build_torchrun_command(
        "train.py",
        HardwareProfile(nodes=2, accelerators_per_node=8),
        node_rank=1,
        master_addr="10.0.0.1",
        script_args=("--config", "s4.yaml"),
    )
    assert command[:3] == ("torchrun", "--nnodes=2", "--nproc-per-node=8")
    assert command[-3:] == ("train.py", "--config", "s4.yaml")
