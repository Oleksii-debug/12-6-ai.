from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from twelve_six.distributed.checkpointing import (
    D05CheckpointRef,
    DistributedCheckpointEnvelope,
    DistributedShardRecord,
    ResumeMode,
    decide_resume,
)
from twelve_six.distributed.contracts import ModelScaleSpec, ParallelPlan
from twelve_six.distributed.memory import estimate_training_memory
from twelve_six.distributed.profiling import measure_model_parameter_bytes
from twelve_six.distributed.rank_layout import RankLayout
from twelve_six.distributed.runtime import build_torch_mesh_spec, choose_backend

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
GIT_SHA = "f" * 40


def _d05_manifest() -> dict[str, object]:
    return {
        "format": "12-6-checkpoint",
        "format_version": 1,
        "checkpoint_id": SHA_C,
        "identity": {
            "git_sha": GIT_SHA,
            "model_spec_hash": SHA_D,
            "run_manifest_hash": SHA_E,
            "environment_lock_hash": SHA_A,
            "step": 12,
            "tokens_seen": 3456,
        },
    }


def _envelope(layout: RankLayout) -> DistributedCheckpointEnvelope:
    ref = D05CheckpointRef.from_manifest(_d05_manifest(), manifest_sha256=SHA_B)
    return DistributedCheckpointEnvelope(
        d05_parent=ref,
        save_layout_sha256=layout.identity_sha256,
        save_world_size=layout.plan.world_size,
        state_dict_schema_sha256="1" * 64,
        shards=tuple(
            DistributedShardRecord(
                relative_path=f"__{rank}_0.distcp",
                sha256=f"{rank + 2:x}" * 64,
                size_bytes=100 + rank,
                writer_rank=rank,
            )
            for rank in range(layout.plan.world_size)
        ),
        rank_rng_sha256=tuple(f"{rank + 8:x}" * 64 for rank in range(layout.plan.world_size)),
    )


def test_rank_layout_roundtrips_dp_tp_pp_cp_and_groups() -> None:
    plan = ParallelPlan(
        data_parallel=2,
        tensor_parallel=2,
        pipeline_parallel=2,
        context_parallel=2,
    )
    layout = RankLayout(plan)
    assert plan.world_size == 16
    for rank in range(plan.world_size):
        assert layout.rank(layout.coordinate(rank)) == rank
        assert len(layout.axis_group(rank, "dp")) == 2
        assert len(layout.axis_group(rank, "tp")) == 2
        assert len(layout.axis_group(rank, "pp")) == 2
        assert len(layout.axis_group(rank, "cp")) == 2
        assert len(layout.dense_gradient_sync_group(rank)) == 4


def test_expert_parallel_is_subgroup_of_project_dp_total() -> None:
    plan = ParallelPlan(
        data_parallel=4,
        tensor_parallel=2,
        context_parallel=2,
        expert_parallel=2,
    )
    layout = RankLayout(plan)
    assert plan.world_size == 16
    assert plan.expert_data_parallel == 2
    for rank in range(plan.world_size):
        dp_group = set(layout.axis_group(rank, "dp"))
        assert set(layout.expert_parallel_group(rank)).issubset(dp_group)
        assert set(layout.expert_data_parallel_group(rank)).issubset(dp_group)
        assert len(layout.expert_parallel_group(rank)) == 2
        assert len(layout.expert_data_parallel_group(rank)) == 2


def test_torch_mesh_splits_project_dp_into_fsdp_replicate_and_shard() -> None:
    plan = ParallelPlan(
        data_parallel=8,
        tensor_parallel=2,
        pipeline_parallel=2,
        context_parallel=2,
        expert_parallel=4,
    )
    spec = build_torch_mesh_spec(plan, fsdp_shard_degree=4)
    assert spec.shape == (2, 4, 2, 2, 2)
    assert spec.world_size == plan.world_size
    assert spec.megatron_axis_degrees == {"tp": 2, "pp": 2, "cp": 2, "ep": 4, "dp": 2}
    with pytest.raises(ValueError, match="EP>1"):
        spec.fsdp2_kwargs(object())


def test_fsdp_shard_degree_must_divide_project_dp() -> None:
    with pytest.raises(ValueError, match="must divide"):
        build_torch_mesh_spec(ParallelPlan(data_parallel=6), fsdp_shard_degree=4)


def test_d05_identity_is_separate_from_distributed_physical_shards() -> None:
    layout = RankLayout(ParallelPlan(data_parallel=2))
    envelope = _envelope(layout)
    envelope.validate()
    ref = envelope.d05_parent
    assert ref.semantic_parent_sha256 == ref.identity_sha256
    assert ref.checkpoint_id == SHA_C
    reordered = replace(envelope, shards=tuple(reversed(envelope.shards)))
    assert reordered.artifact_set_sha256 == envelope.artifact_set_sha256
    assert reordered.envelope_sha256 == envelope.envelope_sha256


def test_reshard_resume_preserves_state_identity_but_not_bitwise_rng_claim() -> None:
    source = RankLayout(ParallelPlan(data_parallel=2))
    target = RankLayout(ParallelPlan(data_parallel=4))
    envelope = _envelope(source)
    decision = decide_resume(
        envelope,
        target,
        mode=ResumeMode.RESHARD,
        target_state_dict_schema_sha256="1" * 64,
    )
    assert decision.allowed
    assert not decision.exact_trajectory_claim_allowed
    assert "no-bitwise" in decision.rng_policy
    exact = decide_resume(
        envelope,
        target,
        mode=ResumeMode.EXACT_TOPOLOGY,
        target_state_dict_schema_sha256="1" * 64,
    )
    assert not exact.allowed


def test_resume_fails_closed_on_state_dict_schema_change() -> None:
    layout = RankLayout(ParallelPlan(data_parallel=2))
    decision = decide_resume(
        _envelope(layout),
        layout,
        mode=ResumeMode.RESHARD,
        target_state_dict_schema_sha256="9" * 64,
    )
    assert not decision.allowed
    assert decision.rng_policy == "blocked"


def test_shard_paths_cannot_escape_checkpoint_directory() -> None:
    layout = RankLayout(ParallelPlan())
    envelope = _envelope(layout)
    bad = replace(
        envelope,
        shards=(DistributedShardRecord("../escape", SHA_A, 1, 0),),
    )
    with pytest.raises(ValueError, match="inside the checkpoint"):
        bad.validate()


def test_parameter_memory_estimator_matches_small_local_fp32_measurement() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    )
    measured = measure_model_parameter_bytes(model)
    assert measured.element_count == 40
    assert measured.total_bytes == 160
    estimate = estimate_training_memory(
        ModelScaleSpec(
            total_parameters=40,
            hidden_size=8,
            num_layers=1,
            num_attention_heads=1,
            sequence_length=1,
        ),
        ParallelPlan(),
        parameter_bytes=4,
        gradient_bytes=0,
        optimizer_bytes_per_parameter=0,
        master_weight_bytes=0,
        activation_bytes=0,
        activation_multiplier=0,
    )
    assert estimate.parameter_bytes_per_rank == measured.total_bytes
    assert estimate.total_bytes_per_rank == measured.total_bytes


def test_backend_adoption_stays_simple_until_scale_signal() -> None:
    simple = choose_backend(
        requires_moe=False,
        requires_pipeline_parallel=False,
        requires_context_parallel=False,
        requires_float8=False,
        requires_distributed_checkpoint_reshard=False,
        nvidia_only_scale_target=False,
    )
    assert simple.backend == "single-device-or-native-pytorch"
    moe = choose_backend(
        requires_moe=True,
        requires_pipeline_parallel=True,
        requires_context_parallel=True,
        requires_float8=True,
        requires_distributed_checkpoint_reshard=True,
        nvidia_only_scale_target=True,
    )
    assert moe.backend == "megatron-core"
