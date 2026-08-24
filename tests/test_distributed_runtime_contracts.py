from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from twelve_six.distributed import (
    ModelScaleSpec,
    ParallelPlan,
    ShardRecord,
    bind_d05_manifest,
    build_distributed_checkpoint_manifest,
    build_torch_native_plan,
    coordinate_for_rank,
    estimate_training_memory,
    fake_process_groups,
    measure_module_state,
    plan_resume,
    rank_identity,
    torch_native_capabilities,
    verify_distributed_checkpoint_manifest,
)


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _d05_manifest() -> dict[str, object]:
    identity = {
        "git_sha": "a" * 40,
        "model_spec_hash": "b" * 64,
        "tokenizer_hash": "c" * 64,
        "tokenizer_vocab_hash": "d" * 64,
        "dataset_manifest_hash": "e" * 64,
        "run_manifest_hash": "f" * 64,
        "training_config_hash": "1" * 64,
        "optimizer_hash": "2" * 64,
        "scheduler_hash": "3" * 64,
        "environment_hash": "4" * 64,
        "environment_lock_hash": "5" * 64,
        "step": 7,
        "tokens_seen": 8192,
    }
    files = {"weights.safetensors": {"sha256": "6" * 64, "bytes": 100}}
    return {
        "format": "12-6-checkpoint",
        "format_version": 1,
        "identity": identity,
        "files": files,
        "checkpoint_id": _hash_json({"identity": identity, "files": files}),
    }


def _one_shard(plan: ParallelPlan) -> ShardRecord:
    payload = b"d12-test-shard"
    return ShardRecord(
        relative_path="rank-00000.distcp",
        writer_rank=0,
        rank_identity=rank_identity(plan, 0),
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


def test_rank_coordinate_roundtrip_and_all_physical_axes() -> None:
    plan = ParallelPlan(
        data_parallel=4,
        tensor_parallel=2,
        pipeline_parallel=2,
        context_parallel=2,
        expert_parallel=2,
    )
    coordinates = [coordinate_for_rank(rank, plan) for rank in range(plan.world_size)]
    assert len(coordinates) == 32
    assert {item.data_parallel_rank for item in coordinates} == {0, 1, 2, 3}
    assert {item.tensor_parallel_rank for item in coordinates} == {0, 1}
    assert {item.pipeline_parallel_rank for item in coordinates} == {0, 1}
    assert {item.context_parallel_rank for item in coordinates} == {0, 1}
    assert {item.expert_parallel_rank for item in coordinates} == {0, 1}
    assert {item.expert_data_parallel_rank for item in coordinates} == {0, 1}


def test_fake_groups_model_ep_as_dp_subgroup_not_physical_axis() -> None:
    plan = ParallelPlan(data_parallel=4, tensor_parallel=2, expert_parallel=2)
    groups = fake_process_groups(plan)
    assert plan.world_size == 8
    assert all(len(group) == 4 for group in groups.data_parallel)
    assert all(len(group) == 2 for group in groups.tensor_parallel)
    assert all(len(group) == 2 for group in groups.expert_parallel)
    assert all(len(group) == 2 for group in groups.expert_data_parallel)
    for rank in range(plan.world_size):
        assert sum(rank in group for group in groups.data_parallel) == 1
        assert sum(rank in group for group in groups.tensor_parallel) == 1
        assert sum(rank in group for group in groups.expert_parallel) == 1
        assert sum(rank in group for group in groups.expert_data_parallel) == 1


def test_fake_groups_cover_pipeline_and_context_axes() -> None:
    plan = ParallelPlan(data_parallel=2, pipeline_parallel=3, context_parallel=2)
    groups = fake_process_groups(plan)
    assert all(len(group) == 3 for group in groups.pipeline_parallel)
    assert all(len(group) == 2 for group in groups.context_parallel)
    for rank in range(plan.world_size):
        assert sum(rank in group for group in groups.pipeline_parallel) == 1
        assert sum(rank in group for group in groups.context_parallel) == 1


def test_torch_native_plan_is_noop_for_s0_and_lazy() -> None:
    native = build_torch_native_plan(ParallelPlan())
    assert native.physical_mesh_dim_names == ("pp", "dp", "cp", "tp")
    assert native.physical_mesh_shape == (1, 1, 1, 1)
    assert not native.requires_distributed_runtime
    assert not native.use_fsdp2


def test_torch_native_plan_exposes_fsdp2_tp_pp_cp_seams() -> None:
    native = build_torch_native_plan(
        ParallelPlan(
            data_parallel=2,
            tensor_parallel=2,
            pipeline_parallel=2,
            context_parallel=2,
            shard_model_state_across_data_parallel=True,
        )
    )
    assert native.physical_mesh_shape == (2, 2, 2, 2)
    assert native.requires_distributed_runtime
    assert native.use_fsdp2
    assert native.use_tensor_parallel
    assert native.use_pipeline_parallel
    assert native.use_context_parallel


def test_current_locked_torch_exposes_native_scale_apis() -> None:
    capabilities = torch_native_capabilities()
    assert capabilities.fsdp2_fully_shard
    assert capabilities.dtensor
    assert capabilities.tensor_parallel
    assert capabilities.distributed_checkpoint


def test_cpu_measurement_matches_dense_state_estimator_coefficients() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(3, 2, bias=True)
    module(torch.ones(1, 3)).sum().backward()
    measured = measure_module_state(module)
    assert measured.parameter_count == 8
    assert measured.parameter_bytes == 32
    assert measured.gradient_bytes == 32

    estimate = estimate_training_memory(
        ModelScaleSpec(8, 1, 1, 1, 1),
        ParallelPlan(),
        parameter_bytes=4,
        gradient_bytes=4,
        optimizer_bytes_per_parameter=0,
        master_weight_bytes=0,
        activation_bytes=0,
        activation_multiplier=0,
    )
    assert estimate.parameter_bytes_per_rank == measured.parameter_bytes
    assert estimate.gradient_bytes_per_rank == measured.gradient_bytes
    assert estimate.total_bytes_per_rank == 64


def test_d05_logical_identity_survives_layout_reshard() -> None:
    binding = bind_d05_manifest(_d05_manifest())
    saved = ParallelPlan(data_parallel=2, shard_model_state_across_data_parallel=True)
    manifest = build_distributed_checkpoint_manifest(
        d05=binding,
        saved_plan=saved,
        backend_format="torch_dcp",
        shards=(_one_shard(saved),),
        reshardable=True,
        optimizer_reshardable=True,
    )
    verify_distributed_checkpoint_manifest(manifest)
    resume = plan_resume(
        manifest,
        ParallelPlan(data_parallel=4, shard_model_state_across_data_parallel=True),
    )
    assert resume.mode == "backend_reshard"
    assert resume.source_world_size == 2
    assert resume.target_world_size == 4
    assert resume.preserves_d05_identity_sha256 == binding.identity_sha256


def test_layout_checksum_is_order_independent_but_rank_and_bytes_are_bound() -> None:
    binding = bind_d05_manifest(_d05_manifest())
    plan = ParallelPlan(data_parallel=2)
    first = _one_shard(plan)
    second = ShardRecord(
        relative_path="rank-00001.distcp",
        writer_rank=1,
        rank_identity=rank_identity(plan, 1),
        sha256="7" * 64,
        byte_count=21,
    )
    left = build_distributed_checkpoint_manifest(
        d05=binding,
        saved_plan=plan,
        backend_format="torch_dcp",
        shards=(first, second),
        reshardable=True,
        optimizer_reshardable=True,
    )
    right = build_distributed_checkpoint_manifest(
        d05=binding,
        saved_plan=plan,
        backend_format="torch_dcp",
        shards=(second, first),
        reshardable=True,
        optimizer_reshardable=True,
    )
    assert left.artifact_set_sha256 == right.artifact_set_sha256
    assert left.layout_id == right.layout_id

    changed = replace(first, writer_rank=1, rank_identity=rank_identity(plan, 1))
    mutated = build_distributed_checkpoint_manifest(
        d05=binding,
        saved_plan=plan,
        backend_format="torch_dcp",
        shards=(changed, second),
        reshardable=True,
        optimizer_reshardable=True,
    )
    assert mutated.layout_id != left.layout_id


def test_non_reshardable_layout_fails_closed_on_topology_change() -> None:
    binding = bind_d05_manifest(_d05_manifest())
    plan = ParallelPlan(data_parallel=2)
    manifest = build_distributed_checkpoint_manifest(
        d05=binding,
        saved_plan=plan,
        backend_format="rank_local_custom",
        shards=(_one_shard(plan),),
        reshardable=False,
        optimizer_reshardable=False,
    )
    with pytest.raises(ValueError, match="not reshardable"):
        plan_resume(manifest, ParallelPlan(data_parallel=4))


def test_manifest_tamper_is_rejected() -> None:
    binding = bind_d05_manifest(_d05_manifest())
    plan = ParallelPlan(data_parallel=2)
    manifest = build_distributed_checkpoint_manifest(
        d05=binding,
        saved_plan=plan,
        backend_format="torch_dcp",
        shards=(_one_shard(plan),),
        reshardable=True,
        optimizer_reshardable=True,
    )
    with pytest.raises(ValueError, match="artifact_set_sha256"):
        verify_distributed_checkpoint_manifest(replace(manifest, artifact_set_sha256="0" * 64))
