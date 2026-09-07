from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from itertools import product
from math import prod

import pytest

from twelve_six.distributed.checkpointing import (
    D05CheckpointRef,
    DistributedCheckpointEnvelope,
    DistributedShardRecord,
    ResumeMode,
    decide_resume,
)
from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.cpu_probe import run_cpu_gloo_probe
from twelve_six.distributed.rank_layout import RankLayout
from twelve_six.distributed.runtime import build_torch_mesh_spec, choose_backend


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(candidate for candidate in range(1, value + 1) if value % candidate == 0)


def _valid_plans() -> tuple[ParallelPlan, ...]:
    plans: list[ParallelPlan] = []
    for data_parallel, tensor_parallel, pipeline_parallel, context_parallel in product(
        (1, 2, 4, 6, 8),
        (1, 2),
        (1, 2),
        (1, 2),
    ):
        for expert_parallel in _divisors(data_parallel):
            plans.append(
                ParallelPlan(
                    data_parallel=data_parallel,
                    tensor_parallel=tensor_parallel,
                    pipeline_parallel=pipeline_parallel,
                    context_parallel=context_parallel,
                    expert_parallel=expert_parallel,
                )
            )
    return tuple(plans)


def _d05_ref() -> D05CheckpointRef:
    return D05CheckpointRef(
        checkpoint_id=_digest("d05-checkpoint"),
        manifest_sha256=_digest("d05-manifest"),
        identity_sha256=_digest("d05-logical-identity"),
        git_sha="f" * 40,
        model_spec_hash=_digest("model-spec"),
        run_manifest_hash=_digest("run-manifest"),
        environment_lock_hash=_digest("environment-lock"),
        step=17,
        tokens_seen=65536,
    )


def _envelope(layout: RankLayout) -> DistributedCheckpointEnvelope:
    return DistributedCheckpointEnvelope(
        d05_parent=_d05_ref(),
        save_layout_sha256=layout.identity_sha256,
        save_world_size=layout.plan.world_size,
        state_dict_schema_sha256=_digest("canonical-state-dict-schema"),
        shards=tuple(
            DistributedShardRecord(
                relative_path=f"rank-{rank:05d}.distcp",
                sha256=_digest(f"shard-{rank}"),
                size_bytes=1024 + rank,
                writer_rank=rank,
            )
            for rank in range(layout.plan.world_size)
        ),
        rank_rng_sha256=tuple(
            _digest(f"rank-rng-{rank}") for rank in range(layout.plan.world_size)
        ),
    )


def test_exhaustive_small_topology_matrix_is_bijective_and_partitioned() -> None:
    case_count = 0
    coordinate_count = 0
    mesh_factorization_count = 0

    for plan in _valid_plans():
        plan.validate()
        case_count += 1
        layout = RankLayout(plan)
        all_ranks = set(range(plan.world_size))
        coordinates = {layout.coordinate(rank) for rank in all_ranks}
        coordinate_count += len(coordinates)

        assert len(coordinates) == plan.world_size
        assert layout.shape == (
            plan.data_parallel,
            plan.pipeline_parallel,
            plan.context_parallel,
            plan.tensor_parallel,
        )
        assert all(layout.rank(layout.coordinate(rank)) == rank for rank in all_ranks)

        axis_degrees = {
            "dp": plan.data_parallel,
            "pp": plan.pipeline_parallel,
            "cp": plan.context_parallel,
            "tp": plan.tensor_parallel,
        }
        for axis, degree in axis_degrees.items():
            groups = {layout.axis_group(rank, axis) for rank in all_ranks}
            assert len(groups) == plan.world_size // degree
            assert sum(len(group) for group in groups) == plan.world_size
            assert set().union(*(set(group) for group in groups)) == all_ranks
            assert all(len(group) == degree == len(set(group)) for group in groups)

        dp_groups = {layout.axis_group(rank, "dp") for rank in all_ranks}
        for dp_group in dp_groups:
            members = set(dp_group)
            ep_groups = {layout.expert_parallel_group(rank) for rank in members}
            edp_groups = {layout.expert_data_parallel_group(rank) for rank in members}
            assert len(ep_groups) == plan.expert_data_parallel
            assert len(edp_groups) == plan.expert_parallel
            assert sum(len(group) for group in ep_groups) == plan.data_parallel
            assert sum(len(group) for group in edp_groups) == plan.data_parallel
            assert set().union(*(set(group) for group in ep_groups)) == members
            assert set().union(*(set(group) for group in edp_groups)) == members
            assert all(len(group) == plan.expert_parallel for group in ep_groups)
            assert all(len(group) == plan.expert_data_parallel for group in edp_groups)

        dense_groups = {layout.dense_gradient_sync_group(rank) for rank in all_ranks}
        dense_degree = plan.data_parallel * plan.context_parallel
        assert len(dense_groups) == plan.world_size // dense_degree
        assert all(len(group) == dense_degree for group in dense_groups)
        assert set().union(*(set(group) for group in dense_groups)) == all_ranks

        for fsdp_shard_degree in _divisors(plan.data_parallel):
            mesh_factorization_count += 1
            mesh = build_torch_mesh_spec(plan, fsdp_shard_degree=fsdp_shard_degree)
            assert mesh.world_size == plan.world_size
            assert prod(mesh.shape) == plan.world_size
            assert mesh.logical_layout_sha256 == layout.identity_sha256
            assert mesh.fsdp_shard_degree == fsdp_shard_degree
            assert mesh.fsdp_replicate_degree * mesh.fsdp_shard_degree == plan.data_parallel
            megatron = mesh.megatron_axis_degrees
            assert megatron == {
                "tp": plan.tensor_parallel,
                "pp": plan.pipeline_parallel,
                "cp": plan.context_parallel,
                "ep": plan.expert_parallel,
                "dp": plan.expert_data_parallel,
            }
            assert prod(megatron.values()) == plan.world_size

    assert case_count == 112
    assert coordinate_count == 1971
    assert mesh_factorization_count == 368


def test_invalid_hsdp_factorizations_fail_closed() -> None:
    for data_parallel in (2, 4, 6, 8):
        with pytest.raises(ValueError, match="must divide"):
            build_torch_mesh_spec(
                ParallelPlan(data_parallel=data_parallel),
                fsdp_shard_degree=data_parallel + 1,
            )
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            build_torch_mesh_spec(ParallelPlan(data_parallel=4), fsdp_shard_degree=invalid)


def test_checkpoint_identity_layers_bind_physical_and_rng_changes_separately() -> None:
    layout = RankLayout(ParallelPlan(data_parallel=2, tensor_parallel=2))
    envelope = _envelope(layout)
    envelope.validate()

    reordered = replace(envelope, shards=tuple(reversed(envelope.shards)))
    assert reordered.artifact_set_sha256 == envelope.artifact_set_sha256
    assert reordered.envelope_sha256 == envelope.envelope_sha256

    first = envelope.shards[0]
    reassigned = replace(
        envelope,
        shards=(replace(first, writer_rank=1), *envelope.shards[1:]),
    )
    assert reassigned.artifact_set_sha256 != envelope.artifact_set_sha256
    assert reassigned.envelope_sha256 != envelope.envelope_sha256
    assert reassigned.d05_parent.semantic_parent_sha256 == envelope.d05_parent.semantic_parent_sha256

    changed_rng = replace(
        envelope,
        rank_rng_sha256=(_digest("replacement-rng"), *envelope.rank_rng_sha256[1:]),
    )
    assert changed_rng.artifact_set_sha256 == envelope.artifact_set_sha256
    assert changed_rng.envelope_sha256 != envelope.envelope_sha256
    assert changed_rng.d05_parent.semantic_parent_sha256 == envelope.d05_parent.semantic_parent_sha256


def test_resume_matrix_separates_reshardability_from_exact_rng_trajectory() -> None:
    source = RankLayout(ParallelPlan(data_parallel=2, tensor_parallel=2))
    same = RankLayout(ParallelPlan(data_parallel=2, tensor_parallel=2))
    different_same_world = RankLayout(ParallelPlan(data_parallel=4))
    envelope = _envelope(source)
    schema = envelope.state_dict_schema_sha256

    exact_same = decide_resume(
        envelope,
        same,
        mode=ResumeMode.EXACT_TOPOLOGY,
        target_state_dict_schema_sha256=schema,
    )
    assert exact_same.allowed
    assert exact_same.exact_trajectory_claim_allowed
    assert exact_same.rng_policy == "restore-rank-local-rng-by-logical-rank"

    reshard_same = decide_resume(
        envelope,
        same,
        mode=ResumeMode.RESHARD,
        target_state_dict_schema_sha256=schema,
    )
    assert reshard_same.allowed
    assert reshard_same.exact_trajectory_claim_allowed

    exact_changed = decide_resume(
        envelope,
        different_same_world,
        mode=ResumeMode.EXACT_TOPOLOGY,
        target_state_dict_schema_sha256=schema,
    )
    assert not exact_changed.allowed
    assert not exact_changed.exact_trajectory_claim_allowed

    reshard_changed = decide_resume(
        envelope,
        different_same_world,
        mode=ResumeMode.RESHARD,
        target_state_dict_schema_sha256=schema,
    )
    assert reshard_changed.allowed
    assert not reshard_changed.exact_trajectory_claim_allowed
    assert "no-bitwise" in reshard_changed.rng_policy

    schema_changed = decide_resume(
        envelope,
        same,
        mode=ResumeMode.RESHARD,
        target_state_dict_schema_sha256=_digest("different-schema"),
    )
    assert not schema_changed.allowed
    assert not schema_changed.exact_trajectory_claim_allowed
    assert schema_changed.rng_policy == "blocked"


def test_checkpoint_envelope_rejects_duplicate_paths_and_rng_cardinality_drift() -> None:
    layout = RankLayout(ParallelPlan(data_parallel=2))
    envelope = _envelope(layout)
    duplicate = replace(
        envelope,
        shards=(envelope.shards[0], replace(envelope.shards[1], relative_path="rank-00000.distcp")),
    )
    with pytest.raises(ValueError, match="duplicate shard path"):
        duplicate.validate()

    missing_rng = replace(envelope, rank_rng_sha256=envelope.rank_rng_sha256[:-1])
    with pytest.raises(ValueError, match="one digest per logical save rank"):
        missing_rng.validate()


def test_backend_adoption_precedence_is_total_over_boolean_trigger_space() -> None:
    seen_backends: set[str] = set()
    for requires_moe, requires_pp, requires_cp, requires_float8, requires_dcp, nvidia in product(
        (False, True),
        repeat=6,
    ):
        decision = choose_backend(
            requires_moe=requires_moe,
            requires_pipeline_parallel=requires_pp,
            requires_context_parallel=requires_cp,
            requires_float8=requires_float8,
            requires_distributed_checkpoint_reshard=requires_dcp,
            nvidia_only_scale_target=nvidia,
        )
        seen_backends.add(decision.backend)
        if requires_moe or (requires_pp and nvidia):
            expected = "megatron-core"
        elif requires_cp or requires_float8:
            expected = "torchtitan"
        elif requires_dcp:
            expected = "native-pytorch"
        elif requires_pp:
            expected = "olmo-core-or-native-pytorch"
        else:
            expected = "single-device-or-native-pytorch"
        assert decision.backend == expected
        assert decision.reasons

    assert seen_backends == {
        "megatron-core",
        "torchtitan",
        "native-pytorch",
        "olmo-core-or-native-pytorch",
        "single-device-or-native-pytorch",
    }


def test_cpu_probe_refuses_unbounded_local_process_fanout_before_execution() -> None:
    with pytest.raises(ValueError, match="world_size <= 8"):
        run_cpu_gloo_probe(ParallelPlan(data_parallel=9))
