from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.framework_adoption import (
    AdoptionSignals,
    build_megatron_core_plan,
    build_native_integration_plan,
    decision_payload,
    framework_assessments,
    optional_framework_availability,
    recommend_framework,
)
from twelve_six.model import InitSpec, ModelSpec, load_stage_config


def _gqa_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=4096,
        max_seq_len=2048,
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        head_dim=64,
        d_ff=1408,
        rope_rotary_dim=32,
    )


def test_decision_matrix_selects_only_native_and_megatron() -> None:
    assessments = framework_assessments()
    selected = {item.name for item in assessments if item.selected_path}
    assert selected == {"pytorch_native", "megatron_core"}


def test_native_adapter_preserves_real_s3_identity() -> None:
    stage = load_stage_config(Path("configs/stages/s3_10m.json"))
    plan = ParallelPlan(
        data_parallel=2,
        tensor_parallel=2,
        shard_model_state_across_data_parallel=True,
    )
    native = build_native_integration_plan(stage.model, stage.init, plan)
    assert native.model_spec_sha256 == stage.expected_model_identity_sha256
    assert native.init_spec_sha256 == stage.expected_init_identity_sha256
    assert native.topology.physical_mesh_shape == (1, 2, 1, 2)
    assert native.topology.use_fsdp2
    assert native.topology.use_tensor_parallel
    assert native.simple_tp_compatible
    assert not native.model_rewrite_required


def test_megatron_adapter_maps_gqa_partial_rope_and_init() -> None:
    spec = _gqa_spec()
    init_spec = InitSpec(std=0.02)
    plan = ParallelPlan(data_parallel=2, tensor_parallel=2, pipeline_parallel=2)
    mapped = build_megatron_core_plan(spec, init_spec, plan)
    assert mapped.transformer_config["num_query_groups"] == 2
    assert mapped.transformer_config["kv_channels"] == 64
    assert mapped.transformer_config["rotary_interleaved"] is True
    assert mapped.transformer_config["rotary_percent"] == 0.5
    assert mapped.transformer_config["gated_linear_unit"] is True
    assert mapped.transformer_config["activation_func"] == "torch.nn.functional.silu"
    assert mapped.parallel_config["world_size"] == 8
    assert mapped.parallel_config["expert_model_parallel_size"] == 1
    assert mapped.output_layer_init_std == pytest.approx(0.02 / math.sqrt(16.0))


def test_megatron_adapter_rejects_d12_ep_subgroup_semantics() -> None:
    spec = _gqa_spec()
    with pytest.raises(ValueError, match="topology v2"):
        build_megatron_core_plan(
            spec,
            InitSpec(),
            ParallelPlan(data_parallel=4, expert_parallel=2),
        )


def test_framework_gate_keeps_native_until_measured_validated_win() -> None:
    plan = ParallelPlan(data_parallel=2, tensor_parallel=2, pipeline_parallel=2)
    candidate = recommend_framework(
        plan,
        AdoptionSignals(
            nvidia_cluster=True,
            measured_megatron_speedup=1.30,
            megatron_runtime_validated=False,
        ),
    )
    assert candidate.incumbent == "pytorch_native"
    assert candidate.benchmark_alternative == "megatron_core"
    assert not candidate.migration_authorized

    accepted = recommend_framework(
        plan,
        AdoptionSignals(
            nvidia_cluster=True,
            measured_megatron_speedup=1.30,
            megatron_runtime_validated=True,
        ),
    )
    assert accepted.incumbent == "megatron_core"
    assert accepted.migration_authorized


def test_moe_never_migrates_before_topology_v2() -> None:
    decision = recommend_framework(
        ParallelPlan(data_parallel=4, expert_parallel=2),
        AdoptionSignals(
            nvidia_cluster=True,
            measured_megatron_speedup=2.0,
            megatron_runtime_validated=True,
        ),
    )
    assert decision.incumbent == "pytorch_native"
    assert decision.benchmark_alternative == "megatron_core"
    assert not decision.migration_authorized
    assert "topology v2" in decision.trigger


def test_payload_is_json_ready_and_optional_probe_is_non_importing() -> None:
    spec = _gqa_spec()
    payload = decision_payload(
        spec,
        InitSpec(),
        ParallelPlan(data_parallel=2, tensor_parallel=2),
        AdoptionSignals(nvidia_cluster=False),
    )
    json.dumps(payload, sort_keys=True)
    availability = optional_framework_availability()
    assert set(availability) == {"torchtitan", "olmo_core", "megatron_core"}
    assert all(isinstance(value, bool) for value in availability.values())
