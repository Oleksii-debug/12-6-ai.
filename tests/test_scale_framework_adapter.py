from __future__ import annotations

from pathlib import Path

import pytest
import torch

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.framework_adapter import (
    assess_torchtitan_fit,
    build_scale_framework_plan,
    execute_local_scale_smoke,
    probe_torchtitan,
    step_metrics_event,
)
from twelve_six.model import InitSpec, ModelSpec, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import StepMetrics

_ROOT = Path(__file__).resolve().parents[1]


def _stage(name: str):
    return load_stage_config(_ROOT / "configs" / "stages" / name)


def test_scale_plan_preserves_project_identities_and_maps_runtime() -> None:
    stage = _stage("s2_1m.json")
    trainer = TrainerConfig(
        learning_rate=2e-4,
        weight_decay=0.05,
        max_steps=4,
    )
    parallel = ParallelPlan(
        data_parallel=2,
        tensor_parallel=2,
        pipeline_parallel=1,
        context_parallel=1,
        shard_model_state_across_data_parallel=True,
    )

    plan = build_scale_framework_plan(stage.model, stage.init, trainer, parallel)

    assert plan.selected_backend == "pytorch_native"
    assert plan.model.model_spec_identity_sha256 == stage.expected_model_identity_sha256
    assert plan.model.init_spec_identity_sha256 == stage.expected_init_identity_sha256
    assert plan.model.parameter_count == stage.expected_parameters
    assert plan.model.weight_origin == "scratch_random_init_only"
    assert plan.distributed.physical_mesh_dim_names == ("pp", "dp", "cp", "tp")
    assert plan.distributed.physical_mesh_shape == (1, 2, 1, 2)
    assert plan.distributed.use_fsdp2 is True
    assert plan.distributed.use_tensor_parallel is True
    assert plan.dataset.effective_data_parallel_degree == 2
    assert plan.optimizer.learning_rate == 2e-4
    assert plan.optimizer.weight_decay == 0.05
    assert plan.checkpoint.semantic_identity_owner == "D05_D18_12-6_checkpoint_identity"
    assert plan.checkpoint.scale_storage == "torch.distributed.checkpoint"


def test_future_100m_planning_uses_same_identity_preserving_seam() -> None:
    model = ModelSpec(
        schema_version=1,
        vocab_size=32_000,
        max_seq_len=2_048,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        head_dim=64,
        d_ff=2_048,
        rope_rotary_dim=64,
    )
    init = InitSpec()
    parallel = ParallelPlan(
        data_parallel=4,
        tensor_parallel=2,
        pipeline_parallel=1,
        context_parallel=1,
        shard_model_state_across_data_parallel=True,
    )

    plan = build_scale_framework_plan(
        model,
        init,
        TrainerConfig(max_steps=1),
        parallel,
    )

    assert model.parameter_count() == 100_092_672
    assert plan.model.parameter_count == 100_092_672
    assert plan.model.model_spec_identity_sha256 == model.identity_sha256()
    assert plan.model.init_spec_identity_sha256 == init.identity_sha256()
    assert plan.model.weight_origin == "scratch_random_init_only"
    assert plan.selected_backend == "pytorch_native"
    assert plan.distributed.physical_mesh_shape == (1, 4, 1, 2)
    assert plan.distributed.use_fsdp2 is True
    assert plan.distributed.use_tensor_parallel is True


def test_current_model_fails_closed_for_direct_torchtitan_registration() -> None:
    stage = _stage("s1_100k.json")
    fit = assess_torchtitan_fit()

    assert fit.direct_adoption_ready is False
    assert "torchtitan_training_driver_adapter_not_implemented" in fit.blockers
    assert "model_missing_torchtitan_nested_config" in fit.blockers
    assert "model_missing_torchtitan_init_states_protocol" in fit.blockers
    assert "model_missing_torchtitan_parallelize_protocol" in fit.blockers
    assert "post_materialization_initialization_contract_missing" in fit.blockers

    with pytest.raises(RuntimeError, match="direct TorchTitan adoption is blocked"):
        build_scale_framework_plan(
            stage.model,
            stage.init,
            TrainerConfig(max_steps=1),
            ParallelPlan(),
            backend="torchtitan",
        )


def test_torchtitan_probe_does_not_initialize_distributed_runtime() -> None:
    before = torch.distributed.is_initialized()
    result = probe_torchtitan()
    after = torch.distributed.is_initialized()

    assert before == after
    assert result.installed is bool(result.installed)
    if not result.installed:
        assert result.available_components == ()
        assert result.missing_components


def test_metrics_adapter_preserves_d02_step_fields() -> None:
    metrics = StepMetrics(
        micro_step=3,
        optimizer_step=1,
        loss=2.0,
        update_loss=1.5,
        learning_rate=3e-4,
        grad_norm=0.75,
        tokens=128,
        optimizer_stepped=True,
    )

    event = step_metrics_event(metrics)

    assert event == {
        "schema": "12-6.scale-step-metrics.v1",
        "micro_step": 3,
        "optimizer_step": 1,
        "loss": 2.0,
        "update_loss": 1.5,
        "learning_rate": 3e-4,
        "grad_norm": 0.75,
        "tokens": 128,
        "optimizer_stepped": True,
    }


def test_local_smoke_executes_real_scratch_model_optimizer_step() -> None:
    stage = _stage("s1_100k.json")
    trainer = TrainerConfig(max_steps=1, seed=19019)
    before = torch.distributed.is_initialized()

    evidence = execute_local_scale_smoke(
        stage.model,
        stage.init,
        trainer,
        ParallelPlan(),
        sequence_length=8,
    )

    assert evidence["schema"] == "12-6.scale-framework-local-smoke.v1"
    assert evidence["parameter_count"] == stage.expected_parameters
    assert evidence["model_spec_identity_sha256"] == stage.expected_model_identity_sha256
    assert evidence["init_spec_identity_sha256"] == stage.expected_init_identity_sha256
    assert evidence["weight_origin"] == "scratch_random_init_only"
    assert evidence["state_sha256_before"] != evidence["state_sha256_after"]
    assert evidence["distributed_initialized_before"] == before
    assert evidence["distributed_initialized_after"] == before
    assert evidence["framework_plan"]["selected_backend"] == "pytorch_native"
