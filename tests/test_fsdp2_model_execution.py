from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.fsdp2_training import (
    run_local_cpu_fsdp2,
    write_execution_evidence,
)
from twelve_six.distributed.runtime import build_torch_mesh_spec

ROOT = Path(__file__).resolve().parents[1]


class _RecordingMesh:
    def __init__(self) -> None:
        self.keys: list[object] = []

    def __getitem__(self, key: object) -> object:
        self.keys.append(key)
        return ("submesh", key)


def test_plain_tensor_fsdp2_uses_only_data_parallel_submesh() -> None:
    mesh = _RecordingMesh()
    fsdp = build_torch_mesh_spec(ParallelPlan(data_parallel=4), fsdp_shard_degree=4)
    kwargs = fsdp.fsdp2_kwargs(mesh)
    assert kwargs["mesh"] == ("submesh", "dp_shard")
    assert "dp_mesh_dims" not in kwargs

    hsdp_mesh = _RecordingMesh()
    hsdp = build_torch_mesh_spec(ParallelPlan(data_parallel=4), fsdp_shard_degree=2)
    hsdp_kwargs = hsdp.fsdp2_kwargs(hsdp_mesh)
    assert hsdp_kwargs["mesh"] == ("submesh", ("dp_replicate", "dp_shard"))
    assert "dp_mesh_dims" not in hsdp_kwargs


def test_local_cpu_gloo_fsdp2_executes_real_s1_shaped_model() -> None:
    result = run_local_cpu_fsdp2(
        ROOT / "configs" / "stages" / "s1_100k.json",
        world_size=2,
        samples_per_rank=1,
        sequence_length=8,
        exercise_error_recovery=True,
    )
    assert result.world_size == 2
    assert result.backend == "gloo"
    assert result.device_type == "cpu"
    assert result.parameter_count == 107_856
    assert result.ranks_seen == (0, 1)
    assert result.sampler_indices == ((0,), (1,))
    assert result.global_tokens == 14
    assert math.isfinite(result.reduced_loss)
    assert result.grad_norm_min > 0
    assert result.grad_norm_max >= result.grad_norm_min
    assert result.optimizer_steps == (1, 1)
    assert result.global_parameter_update_l1 > 0
    assert result.parameters_are_dtensor
    assert result.error_recovery_exercised
    assert result.clean_shutdown

    evidence_out = os.environ.get("TWELVE_SIX_FSDP2_EVIDENCE_OUT")
    source_sha = os.environ.get("TWELVE_SIX_FSDP2_SOURCE_SHA")
    if evidence_out or source_sha:
        assert evidence_out is not None
        assert source_sha is not None
        evidence = write_execution_evidence(
            result,
            source_sha=source_sha,
            output_path=evidence_out,
        )
        assert evidence["claims"]["gpu_nccl_executed"] is False
        reloaded = json.loads(Path(evidence_out).read_text(encoding="utf-8"))
        assert reloaded["evidence_sha256"] == evidence["evidence_sha256"]


def test_local_cpu_gloo_fsdp2_propagates_rank_failure_without_hanging() -> None:
    with pytest.raises(RuntimeError, match="child failures"):
        run_local_cpu_fsdp2(
            ROOT / "configs" / "stages" / "s0_10k.json",
            world_size=2,
            samples_per_rank=1,
            sequence_length=4,
            exercise_error_recovery=False,
            inject_failure_rank=1,
            timeout_seconds=60,
        )
