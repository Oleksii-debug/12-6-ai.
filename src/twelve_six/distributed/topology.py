"""Distributed topology validation and launch-plan construction."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from .contracts import HardwareProfile, ModelScaleSpec, ParallelPlan


@dataclass(frozen=True, slots=True)
class TopologyReport:
    world_size: int
    valid: bool
    notes: tuple[str, ...]


def validate_topology(
    hardware: HardwareProfile,
    model: ModelScaleSpec,
    plan: ParallelPlan,
) -> TopologyReport:
    hardware.validate()
    model.validate()
    plan.validate()
    notes: list[str] = []

    if hardware.world_size != plan.world_size:
        raise ValueError(
            f"parallel plan world_size={plan.world_size} does not match "
            f"hardware world_size={hardware.world_size}"
        )
    if model.num_attention_heads % plan.tensor_parallel != 0:
        raise ValueError("num_attention_heads must be divisible by tensor_parallel")
    if model.num_layers % plan.pipeline_parallel != 0:
        raise ValueError(
            "num_layers must be divisible by pipeline_parallel for the baseline stage plan"
        )
    if model.sequence_length % plan.context_parallel != 0:
        raise ValueError("sequence_length must be divisible by context_parallel")

    if hardware.world_size == 1:
        notes.append("single-device baseline: distributed runtime is unnecessary for S0")
    if plan.shard_model_state_across_data_parallel and plan.data_parallel == 1:
        notes.append("model-state sharding flag has no effect with data_parallel=1")
    if plan.expert_parallel > 1:
        notes.append(
            "expert parallelism is a subgroup of data parallelism and requires an MoE-capable "
            "backend; dense S0 must keep EP=1"
        )
    return TopologyReport(world_size=hardware.world_size, valid=True, notes=tuple(notes))


def build_torchrun_command(
    training_script: str | Path,
    hardware: HardwareProfile,
    *,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    script_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Build, but never execute, an exact torchrun command tuple."""

    hardware.validate()
    if not 0 <= node_rank < hardware.nodes:
        raise ValueError("node_rank must be inside [0, nodes)")
    if not 1 <= master_port <= 65535:
        raise ValueError("master_port must be in [1, 65535]")
    if not master_addr:
        raise ValueError("master_addr must be non-empty")
    return (
        "torchrun",
        f"--nnodes={hardware.nodes}",
        f"--nproc-per-node={hardware.accelerators_per_node}",
        f"--node-rank={node_rank}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        str(training_script),
        *script_args,
    )


def shell_join(command: tuple[str, ...]) -> str:
    """Return a copy/paste-safe display form; execution remains caller-controlled."""

    return shlex.join(command)
