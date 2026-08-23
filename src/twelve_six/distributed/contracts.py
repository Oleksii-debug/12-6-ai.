"""Framework-neutral distributed planning contracts.

These structures intentionally do not import torch. S0 remains single-device while later backends
(FSDP2/TorchTitan/OLMo-core/Megatron Core) can translate the same explicit plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    nodes: int = 1
    accelerators_per_node: int = 1
    device_memory_gib: float | None = None

    @property
    def world_size(self) -> int:
        return self.nodes * self.accelerators_per_node

    def validate(self) -> None:
        if self.nodes < 1 or self.accelerators_per_node < 1:
            raise ValueError("nodes and accelerators_per_node must be >= 1")
        if self.device_memory_gib is not None and self.device_memory_gib <= 0:
            raise ValueError("device_memory_gib must be positive when provided")


@dataclass(frozen=True, slots=True)
class ModelScaleSpec:
    total_parameters: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    sequence_length: int
    micro_batch_size: int = 1

    def validate(self) -> None:
        for name in (
            "total_parameters",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "sequence_length",
            "micro_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    data_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1
    shard_model_state_across_data_parallel: bool = False

    @property
    def world_size(self) -> int:
        return (
            self.data_parallel
            * self.tensor_parallel
            * self.pipeline_parallel
            * self.context_parallel
            * self.expert_parallel
        )

    @property
    def model_state_shard_factor(self) -> int:
        factor = self.tensor_parallel * self.pipeline_parallel * self.expert_parallel
        if self.shard_model_state_across_data_parallel:
            factor *= self.data_parallel
        return factor

    @property
    def activation_shard_factor(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.context_parallel

    def validate(self) -> None:
        for name in (
            "data_parallel",
            "tensor_parallel",
            "pipeline_parallel",
            "context_parallel",
            "expert_parallel",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
