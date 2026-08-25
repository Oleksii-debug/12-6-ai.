"""Lazy PyTorch-native distributed adapter seam.

Importing this module does not import torch and does not initialize a process
group. S0 can therefore keep using the tiny single-device path unchanged.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .contracts import ParallelPlan


@dataclass(frozen=True, slots=True)
class TorchNativePlan:
    physical_mesh_dim_names: tuple[str, ...]
    physical_mesh_shape: tuple[int, ...]
    use_fsdp2: bool
    use_tensor_parallel: bool
    use_pipeline_parallel: bool
    use_context_parallel: bool
    requires_distributed_runtime: bool


@dataclass(frozen=True, slots=True)
class TorchNativeCapabilities:
    torch_version: str
    fsdp2_fully_shard: bool
    dtensor: bool
    tensor_parallel: bool
    distributed_checkpoint: bool


@dataclass(frozen=True, slots=True)
class ModuleStateMeasurement:
    parameter_count: int
    parameter_bytes: int
    gradient_bytes: int


def build_torch_native_plan(plan: ParallelPlan) -> TorchNativePlan:
    """Translate the project topology into a physical PyTorch mesh description.

    EP is absent from the physical mesh because the current 12-6 contract treats
    it as a DP subgroup. A backend with folded/orthogonal EP must use a different
    adapter and cannot silently reuse this mapping.
    """

    plan.validate()
    names = ("pp", "dp", "cp", "tp")
    shape = (
        plan.pipeline_parallel,
        plan.data_parallel,
        plan.context_parallel,
        plan.tensor_parallel,
    )
    return TorchNativePlan(
        physical_mesh_dim_names=names,
        physical_mesh_shape=shape,
        use_fsdp2=plan.shard_model_state_across_data_parallel and plan.data_parallel > 1,
        use_tensor_parallel=plan.tensor_parallel > 1,
        use_pipeline_parallel=plan.pipeline_parallel > 1,
        use_context_parallel=plan.context_parallel > 1,
        requires_distributed_runtime=plan.world_size > 1,
    )


def torch_native_capabilities() -> TorchNativeCapabilities:
    """Probe maintained PyTorch APIs lazily without mutating distributed state."""

    torch = importlib.import_module("torch")
    fsdp = importlib.import_module("torch.distributed.fsdp")
    tensor = importlib.import_module("torch.distributed.tensor")
    tensor_parallel = importlib.import_module("torch.distributed.tensor.parallel")
    dcp = importlib.import_module("torch.distributed.checkpoint")
    return TorchNativeCapabilities(
        torch_version=str(torch.__version__),
        fsdp2_fully_shard=callable(getattr(fsdp, "fully_shard", None)),
        dtensor=getattr(tensor, "DTensor", None) is not None,
        tensor_parallel=callable(getattr(tensor_parallel, "parallelize_module", None)),
        distributed_checkpoint=callable(getattr(dcp, "save", None))
        and callable(getattr(dcp, "load", None)),
    )


def measure_module_state(module: Any) -> ModuleStateMeasurement:
    """Measure materialized parameter/gradient storage without allocator inference."""

    if not hasattr(module, "parameters"):
        raise TypeError("module must provide parameters()")
    parameter_count = 0
    parameter_bytes = 0
    gradient_bytes = 0
    for parameter in module.parameters():
        parameter_count += int(parameter.numel())
        parameter_bytes += int(parameter.numel() * parameter.element_size())
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            gradient_bytes += int(gradient.numel() * gradient.element_size())
    return ModuleStateMeasurement(
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
    )
