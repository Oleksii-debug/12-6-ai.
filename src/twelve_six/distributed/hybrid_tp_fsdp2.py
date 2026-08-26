"""2D Tensor Parallel + FSDP2 composition for dense 12-6 scale stages.

The canonical model is tensor-parallelized first on the TP submesh and then
FSDP2-sharded on the orthogonal data-parallel submesh, matching PyTorch's 2D
TP+FSDP composition contract.  This module is additive: single-device and
plain-FSDP2 paths remain unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.fsdp2_training import FSDP2Trainer, apply_fsdp2
from twelve_six.distributed.runtime import TorchMeshSpec, build_torch_mesh_spec
from twelve_six.distributed.tensor_parallel import (
    TensorParallelPlan,
    parallelize_decoder_tp,
)
from twelve_six.model import TwelveSixDecoder
from twelve_six.training.config import TrainerConfig


@dataclass(frozen=True, slots=True)
class HybridTPFSDP2Binding:
    """Runtime binding for one dense DP x TP composition."""

    parallel_plan: ParallelPlan
    mesh_spec: TorchMeshSpec
    tensor_parallel_plan: TensorParallelPlan
    full_mesh: Any
    data_parallel_mesh: Any
    tensor_parallel_mesh: Any
    data_parallel_group: Any
    tensor_parallel_group: Any

    @property
    def world_size(self) -> int:
        return self.parallel_plan.world_size

    @property
    def model_state_shard_factor(self) -> int:
        return self.parallel_plan.model_state_shard_factor


def _validate_hybrid_plan(plan: ParallelPlan) -> None:
    plan.validate()
    if plan.data_parallel < 2:
        raise ValueError("hybrid TP+FSDP2 requires data_parallel >= 2")
    if plan.tensor_parallel < 2:
        raise ValueError("hybrid TP+FSDP2 requires tensor_parallel >= 2")
    if plan.pipeline_parallel != 1:
        raise ValueError("hybrid TP+FSDP2 v1 does not compose pipeline parallelism")
    if plan.context_parallel != 1:
        raise ValueError("hybrid TP+FSDP2 v1 does not compose context parallelism")
    if plan.expert_parallel != 1:
        raise ValueError("hybrid TP+FSDP2 v1 is dense-only and rejects expert parallelism")
    if not plan.shard_model_state_across_data_parallel:
        raise ValueError("hybrid TP+FSDP2 requires model-state sharding across DP")


def apply_hybrid_tp_fsdp2(
    model: TwelveSixDecoder,
    plan: ParallelPlan,
    *,
    device_type: str,
    reshard_after_forward: bool = True,
) -> tuple[TwelveSixDecoder, HybridTPFSDP2Binding]:
    """Apply TP first, then FSDP2 on the orthogonal DP submesh.

    V1 deliberately uses full DP sharding (no HSDP replication dimension).  The
    same global ``ModelSpec`` remains authoritative while transformer-block
    parameters become TP DTensors and are then FSDP2-managed across DP ranks.
    """

    _validate_hybrid_plan(plan)
    mesh_spec = build_torch_mesh_spec(plan, fsdp_shard_degree=plan.data_parallel)
    if mesh_spec.fsdp_replicate_degree != 1:
        raise AssertionError("full DP sharding unexpectedly produced HSDP replication")

    full_mesh = mesh_spec.create_device_mesh(device_type)
    tp_mesh = mesh_spec.tensor_parallel_mesh(full_mesh)
    tp_plan = TensorParallelPlan.from_model_spec(model.spec, plan.tensor_parallel)
    parallelize_decoder_tp(model, tp_mesh, plan=tp_plan)

    # PyTorch's supported 2D composition applies TP on its 1D submesh first and
    # then fully_shard on the orthogonal DP submesh.  Do not use the full-SPMD
    # dp_mesh_dims path here: the incumbent TP seam produces DTensors on the TP
    # submesh rather than full-mesh DTensors with explicit DP Replicate placements.
    model = apply_fsdp2(
        model,
        **mesh_spec.fsdp2_kwargs(
            full_mesh,
            reshard_after_forward=reshard_after_forward,
        ),
    )
    dp_mesh = mesh_spec.fsdp2_data_parallel_mesh(full_mesh)
    if getattr(dp_mesh, "ndim", None) != 1:
        raise RuntimeError("hybrid TP+FSDP2 v1 requires a one-dimensional DP shard mesh")

    binding = HybridTPFSDP2Binding(
        parallel_plan=plan,
        mesh_spec=mesh_spec,
        tensor_parallel_plan=tp_plan,
        full_mesh=full_mesh,
        data_parallel_mesh=dp_mesh,
        tensor_parallel_mesh=tp_mesh,
        data_parallel_group=dp_mesh.get_group(),
        tensor_parallel_group=tp_mesh.get_group(),
    )
    return model, binding


class HybridTPFSDP2Trainer(FSDP2Trainer):
    """FSDP2 Trainer whose token-mean normalization is scoped to DP only.

    TP ranks are model shards, not independent data replicas.  Every TP rank in a
    replica sees the same logical microbatch, while FSDP2 averages gradients only
    across the data-parallel group.  Therefore token accounting for gradient
    normalization must use that same DP group, never the full DP x TP world.
    """

    def __init__(
        self,
        model: Any,
        config: TrainerConfig,
        *,
        device: Any,
        data_parallel_group: Any,
        data_parallel_degree: int,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> None:
        super().__init__(
            model,
            config,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("hybrid TP+FSDP2 Trainer requires initialized torch.distributed")
        if not isinstance(data_parallel_degree, int) or isinstance(data_parallel_degree, bool):
            raise TypeError("data_parallel_degree must be an integer")
        if data_parallel_degree < 2:
            raise ValueError("data_parallel_degree must be >= 2 for hybrid TP+FSDP2")
        observed = dist.get_world_size(group=data_parallel_group)
        if observed != data_parallel_degree:
            raise ValueError(
                "data-parallel process-group size differs from declared degree: "
                f"observed={observed}, declared={data_parallel_degree}"
            )
        self._data_parallel_group = data_parallel_group
        self._data_parallel_degree = data_parallel_degree
        self._last_data_parallel_global_tokens: int | None = None

    @property
    def data_parallel_degree(self) -> int:
        return self._data_parallel_degree

    @property
    def last_data_parallel_global_tokens(self) -> int | None:
        return self._last_data_parallel_global_tokens

    def _normalize_gradients_and_norm(self, token_count: int):
        if token_count <= 0:
            raise RuntimeError("optimizer update requires at least one valid target token")
        import torch
        import torch.distributed as dist

        token_tensor = torch.tensor(token_count, dtype=torch.int64, device=self.device)
        dist.all_reduce(
            token_tensor,
            op=dist.ReduceOp.SUM,
            group=self._data_parallel_group,
        )
        global_tokens = int(token_tensor.item())
        if global_tokens <= 0:
            raise RuntimeError("distributed optimizer update requires positive DP-global tokens")
        self._last_data_parallel_global_tokens = global_tokens

        found = False
        gradient_scale = self._data_parallel_degree / global_tokens
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            found = True
            parameter.grad.mul_(gradient_scale)

        if not found:
            return torch.zeros((), device=self.device)

        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=math.inf,
            error_if_nonfinite=True,
        )
        full_tensor = getattr(norm, "full_tensor", None)
        if callable(full_tensor):
            norm = full_tensor()
        if not isinstance(norm, torch.Tensor):
            norm = torch.as_tensor(norm, device=self.device)
        return norm.to(device=self.device)
