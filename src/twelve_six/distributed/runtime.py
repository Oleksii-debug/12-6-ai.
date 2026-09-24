"""Lazy PyTorch distributed adapter contracts that keep single-device S0 dependency-light."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from .contracts import ParallelPlan
from .rank_layout import RankLayout

_MESH_DIM_NAMES = ("dp_replicate", "dp_shard", "cp", "pp", "tp")


@dataclass(frozen=True, slots=True)
class TorchMeshSpec:
    """Backend translation of ParallelPlan without creating a process group or DeviceMesh."""

    shape: tuple[int, int, int, int, int]
    dim_names: tuple[str, str, str, str, str]
    logical_layout_sha256: str
    fsdp_shard_degree: int
    fsdp_replicate_degree: int
    expert_parallel_degree: int
    expert_data_parallel_degree: int

    @property
    def world_size(self) -> int:
        return prod(self.shape)

    @property
    def megatron_axis_degrees(self) -> dict[str, int]:
        """Translate project DP-total semantics to Megatron EP x expert-DP semantics."""

        values = {
            "tp": self.shape[self.dim_names.index("tp")],
            "pp": self.shape[self.dim_names.index("pp")],
            "cp": self.shape[self.dim_names.index("cp")],
            "ep": self.expert_parallel_degree,
            "dp": self.expert_data_parallel_degree,
        }
        if prod(values.values()) != self.world_size:
            raise AssertionError("Megatron axis translation changed physical world size")
        return values

    @property
    def fsdp2_enabled(self) -> bool:
        return self.fsdp_shard_degree > 1 or self.fsdp_replicate_degree > 1

    @property
    def tensor_parallel_enabled(self) -> bool:
        return self.shape[self.dim_names.index("tp")] > 1

    @property
    def pipeline_parallel_enabled(self) -> bool:
        return self.shape[self.dim_names.index("pp")] > 1

    @property
    def context_parallel_enabled(self) -> bool:
        return self.shape[self.dim_names.index("cp")] > 1

    def create_device_mesh(self, device_type: str) -> Any:
        """Create the PyTorch DeviceMesh lazily; caller owns distributed initialization."""

        if not isinstance(device_type, str) or not device_type.strip():
            raise ValueError("device_type must be a non-empty string")
        try:
            from torch.distributed.device_mesh import init_device_mesh
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("PyTorch DeviceMesh is unavailable in this environment") from exc
        return init_device_mesh(device_type, self.shape, mesh_dim_names=self.dim_names)

    def fsdp2_data_parallel_mesh(self, device_mesh: Any) -> Any:
        """Return the 1D FSDP or 2D HSDP submesh for plain Tensor parameters.

        Canonical 12-6 model parameters are ordinary Tensors before ``fully_shard``.
        PyTorch's ``dp_mesh_dims`` path is for parameters that are already DTensors on
        a full SPMD mesh, so the plain-model path must pass only the data-parallel mesh.
        """

        if not self.fsdp2_enabled:
            raise ValueError("FSDP2 is disabled because data_parallel degree is 1")
        if self.expert_parallel_degree > 1:
            raise ValueError(
                "generic FSDP2 binding refuses EP>1; expert parameters need backend-specific groups"
            )
        if self.fsdp_replicate_degree > 1:
            return device_mesh[("dp_replicate", "dp_shard")]
        return device_mesh["dp_shard"]

    def fsdp2_kwargs(
        self, device_mesh: Any, *, reshard_after_forward: bool = True
    ) -> dict[str, Any]:
        """Return ``fully_shard`` kwargs for canonical plain-Tensor 12-6 modules."""

        return {
            "mesh": self.fsdp2_data_parallel_mesh(device_mesh),
            "reshard_after_forward": reshard_after_forward,
        }

    def fsdp2_spmd_kwargs(
        self, device_mesh: Any, *, reshard_after_forward: bool = True
    ) -> dict[str, Any]:
        """Return full-SPMD kwargs only for parameters already represented as DTensors."""

        if not self.fsdp2_enabled:
            raise ValueError("FSDP2 is disabled because data_parallel degree is 1")
        if self.expert_parallel_degree > 1:
            raise ValueError(
                "generic FSDP2 binding refuses EP>1; expert parameters need backend-specific groups"
            )
        try:
            from torch.distributed.fsdp import DataParallelMeshDims
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("PyTorch FSDP2 DataParallelMeshDims is unavailable") from exc
        shard: str | None = "dp_shard" if self.fsdp_shard_degree > 1 else None
        replicate: str | None = "dp_replicate" if self.fsdp_replicate_degree > 1 else None
        return {
            "mesh": device_mesh,
            "dp_mesh_dims": DataParallelMeshDims(shard=shard, replicate=replicate),
            "reshard_after_forward": reshard_after_forward,
        }

    def tensor_parallel_mesh(self, device_mesh: Any) -> Any:
        if not self.tensor_parallel_enabled:
            raise ValueError("tensor parallelism is disabled")
        return device_mesh["tp"]

    def pipeline_parallel_mesh(self, device_mesh: Any) -> Any:
        if not self.pipeline_parallel_enabled:
            raise ValueError("pipeline parallelism is disabled")
        return device_mesh["pp"]

    def context_parallel_mesh(self, device_mesh: Any) -> Any:
        if not self.context_parallel_enabled:
            raise ValueError("context parallelism is disabled")
        return device_mesh["cp"]


@dataclass(frozen=True, slots=True)
class BackendAdoptionDecision:
    backend: str
    status: str
    reasons: tuple[str, ...]


def build_torch_mesh_spec(
    plan: ParallelPlan,
    *,
    fsdp_shard_degree: int = 1,
) -> TorchMeshSpec:
    """Split existing DP into HSDP replicate/shard dims; EP remains a DP subgroup."""

    plan.validate()
    if (
        not isinstance(fsdp_shard_degree, int)
        or isinstance(fsdp_shard_degree, bool)
        or fsdp_shard_degree < 1
    ):
        raise ValueError("fsdp_shard_degree must be a positive integer")
    if plan.data_parallel % fsdp_shard_degree != 0:
        raise ValueError("fsdp_shard_degree must divide data_parallel")
    replicate_degree = plan.data_parallel // fsdp_shard_degree
    layout = RankLayout(plan)
    shape = (
        replicate_degree,
        fsdp_shard_degree,
        plan.context_parallel,
        plan.pipeline_parallel,
        plan.tensor_parallel,
    )
    if prod(shape) != plan.world_size:
        raise AssertionError("Torch mesh shape changed physical world size")
    return TorchMeshSpec(
        shape=shape,
        dim_names=_MESH_DIM_NAMES,
        logical_layout_sha256=layout.identity_sha256,
        fsdp_shard_degree=fsdp_shard_degree,
        fsdp_replicate_degree=replicate_degree,
        expert_parallel_degree=plan.expert_parallel,
        expert_data_parallel_degree=plan.expert_data_parallel,
    )


def choose_backend(
    *,
    requires_moe: bool,
    requires_pipeline_parallel: bool,
    requires_context_parallel: bool,
    requires_float8: bool,
    requires_distributed_checkpoint_reshard: bool,
    nvidia_only_scale_target: bool,
) -> BackendAdoptionDecision:
    """Conservative stage-triggered recommendation, not an execution or promotion decision."""

    if requires_moe or (requires_pipeline_parallel and nvidia_only_scale_target):
        return BackendAdoptionDecision(
            backend="megatron-core",
            status="evaluate-first",
            reasons=(
                "PP/CP/EP/MoE and NVIDIA-scale kernels are first-class Megatron Core surfaces",
                "retain 12-6 semantic ModelSpec/checkpoint identities above the backend",
            ),
        )
    if requires_context_parallel or requires_float8:
        return BackendAdoptionDecision(
            backend="torchtitan",
            status="evaluate-first",
            reasons=(
                "TorchTitan composes PyTorch DP/TP/PP/CP and current float8/debug tooling",
                "use it only after native 12-6 interfaces prove insufficient at the measured stage",
            ),
        )
    if requires_distributed_checkpoint_reshard:
        return BackendAdoptionDecision(
            backend="native-pytorch",
            status="adopt-dcp-fsdp2",
            reasons=(
                "PyTorch DCP supports topology-aware load-time resharding",
                "FSDP2/DTensor keeps the adapter close to existing 12-6 model/trainer contracts",
            ),
        )
    if requires_pipeline_parallel:
        return BackendAdoptionDecision(
            backend="olmo-core-or-native-pytorch",
            status="benchmark-before-adoption",
            reasons=(
                "OLMo-core offers composable maintained training/distributed components",
                "native PyTorch remains viable until operational complexity justifies a framework",
            ),
        )
    return BackendAdoptionDecision(
        backend="single-device-or-native-pytorch",
        status="stay-simple",
        reasons=(
            "do not introduce distributed runtime complexity before measured need",
            "S0 remains single-device and unaffected by this adapter package",
        ),
    )
