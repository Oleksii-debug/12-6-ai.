"""PyTorch-native tensor-parallel planning and execution seam for future 12-6 stages.

The planning surface is torch-free. Runtime imports are lazy so canonical S0 does not initialize or
require a distributed process group. The first seam shards transformer-block attention and SwiGLU
weights only; token embeddings, final norm, and LM head remain replicated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from .rank_layout import RankLayout

if TYPE_CHECKING:
    from twelve_six.model import ModelSpec, TwelveSixDecoder

_TP_SCHEMA = "12-6.tensor-parallel-plan.v1"
_TP_CHECKPOINT_SCHEMA = "12-6.tensor-parallel-checkpoint-layout.v1"

ShardStyle = Literal["colwise", "rowwise", "replicate"]


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TensorPartitionRule:
    """Global-to-local partition rule for one canonical transformer-block parameter."""

    parameter: str
    style: ShardStyle
    shard_dim: int | None
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TensorParallelRankGeometry:
    """Head-aligned local geometry owned by one rank inside a TP group."""

    tp_rank: int
    query_head_start: int
    query_head_stop: int
    kv_head_start: int
    kv_head_stop: int
    ffn_start: int
    ffn_stop: int
    local_q_dim: int
    local_kv_dim: int
    local_d_ff: int

    @property
    def local_query_heads(self) -> int:
        return self.query_head_stop - self.query_head_start

    @property
    def local_kv_heads(self) -> int:
        return self.kv_head_stop - self.kv_head_start


@dataclass(frozen=True, slots=True)
class TensorParallelPlan:
    """ModelSpec-bound TP plan that keeps GQA heads intact on every rank."""

    model_spec_sha256: str
    tp_degree: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    q_dim: int
    kv_dim: int
    gqa_group_size: int
    schema: str = _TP_SCHEMA

    @classmethod
    def from_model_spec(cls, spec: ModelSpec, tp_degree: int) -> TensorParallelPlan:
        degree = _positive_int(tp_degree, "tp_degree")
        if spec.n_heads % degree != 0:
            raise ValueError("tp_degree must divide n_heads for head-aligned query sharding")
        if spec.n_kv_heads % degree != 0:
            raise ValueError("tp_degree must divide n_kv_heads for head-aligned GQA sharding")
        if spec.d_ff % degree != 0:
            raise ValueError("tp_degree must divide d_ff for even SwiGLU sharding")
        return cls(
            model_spec_sha256=spec.identity_sha256(),
            tp_degree=degree,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            n_kv_heads=spec.n_kv_heads,
            head_dim=spec.head_dim,
            d_ff=spec.d_ff,
            q_dim=spec.q_dim,
            kv_dim=spec.kv_dim,
            gqa_group_size=spec.n_heads // spec.n_kv_heads,
        )

    @property
    def enabled(self) -> bool:
        return self.tp_degree > 1

    @property
    def identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["partition_policy"] = {
            "attention_qkv": "colwise-head-aligned",
            "attention_output": "rowwise-reduce-to-replicated",
            "mlp_gate_up": "colwise-ffn-aligned",
            "mlp_down": "rowwise-reduce-to-replicated",
            "embedding": "replicated-v1",
            "lm_head": "replicated-v1",
        }
        return payload

    @property
    def identity_sha256(self) -> str:
        return _hash_json(self.identity_payload)

    def rank_geometry(self, tp_rank: int) -> TensorParallelRankGeometry:
        if not isinstance(tp_rank, int) or isinstance(tp_rank, bool):
            raise TypeError("tp_rank must be an integer")
        if not 0 <= tp_rank < self.tp_degree:
            raise ValueError(f"tp_rank must be in [0, {self.tp_degree})")
        query_heads = self.n_heads // self.tp_degree
        kv_heads = self.n_kv_heads // self.tp_degree
        local_ff = self.d_ff // self.tp_degree
        return TensorParallelRankGeometry(
            tp_rank=tp_rank,
            query_head_start=tp_rank * query_heads,
            query_head_stop=(tp_rank + 1) * query_heads,
            kv_head_start=tp_rank * kv_heads,
            kv_head_stop=(tp_rank + 1) * kv_heads,
            ffn_start=tp_rank * local_ff,
            ffn_stop=(tp_rank + 1) * local_ff,
            local_q_dim=self.q_dim // self.tp_degree,
            local_kv_dim=self.kv_dim // self.tp_degree,
            local_d_ff=local_ff,
        )

    def rank_geometry_for_global_rank(
        self,
        layout: RankLayout,
        global_rank: int,
    ) -> TensorParallelRankGeometry:
        if layout.plan.tensor_parallel != self.tp_degree:
            raise ValueError("RankLayout tensor_parallel degree differs from TensorParallelPlan")
        return self.rank_geometry(layout.coordinate(global_rank).tp)

    def parameter_partitions(self) -> tuple[TensorPartitionRule, ...]:
        local_q = self.q_dim // self.tp_degree
        local_kv = self.kv_dim // self.tp_degree
        local_ff = self.d_ff // self.tp_degree
        return (
            TensorPartitionRule(
                "attn.q_proj.weight",
                "colwise",
                0,
                (self.q_dim, self.d_model),
                (local_q, self.d_model),
            ),
            TensorPartitionRule(
                "attn.k_proj.weight",
                "colwise",
                0,
                (self.kv_dim, self.d_model),
                (local_kv, self.d_model),
            ),
            TensorPartitionRule(
                "attn.v_proj.weight",
                "colwise",
                0,
                (self.kv_dim, self.d_model),
                (local_kv, self.d_model),
            ),
            TensorPartitionRule(
                "attn.out_proj.weight",
                "rowwise",
                1,
                (self.d_model, self.q_dim),
                (self.d_model, local_q),
            ),
            TensorPartitionRule(
                "mlp.gate_proj.weight",
                "colwise",
                0,
                (self.d_ff, self.d_model),
                (local_ff, self.d_model),
            ),
            TensorPartitionRule(
                "mlp.up_proj.weight",
                "colwise",
                0,
                (self.d_ff, self.d_model),
                (local_ff, self.d_model),
            ),
            TensorPartitionRule(
                "mlp.down_proj.weight",
                "rowwise",
                1,
                (self.d_model, self.d_ff),
                (self.d_model, local_ff),
            ),
        )

    @property
    def checkpoint_layout_payload(self) -> dict[str, Any]:
        """Topology identity layered above, not substituted for, canonical D05 identity."""

        return {
            "schema": _TP_CHECKPOINT_SCHEMA,
            "model_spec_sha256": self.model_spec_sha256,
            "tensor_parallel_plan_sha256": self.identity_sha256,
            "tensor_parallel_degree": self.tp_degree,
            "canonical_fqns_unchanged": True,
            "canonical_global_shapes_unchanged": True,
            "d05_semantic_model_identity_unchanged": True,
            "physical_layout_identity_is_topology_specific": True,
            "sharded_checkpoint_backend": "torch.distributed.checkpoint",
        }

    @property
    def checkpoint_layout_sha256(self) -> str:
        return _hash_json(self.checkpoint_layout_payload)


@dataclass(frozen=True, slots=True)
class MegatronCoreTPAdapter:
    """Dependency-free boundary for a future Megatron Core backend adapter."""

    model_spec_sha256: str
    tensor_parallel_plan_sha256: str
    tensor_model_parallel_size: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    ffn_hidden_size: int
    qkv_fusion_required: bool = True
    swiglu_fc1_fusion_required: bool = True
    canonical_checkpoint_parent_required: bool = True

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "backend": "megatron-core",
            "status": "adapter-boundary-only-not-runtime-tested",
            **asdict(self),
            "translation": {
                "q_k_v": "fuse canonical q_proj/k_proj/v_proj at adapter boundary",
                "attention_output": "canonical out_proj -> Megatron attention output projection",
                "gate_up": "fuse canonical SwiGLU gate_proj/up_proj at adapter boundary",
                "mlp_down": "canonical down_proj -> Megatron second MLP projection",
            },
        }

    @property
    def identity_sha256(self) -> str:
        return _hash_json(self.payload)


def build_megatron_core_tp_adapter(plan: TensorParallelPlan) -> MegatronCoreTPAdapter:
    return MegatronCoreTPAdapter(
        model_spec_sha256=plan.model_spec_sha256,
        tensor_parallel_plan_sha256=plan.identity_sha256,
        tensor_model_parallel_size=plan.tp_degree,
        hidden_size=plan.d_model,
        num_attention_heads=plan.n_heads,
        num_query_groups=plan.n_kv_heads,
        kv_channels=plan.head_dim,
        ffn_hidden_size=plan.d_ff,
    )


def _parallel_styles() -> dict[str, Any]:
    try:
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyTorch native tensor parallel APIs are unavailable") from exc
    return {
        "attn.q_proj": ColwiseParallel(),
        "attn.k_proj": ColwiseParallel(),
        "attn.v_proj": ColwiseParallel(),
        "attn.out_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }


def _mesh_size(tp_mesh: Any) -> int:
    try:
        ndim = tp_mesh.ndim
        size = tp_mesh.size()
    except (AttributeError, TypeError) as exc:
        raise TypeError("tp_mesh must be a one-dimensional PyTorch DeviceMesh") from exc
    if ndim != 1:
        raise ValueError("PyTorch tensor parallel requires a one-dimensional TP DeviceMesh")
    return int(size)


def parallelize_decoder_tp(
    model: TwelveSixDecoder,
    tp_mesh: Any,
    *,
    plan: TensorParallelPlan | None = None,
    src_data_rank: int | None = 0,
) -> TensorParallelPlan:
    """Apply block-local PyTorch DTensor TP while keeping decoder boundaries replicated.

    The model's ModelSpec remains global. Attention's derived runtime dimensions are localized after
    ColwiseParallel shards Q/K/V so the existing reshape/GQA implementation operates on whole local
    heads. RowwiseParallel restores replicated residual-stream outputs after attention and MLP.
    """

    mesh_degree = _mesh_size(tp_mesh)
    resolved = TensorParallelPlan.from_model_spec(model.spec, mesh_degree) if plan is None else plan
    if resolved.tp_degree != mesh_degree:
        raise ValueError("TensorParallelPlan degree differs from TP DeviceMesh size")
    if resolved.model_spec_sha256 != model.spec.identity_sha256():
        raise ValueError("TensorParallelPlan is bound to a different ModelSpec")
    if not resolved.enabled:
        raise ValueError("tensor parallel execution requires tp_degree > 1")
    if getattr(model, "_twelve_six_tp_plan_sha256", None) is not None:
        raise RuntimeError("decoder is already tensor-parallelized")

    try:
        from torch.distributed.tensor.parallel import parallelize_module
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyTorch native tensor parallel APIs are unavailable") from exc

    for block in model.blocks:
        parallelize_module(
            block,
            tp_mesh,
            _parallel_styles(),
            src_data_rank=src_data_rank,
        )
        block.attn.n_heads //= resolved.tp_degree
        block.attn.n_kv_heads //= resolved.tp_degree
        block.attn.q_dim //= resolved.tp_degree
        block.attn.kv_dim //= resolved.tp_degree

    model._twelve_six_tp_plan_sha256 = resolved.identity_sha256
    return resolved
