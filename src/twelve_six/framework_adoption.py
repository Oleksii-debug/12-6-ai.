"""Stage-triggered framework adoption adapters for future distributed training.

This module is intentionally additive over the D12 runtime. It does not initialize
``torch.distributed`` and does not replace the project's PyTorch-native runtime.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Literal

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.runtime import TorchNativePlan, build_torch_native_plan
from twelve_six.model import InitSpec, ModelSpec

FrameworkName = Literal["pytorch_native", "torchtitan", "olmo_core", "megatron_core"]


@dataclass(frozen=True, slots=True)
class FrameworkAssessment:
    name: FrameworkName
    selected_path: bool
    role: str
    model_integration: str
    data_boundary: str
    checkpoint_boundary: str
    parallelism: str
    moe_path: str
    fault_recovery: str
    logging: str
    dependency_burden: str
    maintenance_cost: str


@dataclass(frozen=True, slots=True)
class NativeIntegrationPlan:
    model_spec_sha256: str
    init_spec_sha256: str
    topology: TorchNativePlan
    simple_tp_compatible: bool
    simple_pp_compatible: bool
    data_owner: str
    checkpoint_owner: str
    initialization_mode: str
    model_rewrite_required: bool


@dataclass(frozen=True, slots=True)
class MegatronCorePlan:
    model_spec_sha256: str
    init_spec_sha256: str
    transformer_config: dict[str, object]
    model_config: dict[str, object]
    parallel_config: dict[str, int]
    output_layer_init_std: float
    state_dict_adapter_required: bool
    topology_v2_required_for_moe: bool
    runtime_validation_required: bool


@dataclass(frozen=True, slots=True)
class AdoptionSignals:
    native_fits_memory: bool = True
    native_runtime_complete: bool = True
    nvidia_cluster: bool = False
    measured_megatron_speedup: float | None = None
    minimum_migration_speedup: float = 1.15
    megatron_runtime_validated: bool = False

    def validate(self) -> None:
        if self.measured_megatron_speedup is not None and self.measured_megatron_speedup <= 0:
            raise ValueError("measured_megatron_speedup must be positive when provided")
        if self.minimum_migration_speedup <= 1.0:
            raise ValueError("minimum_migration_speedup must be greater than 1.0")


@dataclass(frozen=True, slots=True)
class FrameworkDecision:
    incumbent: FrameworkName
    benchmark_alternative: FrameworkName | None
    migration_authorized: bool
    trigger: str


def framework_assessments() -> tuple[FrameworkAssessment, ...]:
    """Return the D20 decision matrix in a machine-readable form."""

    return (
        FrameworkAssessment(
            name="pytorch_native",
            selected_path=True,
            role="incumbent multi-GPU runtime",
            model_integration="wrap existing TwelveSixDecoder; no model port",
            data_boundary="project D03/D04 pipeline remains authoritative",
            checkpoint_boundary="D05 logical identity plus DCP physical shards",
            parallelism="FSDP2 first; DTensor TP; PP/CP only behind evidence gates",
            moe_path="defer until expert topology is modeled explicitly",
            fault_recovery="torchrun elastic restart plus DCP resume",
            logging="project metrics plus PyTorch logging/profiler seams",
            dependency_burden="lowest; already a project dependency",
            maintenance_cost="lowest while only a few native scale features are composed",
        ),
        FrameworkAssessment(
            name="megatron_core",
            selected_path=True,
            role="measured scale escape hatch",
            model_integration="ModelSpec-to-Megatron config and state-dict adapter",
            data_boundary="keep project data boundary; adapt batches at trainer edge",
            checkpoint_boundary="Megatron torch_dist bridge plus D05 logical identity",
            parallelism="mature TP/PP/CP and distributed optimizer",
            moe_path="strong EP/EDP path, but requires project topology v2",
            fault_recovery="checkpoint restart; optional NVIDIA resiliency stack",
            logging="Megatron metrics/profiling with project identity sidecar",
            dependency_burden="high; separate Python/CUDA/NVIDIA runtime profile",
            maintenance_cost="worthwhile only after a measured native bottleneck",
        ),
        FrameworkAssessment(
            name="torchtitan",
            selected_path=False,
            role="reference implementation and benchmark source",
            model_integration="requires TorchTitan ModelSpec/config/model integration port",
            data_boundary="trainer-owned dataloader/config integration",
            checkpoint_boundary="excellent DCP checkpoint manager and seed checkpoints",
            parallelism="strong PyTorch-native FSDP/TP/PP/CP composition",
            moe_path="active EP/TP support",
            fault_recovery="checkpoint manager plus launcher/runtime recovery",
            logging="strong built-in metrics, profiling and debugging",
            dependency_burden="medium-high; release/nightly coupling to PyTorch stack",
            maintenance_cost="duplicates D12 orchestration unless native glue grows materially",
        ),
        FrameworkAssessment(
            name="olmo_core",
            selected_path=False,
            role="not selected for current 12-6 lifecycle",
            model_integration="OLMo Transformer/TrainModule lifecycle adoption",
            data_boundary="OLMo composable data pipeline tends to become authoritative",
            checkpoint_boundary="integrated distributed checkpoint stack",
            parallelism="broad DP/TP/PP/CP/EP support",
            moe_path="mature integrated MoE path",
            fault_recovery="trainer/checkpointer orchestration",
            logging="trainer callbacks and integrated metrics",
            dependency_burden="medium-high with optional compiled GPU components",
            maintenance_cost="highest collision with existing D02/D03/D05/D12 ownership",
        ),
    )


def _simple_tp_compatible(spec: ModelSpec, degree: int) -> bool:
    if degree == 1:
        return True
    return all(
        value % degree == 0
        for value in (spec.d_model, spec.n_heads, spec.n_kv_heads, spec.d_ff)
    )


def _simple_pp_compatible(spec: ModelSpec, degree: int) -> bool:
    return degree == 1 or (spec.n_layers >= degree and spec.n_layers % degree == 0)


def build_native_integration_plan(
    spec: ModelSpec,
    init_spec: InitSpec,
    plan: ParallelPlan,
) -> NativeIntegrationPlan:
    """Preserve the project model and translate only the distributed execution plan."""

    native = build_torch_native_plan(plan)
    return NativeIntegrationPlan(
        model_spec_sha256=spec.identity_sha256(),
        init_spec_sha256=init_spec.identity_sha256(),
        topology=native,
        simple_tp_compatible=_simple_tp_compatible(spec, plan.tensor_parallel),
        simple_pp_compatible=_simple_pp_compatible(spec, plan.pipeline_parallel),
        data_owner="12-6",
        checkpoint_owner="D05_LOGICAL_IDENTITY_PLUS_PYTORCH_DCP_LAYOUT",
        initialization_mode="PROJECT_SEED_CHECKPOINT_BEFORE_TOPOLOGY_DEPENDENT_SHARDING",
        model_rewrite_required=False,
    )


def _validate_megatron_dense_representability(spec: ModelSpec, init_spec: InitSpec) -> None:
    if spec.activation != "swiglu":
        raise ValueError("Megatron adapter v1 requires SwiGLU")
    if spec.norm_kind != "rmsnorm" or spec.norm_placement != "pre":
        raise ValueError("Megatron adapter v1 requires pre-RMSNorm")
    if spec.position_embedding != "rope":
        raise ValueError("Megatron adapter v1 requires RoPE")
    if spec.q_dim != spec.d_model:
        raise ValueError("Megatron adapter v1 requires n_heads * head_dim == d_model")
    if spec.attention_bias or spec.mlp_bias or spec.lm_head_bias:
        raise ValueError("Megatron adapter v1 does not map projection/readout biases")
    if not spec.final_norm:
        raise ValueError("Megatron adapter v1 requires final RMSNorm")
    if init_spec.family != "normal":
        raise ValueError("Megatron adapter v1 requires normal initialization")
    if init_spec.residual_branch_scale != "sqrt_2_layers":
        raise ValueError("Megatron adapter v1 requires sqrt_2_layers residual scaling")


def build_megatron_core_plan(
    spec: ModelSpec,
    init_spec: InitSpec,
    plan: ParallelPlan,
) -> MegatronCorePlan:
    """Build a dependency-free dense Megatron Core translation prototype.

    D12 expert parallelism is a subgroup of project DP, while Megatron supports an
    independent/foldable expert dimension. This v1 adapter therefore fails closed
    for EP > 1 instead of silently changing physical world-size semantics.
    """

    plan.validate()
    _validate_megatron_dense_representability(spec, init_spec)
    if plan.expert_parallel != 1:
        raise ValueError("Megatron MoE requires project topology v2; D12 EP cannot map directly")

    transformer_config: dict[str, object] = {
        "num_layers": spec.n_layers,
        "hidden_size": spec.d_model,
        "num_attention_heads": spec.n_heads,
        "num_query_groups": spec.n_kv_heads,
        "kv_channels": spec.head_dim,
        "ffn_hidden_size": spec.d_ff,
        "normalization": "RMSNorm",
        "layernorm_epsilon": spec.norm_eps,
        "attention_dropout": spec.attention_dropout,
        "hidden_dropout": 0.0,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "gated_linear_unit": True,
        "activation_func": "torch.nn.functional.silu",
        "rotary_interleaved": True,
        "rotary_base": spec.rope_theta,
        "rotary_percent": spec.rope_rotary_dim / spec.head_dim,
        "init_method_std": init_spec.std,
    }
    model_config: dict[str, object] = {
        "vocab_size": spec.vocab_size,
        "max_sequence_length": spec.max_seq_len,
        "position_embedding_type": "rope",
        "share_embeddings_and_output_weights": spec.tie_word_embeddings,
    }
    parallel_config = {
        "world_size": plan.world_size,
        "data_parallel_size": plan.data_parallel,
        "tensor_model_parallel_size": plan.tensor_parallel,
        "pipeline_model_parallel_size": plan.pipeline_parallel,
        "context_parallel_size": plan.context_parallel,
        "expert_model_parallel_size": 1,
    }
    return MegatronCorePlan(
        model_spec_sha256=spec.identity_sha256(),
        init_spec_sha256=init_spec.identity_sha256(),
        transformer_config=transformer_config,
        model_config=model_config,
        parallel_config=parallel_config,
        output_layer_init_std=init_spec.residual_std(spec.n_layers),
        state_dict_adapter_required=True,
        topology_v2_required_for_moe=True,
        runtime_validation_required=True,
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def optional_framework_availability() -> dict[str, bool]:
    """Probe optional framework imports without importing or initializing them."""

    return {
        "torchtitan": _module_available("torchtitan"),
        "olmo_core": _module_available("olmo_core"),
        "megatron_core": _module_available("megatron.core"),
    }


def recommend_framework(plan: ParallelPlan, signals: AdoptionSignals) -> FrameworkDecision:
    """Choose by measured trigger, not by parameter count alone."""

    plan.validate()
    signals.validate()
    if plan.world_size == 1:
        return FrameworkDecision(
            incumbent="pytorch_native",
            benchmark_alternative=None,
            migration_authorized=False,
            trigger="single-device path; no framework migration is justified",
        )

    complex_axes = plan.pipeline_parallel > 1 or plan.context_parallel > 1
    if plan.expert_parallel > 1:
        return FrameworkDecision(
            incumbent="pytorch_native",
            benchmark_alternative="megatron_core" if signals.nvidia_cluster else None,
            migration_authorized=False,
            trigger="MoE requires project topology v2 before any Megatron migration",
        )

    if not signals.native_fits_memory or not signals.native_runtime_complete:
        if signals.nvidia_cluster and signals.megatron_runtime_validated:
            return FrameworkDecision(
                incumbent="megatron_core",
                benchmark_alternative="pytorch_native",
                migration_authorized=True,
                trigger="validated Megatron path closes a measured native fit/runtime gap",
            )
        return FrameworkDecision(
            incumbent="pytorch_native",
            benchmark_alternative="megatron_core" if signals.nvidia_cluster else None,
            migration_authorized=False,
            trigger="native gap exists; Megatron must first pass its runtime acceptance gate",
        )

    speedup = signals.measured_megatron_speedup
    if (
        signals.nvidia_cluster
        and signals.megatron_runtime_validated
        and speedup is not None
        and speedup >= signals.minimum_migration_speedup
    ):
        return FrameworkDecision(
            incumbent="megatron_core",
            benchmark_alternative="pytorch_native",
            migration_authorized=True,
            trigger="measured Megatron speedup crosses the configured migration threshold",
        )

    alternative: FrameworkName | None = None
    if signals.nvidia_cluster and complex_axes:
        alternative = "megatron_core"
    return FrameworkDecision(
        incumbent="pytorch_native",
        benchmark_alternative=alternative,
        migration_authorized=False,
        trigger="preserve native incumbent until a measured fit or throughput trigger is crossed",
    )


def decision_payload(
    spec: ModelSpec,
    init_spec: InitSpec,
    plan: ParallelPlan,
    signals: AdoptionSignals,
) -> dict[str, object]:
    """Return a JSON-ready experiment payload for ADR/evidence tooling."""

    native = build_native_integration_plan(spec, init_spec, plan)
    decision = recommend_framework(plan, signals)
    payload: dict[str, object] = {
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init_spec.identity_sha256(),
        "parallel_plan": asdict(plan),
        "native": asdict(native),
        "decision": asdict(decision),
        "optional_frameworks": optional_framework_availability(),
    }
    if plan.expert_parallel == 1:
        payload["megatron_core"] = asdict(build_megatron_core_plan(spec, init_spec, plan))
    else:
        payload["megatron_core"] = {
            "status": "BLOCKED_REQUIRES_PROJECT_TOPOLOGY_V2_FOR_MOE",
        }
    return payload
