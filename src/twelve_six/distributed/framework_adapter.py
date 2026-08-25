"""Framework-adoption seam for future 12-6 scale training.

This module keeps 12-6 semantic identities authoritative and decides which
training-framework orchestration can be delegated without changing model or
weight provenance. Importing it does not initialize torch.distributed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Literal

from ..model import InitSpec, ModelSpec, TwelveSixDecoder
from ..training.config import TrainerConfig
from .contracts import ParallelPlan
from .runtime import TorchNativePlan, build_torch_native_plan, torch_native_capabilities

FrameworkBackend = Literal["auto", "pytorch_native", "torchtitan"]

_REQUIRED_TORCHTITAN_COMPONENTS = (
    ("torchtitan.trainer", "Trainer"),
    ("torchtitan.protocols.model", "BaseModel"),
    ("torchtitan.components.optimizer", "OptimizersContainer"),
    ("torchtitan.components.checkpoint", "CheckpointManager"),
    ("torchtitan.components.metrics", "MetricsProcessor"),
)


@dataclass(frozen=True, slots=True)
class TorchTitanProbe:
    installed: bool
    version: str | None
    available_components: tuple[str, ...]
    missing_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TorchTitanFit:
    probe: TorchTitanProbe
    direct_adoption_ready: bool
    blockers: tuple[str, ...]
    delegable_complexity: tuple[str, ...]
    retained_project_contracts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelRegistrationPlan:
    model_class: str
    model_spec_identity_sha256: str
    init_spec_identity_sha256: str
    parameter_count: int
    weight_origin: str
    registration_backend: str


@dataclass(frozen=True, slots=True)
class DatasetBoundaryPlan:
    batch_contract: str
    accepted_target_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    effective_data_parallel_degree: int
    peer_batch_policy: str
    data_identity_owner: str


@dataclass(frozen=True, slots=True)
class OptimizerBoundaryPlan:
    name: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    source_of_truth: str
    torchtitan_mapping: str


@dataclass(frozen=True, slots=True)
class CheckpointBoundaryPlan:
    semantic_identity_owner: str
    scale_storage: str
    framework_state: tuple[str, ...]
    topology_identity_owner: str


@dataclass(frozen=True, slots=True)
class LoggingBoundaryPlan:
    source_schema: str
    fields: tuple[str, ...]
    framework_sink: str


@dataclass(frozen=True, slots=True)
class ScaleFrameworkPlan:
    selected_backend: str
    model: ModelRegistrationPlan
    dataset: DatasetBoundaryPlan
    optimizer: OptimizerBoundaryPlan
    distributed: TorchNativePlan
    checkpoint: CheckpointBoundaryPlan
    logging: LoggingBoundaryPlan
    torchtitan: TorchTitanFit

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_torchtitan() -> TorchTitanProbe:
    """Inspect the installed TorchTitan surface without initializing a process group."""

    if importlib.util.find_spec("torchtitan") is None:
        return TorchTitanProbe(
            installed=False,
            version=None,
            available_components=(),
            missing_components=tuple(
                f"{module}:{name}" for module, name in _REQUIRED_TORCHTITAN_COMPONENTS
            ),
        )

    available: list[str] = []
    missing: list[str] = []
    for module_name, attribute in _REQUIRED_TORCHTITAN_COMPONENTS:
        label = f"{module_name}:{attribute}"
        try:
            module = importlib.import_module(module_name)
        except Exception:  # probe must report capability, not mask Product execution failures
            missing.append(label)
            continue
        if getattr(module, attribute, None) is None:
            missing.append(label)
        else:
            available.append(label)

    try:
        version = metadata.version("torchtitan")
    except metadata.PackageNotFoundError:
        version = None

    return TorchTitanProbe(
        installed=True,
        version=version,
        available_components=tuple(available),
        missing_components=tuple(missing),
    )


def assess_torchtitan_fit(
    model_class: type = TwelveSixDecoder,
    *,
    eager_project_initialization: bool = True,
) -> TorchTitanFit:
    """Report only concrete blockers to direct TorchTitan model registration."""

    probe = probe_torchtitan()
    blockers: list[str] = ["torchtitan_training_driver_adapter_not_implemented"]
    if not probe.installed:
        blockers.append("torchtitan_not_installed")
    if probe.missing_components:
        blockers.append("required_torchtitan_components_missing")
    if not hasattr(model_class, "Config"):
        blockers.append("model_missing_torchtitan_nested_config")
    if not hasattr(model_class, "init_states"):
        blockers.append("model_missing_torchtitan_init_states_protocol")
    if not hasattr(model_class, "parallelize"):
        blockers.append("model_missing_torchtitan_parallelize_protocol")
    if eager_project_initialization:
        blockers.append("eager_initialization_not_meta_device_ready")

    if probe.installed:
        try:
            base_model = importlib.import_module("torchtitan.protocols.model").BaseModel
        except Exception:
            blockers.append("torchtitan_base_model_unavailable")
        else:
            if not issubclass(model_class, base_model):
                blockers.append("model_not_torchtitan_base_model")

    return TorchTitanFit(
        probe=probe,
        direct_adoption_ready=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        delegable_complexity=(
            "FSDP2_HSDP_TP_PP_CP_composition",
            "meta_device_materialization",
            "activation_checkpointing_and_compile",
            "DCP_and_async_checkpoint_orchestration",
            "checkpointable_distributed_dataloader",
            "distributed_metrics_profiling_and_memory_monitoring",
        ),
        retained_project_contracts=(
            "12-6_ModelSpec_and_InitSpec_identity",
            "scratch_random_init_weight_provenance",
            "D03_D04_data_tokenizer_packing_identity",
            "D05_D18_checkpoint_semantic_and_topology_identity",
        ),
    )


def build_scale_framework_plan(
    model_spec: ModelSpec,
    init_spec: InitSpec,
    trainer_config: TrainerConfig,
    parallel_plan: ParallelPlan,
    *,
    backend: FrameworkBackend = "auto",
) -> ScaleFrameworkPlan:
    """Map 12-6 contracts to a scale runtime without changing semantic ownership."""

    if backend not in {"auto", "pytorch_native", "torchtitan"}:
        raise ValueError(f"unsupported framework backend: {backend!r}")
    parallel_plan.validate()
    native_plan = build_torch_native_plan(parallel_plan)
    titan_fit = assess_torchtitan_fit()

    if backend == "torchtitan" and not titan_fit.direct_adoption_ready:
        joined = ", ".join(titan_fit.blockers)
        raise RuntimeError(f"direct TorchTitan adoption is blocked: {joined}")
    selected = "torchtitan" if backend == "torchtitan" else "pytorch_native"
    if backend == "auto" and titan_fit.direct_adoption_ready:
        selected = "torchtitan"

    registration = (
        "torchtitan_model_registry"
        if selected == "torchtitan"
        else "twelve_six_model_constructor"
    )
    framework_sink = (
        "torchtitan_MetricsProcessor"
        if selected == "torchtitan"
        else "12-6_structured_metrics_adapter"
    )

    return ScaleFrameworkPlan(
        selected_backend=selected,
        model=ModelRegistrationPlan(
            model_class="twelve_six.model.TwelveSixDecoder",
            model_spec_identity_sha256=model_spec.identity_sha256(),
            init_spec_identity_sha256=init_spec.identity_sha256(),
            parameter_count=model_spec.parameter_count(),
            weight_origin="scratch_random_init_only",
            registration_backend=registration,
        ),
        dataset=DatasetBoundaryPlan(
            batch_contract="Mapping[str, Tensor] with input_ids and one target convention",
            accepted_target_fields=("labels", "target_ids"),
            optional_fields=("loss_mask_with_target_ids",),
            effective_data_parallel_degree=parallel_plan.data_parallel,
            peer_batch_policy="TP_PP_CP_peers_share_the_same_DP_owned_batch",
            data_identity_owner="D03_D04_manifest_tokenizer_packing_contracts",
        ),
        optimizer=OptimizerBoundaryPlan(
            name="AdamW",
            learning_rate=trainer_config.learning_rate,
            betas=trainer_config.betas,
            eps=trainer_config.eps,
            weight_decay=trainer_config.weight_decay,
            source_of_truth="twelve_six.training.TrainerConfig",
            torchtitan_mapping="OptimizersContainer/default_adamw_with_explicit_12-6_values",
        ),
        distributed=native_plan,
        checkpoint=CheckpointBoundaryPlan(
            semantic_identity_owner="D05_D18_12-6_checkpoint_identity",
            scale_storage=(
                "TorchTitan_CheckpointManager_over_DCP"
                if selected == "torchtitan"
                else "torch.distributed.checkpoint"
            ),
            framework_state=(
                "model",
                "optimizer",
                "lr_scheduler",
                "dataloader_cursor",
                "train_state",
            ),
            topology_identity_owner="D12_D18_distributed_layout_identity",
        ),
        logging=LoggingBoundaryPlan(
            source_schema="12-6.training.StepMetrics",
            fields=(
                "micro_step",
                "optimizer_step",
                "loss",
                "update_loss",
                "learning_rate",
                "grad_norm",
                "tokens",
                "optimizer_stepped",
            ),
            framework_sink=framework_sink,
        ),
        torchtitan=titan_fit,
    )


def step_metrics_event(metrics: Any) -> dict[str, Any]:
    """Convert the existing D02 metrics contract into a framework-neutral event."""

    fields = (
        "micro_step",
        "optimizer_step",
        "loss",
        "update_loss",
        "learning_rate",
        "grad_norm",
        "tokens",
        "optimizer_stepped",
    )
    missing = [name for name in fields if not hasattr(metrics, name)]
    if missing:
        raise TypeError(f"metrics object is missing fields: {', '.join(missing)}")
    return {
        "schema": "12-6.scale-step-metrics.v1",
        **{name: getattr(metrics, name) for name in fields},
    }


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        materialized = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(materialized.dtype).encode("ascii"))
        digest.update(json.dumps(list(materialized.shape)).encode("ascii"))
        digest.update(materialized.numpy().tobytes())
    return digest.hexdigest()


def execute_local_scale_smoke(
    model_spec: ModelSpec,
    init_spec: InitSpec,
    trainer_config: TrainerConfig,
    parallel_plan: ParallelPlan,
    *,
    sequence_length: int = 8,
) -> dict[str, Any]:
    """Run one real scratch-model optimizer step on CPU without distributed init."""

    if sequence_length < 2 or sequence_length > model_spec.max_seq_len:
        raise ValueError("sequence_length must be in [2, model_spec.max_seq_len]")

    import torch

    from ..training.loss import causal_lm_loss
    from ..training.trainer import build_optimizer

    before_dist = torch.distributed.is_initialized()
    torch.manual_seed(trainer_config.seed)
    model = TwelveSixDecoder(model_spec, init_spec)
    optimizer = build_optimizer(model, trainer_config)
    generator = torch.Generator(device="cpu").manual_seed(trainer_config.seed + 1)
    input_ids = torch.randint(
        0,
        model_spec.vocab_size,
        (1, sequence_length),
        generator=generator,
    )
    before_state = _state_sha256(model)
    output = model(input_ids)
    loss = causal_lm_loss(output.logits, input_ids)
    if not torch.isfinite(loss).item():
        raise RuntimeError("local scale smoke produced non-finite loss")
    loss.backward()
    optimizer.step()
    after_state = _state_sha256(model)
    if before_state == after_state:
        raise RuntimeError("local scale smoke did not update model weights")
    after_dist = torch.distributed.is_initialized()
    if before_dist != after_dist:
        raise RuntimeError("framework probe unexpectedly changed distributed initialization state")

    return {
        "schema": "12-6.scale-framework-local-smoke.v1",
        "authority": "LOCAL_FREE_CPU_MECHANICS_ONLY_NOT_SCALE_PERFORMANCE",
        "model_spec_identity_sha256": model_spec.identity_sha256(),
        "init_spec_identity_sha256": init_spec.identity_sha256(),
        "parameter_count": model_spec.parameter_count(),
        "weight_origin": "scratch_random_init_only",
        "sequence_length": sequence_length,
        "loss": float(loss.detach().cpu()),
        "state_sha256_before": before_state,
        "state_sha256_after": after_state,
        "distributed_initialized_before": before_dist,
        "distributed_initialized_after": after_dist,
        "torch_native_capabilities": asdict(torch_native_capabilities()),
        "framework_plan": build_scale_framework_plan(
            model_spec,
            init_spec,
            trainer_config,
            parallel_plan,
        ).to_dict(),
    }
