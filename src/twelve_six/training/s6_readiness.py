"""Allocation-safe S6 (~1B) readiness checks built on the SCALE-05 runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from twelve_six.model import StageConfig, load_stage_config

from .scale_runtime import ScaleResourceEstimate, build_meta_decoder, estimate_scale_resources

S6_TARGET_PARAMETERS = 1_000_000_000
S6_CURRENT_TOKENIZER_EXPECTED_PARAMETERS = 999_761_920
S6_CURRENT_TOKENIZER_MODEL_SHA256 = (
    "f691e75eea9ca4c4edae197b5284c2564d3784a87fd3a831c6411af55dfc00be"
)


@dataclass(frozen=True, slots=True)
class S6ReadinessReport:
    stage: str
    exact_parameters: int
    target_parameters: int
    relative_target_error: float
    model_identity_sha256: str
    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_ff: int
    meta_parameter_count: int
    embedding_fraction: float
    persistent_total_bytes_per_rank: int
    full_training_checkpoint_bytes: int
    weight_only_checkpoint_bytes: int
    kv_cache_bytes_per_token_per_sequence: int
    estimated_activation_bytes_per_microbatch: int
    estimated_training_flops_per_token: int
    world_size: int
    sequence_length: int
    microbatch_size: int
    activation_checkpointing: bool
    launch_blockers: tuple[str, ...]
    authority: str = "ENGINEERING_READINESS_ONLY_NOT_COMPUTE_AUTHORIZATION"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_s6_candidate(path: str | Path) -> StageConfig:
    """Validate the exact current-tokenizer S6 engineering candidate."""

    config = load_stage_config(path)
    if config.stage != "S6":
        raise ValueError("S6 readiness requires stage S6")
    if config.target_parameters != S6_TARGET_PARAMETERS:
        raise ValueError("S6 target must be exactly 1,000,000,000 parameters")
    if config.expected_parameters != S6_CURRENT_TOKENIZER_EXPECTED_PARAMETERS:
        raise ValueError("unexpected S6 current-tokenizer parameter count")
    if config.model.identity_sha256() != S6_CURRENT_TOKENIZER_MODEL_SHA256:
        raise ValueError("unexpected S6 current-tokenizer ModelSpec identity")
    if config.model.vocab_size != 256:
        raise ValueError("current-tokenizer S6 candidate must use vocab_size=256")
    if config.model.n_heads != 32 or config.model.n_kv_heads != 8:
        raise ValueError("current S6 candidate requires 32Q/8KV GQA")
    if config.model.max_seq_len != 4096:
        raise ValueError("current S6 candidate requires max_seq_len=4096")
    if config.canonical_base != "random_init":
        raise ValueError("S6 Base must remain scratch random-init")
    relative_error = abs(config.expected_parameters - S6_TARGET_PARAMETERS) / S6_TARGET_PARAMETERS
    if relative_error > 0.001:
        raise ValueError("S6 candidate must remain within 0.1% of the 1B target")
    return config


def _launch_blockers(*, compute_authorized: bool) -> tuple[str, ...]:
    blockers = [
        "S5_PRECEDING_STAGE_NOT_PROMOTED",
        "PRODUCTION_TOKENIZER_NOT_FROZEN",
        "REPRESENTATIVE_CORPUS_NOT_FROZEN",
        "TARGET_GPU_NCCL_NOT_MEASURED",
        "NATIVE_GQA_TARGET_GPU_PARITY_NOT_MEASURED",
        "DCP_FSDP2_CHECKPOINT_RESUME_NOT_COMPOSED_ON_S6",
        "HELD_OUT_EVALUATION_NOT_BOUND_TO_S6_RUN",
    ]
    if not compute_authorized:
        blockers.append("COMPUTE_AUTHORIZED_ABSENT")
    return tuple(blockers)


def build_s6_readiness_report(
    path: str | Path,
    *,
    world_size: int = 4,
    sequence_length: int = 4096,
    microbatch_size: int = 1,
    activation_checkpointing: bool = True,
    compute_authorized: bool = False,
) -> S6ReadinessReport:
    """Construct the real 1B graph on meta and return a fail-closed launch report."""

    config = validate_s6_candidate(path)
    model = build_meta_decoder(
        config.model,
        config.init,
        activation_checkpointing=activation_checkpointing,
    )
    meta_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if meta_parameter_count != config.expected_parameters:
        raise RuntimeError(
            "S6 meta construction count mismatch: "
            f"expected {config.expected_parameters}, got {meta_parameter_count}"
        )

    estimate: ScaleResourceEstimate = estimate_scale_resources(
        config.model,
        sequence_length=sequence_length,
        microbatch_size=microbatch_size,
        activation_checkpointing=activation_checkpointing,
        world_size=world_size,
        fsdp2_sharded=world_size > 1,
    )
    return S6ReadinessReport(
        stage=config.stage,
        exact_parameters=config.expected_parameters,
        target_parameters=config.target_parameters,
        relative_target_error=(config.expected_parameters - config.target_parameters)
        / config.target_parameters,
        model_identity_sha256=config.model.identity_sha256(),
        vocab_size=config.model.vocab_size,
        context_length=config.model.max_seq_len,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        meta_parameter_count=meta_parameter_count,
        embedding_fraction=estimate.embedding_fraction,
        persistent_total_bytes_per_rank=estimate.persistent_total_bytes_per_rank,
        full_training_checkpoint_bytes=estimate.full_training_checkpoint_bytes,
        weight_only_checkpoint_bytes=estimate.weight_only_checkpoint_bytes,
        kv_cache_bytes_per_token_per_sequence=estimate.kv_cache_bytes_per_token_per_sequence,
        estimated_activation_bytes_per_microbatch=estimate.estimated_activation_bytes_per_microbatch,
        estimated_training_flops_per_token=estimate.estimated_training_flops_per_token,
        world_size=world_size,
        sequence_length=sequence_length,
        microbatch_size=microbatch_size,
        activation_checkpointing=activation_checkpointing,
        launch_blockers=_launch_blockers(compute_authorized=compute_authorized),
    )
