"""Fail-closed readiness contract for the S6 (~1B) engineering candidate.

This module does not authorize or launch accelerator work. It turns the existing
current-tokenizer S6 geometry into an explicit, machine-readable dependency gate and
reuses the allocation-safe scale runtime for resource planning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from twelve_six.model import StageConfig, load_stage_config

from .scale_runtime import build_meta_decoder, estimate_scale_resources

SCALE_1B_TARGET_PARAMETERS = 1_000_000_000
SCALE_1B_CURRENT_BYTE_VOCAB = 256
SCALE_1B_CONTEXT = 4096
_COMPUTE_AUTHORIZATION_PREFIXES = ("COMPUTE_AUTHORIZED:", "TRAINING_AUTHORIZED:")


def _validate_authority(name: str, value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str or None")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")


@dataclass(frozen=True, slots=True)
class Scale1BDependencies:
    """Immutable evidence references required before a material S6 launch.

    A gate is qualified only when its owning lane supplies a durable authority
    reference. Bare booleans are intentionally not accepted because they would let a
    caller self-attest readiness without binding the decision to terminal evidence.
    """

    preceding_stage_authority: str | None = None
    production_tokenizer_authority: str | None = None
    native_gqa_authority: str | None = None
    distributed_checkpoint_authority: str | None = None
    data_pipeline_authority: str | None = None
    accelerator_runtime_authority: str | None = None
    compute_authorization: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _validate_authority(name, value)
        if self.compute_authorization is not None and not self.compute_authorization.startswith(
            _COMPUTE_AUTHORIZATION_PREFIXES
        ):
            raise ValueError(
                "compute_authorization must begin with COMPUTE_AUTHORIZED: or "
                "TRAINING_AUTHORIZED:"
            )

    def authority_map(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Scale1BReadinessReport:
    stage: str
    target_parameters: int
    exact_parameters: int
    parameter_delta: int
    relative_parameter_error: float
    model_identity_sha256: str
    init_identity_sha256: str
    current_tokenizer_vocab_size: int
    attention_variant: str
    requires_native_gqa: bool
    sequence_length: int
    evidence_authorities: dict[str, str | None]
    engineering_blockers: tuple[str, ...]
    authorization_blockers: tuple[str, ...]
    ready_for_authorization_request: bool
    ready_for_material_compute: bool
    topology_resource_estimates: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("S6 stage config must contain a JSON object")
    return payload


def validate_scale_1b_candidate(path: str | Path) -> StageConfig:
    """Validate the current SCALE-06 engineering candidate without promoting it."""

    payload = _read_payload(path)
    config = load_stage_config(path)

    if config.stage != "S6":
        raise ValueError("SCALE-06 readiness requires stage S6")
    if config.target_parameters != SCALE_1B_TARGET_PARAMETERS:
        raise ValueError("S6 target must be exactly 1,000,000,000 parameters")
    relative_error = abs(config.expected_parameters - SCALE_1B_TARGET_PARAMETERS) / (
        SCALE_1B_TARGET_PARAMETERS
    )
    if relative_error > 0.01:
        raise ValueError("S6 engineering candidate must remain within 1% of the 1B target")

    if payload.get("status") != "engineering_candidate_not_frozen":
        raise ValueError("S6 candidate must remain explicitly non-frozen")
    if payload.get("promotion_allowed") is not False:
        raise ValueError("S6 engineering candidate must fail closed on promotion")
    if payload.get("requires_preceding_stage_pass") is not True:
        raise ValueError("S6 must require preceding-stage admission")

    # SCALE-06 currently exists to prove execution compatibility with the integrated
    # byte tokenizer. A future production tokenizer changes the embedding surface
    # and therefore requires a newly solved and newly hashed ModelSpec rather than a
    # silent vocabulary swap in this candidate.
    if config.model.vocab_size != SCALE_1B_CURRENT_BYTE_VOCAB:
        raise ValueError(
            "current SCALE-06 candidate must bind the 256-token byte vocabulary"
        )
    if config.model.max_seq_len != SCALE_1B_CONTEXT:
        raise ValueError("current SCALE-06 candidate must bind the 4096-token context")

    # The current S6 geometry is GQA. The canonical model path still materializes
    # repeated K/V heads; native-GQA qualification is therefore a real launch gate.
    if config.model.n_kv_heads >= config.model.n_heads:
        raise ValueError("current SCALE-06 candidate is expected to exercise GQA")
    return config


def _attention_variant(config: StageConfig) -> str:
    if config.model.n_kv_heads == config.model.n_heads:
        return "mha"
    if config.model.n_kv_heads == 1:
        return "mqa"
    return "gqa"


def _resource_estimates(
    config: StageConfig,
    *,
    world_sizes: tuple[int, ...],
    microbatch_size: int,
) -> dict[str, dict[str, Any]]:
    if not world_sizes:
        raise ValueError("at least one world size is required")
    if not isinstance(microbatch_size, int) or isinstance(microbatch_size, bool):
        raise TypeError("microbatch_size must be an integer")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")

    estimates: dict[str, dict[str, Any]] = {}
    seen: set[int] = set()
    for world_size in world_sizes:
        if not isinstance(world_size, int) or isinstance(world_size, bool):
            raise TypeError("world sizes must be integers")
        if world_size <= 0:
            raise ValueError("world sizes must be positive")
        if world_size in seen:
            raise ValueError("world sizes must be unique")
        seen.add(world_size)

        sharded = world_size > 1
        estimate = estimate_scale_resources(
            config.model,
            sequence_length=SCALE_1B_CONTEXT,
            microbatch_size=microbatch_size,
            activation_checkpointing=True,
            world_size=world_size,
            fsdp2_sharded=sharded,
        )
        estimates[str(world_size)] = {
            "world_size": world_size,
            "fsdp2_sharded": sharded,
            **asdict(estimate),
        }
    return estimates


def assess_scale_1b_readiness(
    path: str | Path,
    dependencies: Scale1BDependencies | None = None,
    *,
    world_sizes: tuple[int, ...] = (1, 4, 8),
    microbatch_size: int = 1,
) -> Scale1BReadinessReport:
    """Return a fail-closed S6 readiness decision plus allocation-safe budgets."""

    config = validate_scale_1b_candidate(path)
    deps = dependencies or Scale1BDependencies()

    engineering_blockers: list[str] = []
    if deps.preceding_stage_authority is None:
        engineering_blockers.append("preceding_stage_not_admitted")
    if deps.production_tokenizer_authority is None:
        engineering_blockers.append("production_tokenizer_not_qualified")
    if config.model.n_kv_heads < config.model.n_heads and deps.native_gqa_authority is None:
        engineering_blockers.append("native_gqa_not_qualified")
    if deps.distributed_checkpoint_authority is None:
        engineering_blockers.append("distributed_checkpoint_not_qualified")
    if deps.data_pipeline_authority is None:
        engineering_blockers.append("data_pipeline_not_qualified")
    if deps.accelerator_runtime_authority is None:
        engineering_blockers.append("accelerator_runtime_not_qualified")

    authorization_blockers: list[str] = []
    if deps.compute_authorization is None:
        authorization_blockers.append("material_compute_not_authorized")

    exact_parameters = config.model.parameter_count()
    return Scale1BReadinessReport(
        stage=config.stage,
        target_parameters=config.target_parameters,
        exact_parameters=exact_parameters,
        parameter_delta=exact_parameters - config.target_parameters,
        relative_parameter_error=abs(exact_parameters - config.target_parameters)
        / config.target_parameters,
        model_identity_sha256=config.model.identity_sha256(),
        init_identity_sha256=config.init.identity_sha256(),
        current_tokenizer_vocab_size=config.model.vocab_size,
        attention_variant=_attention_variant(config),
        requires_native_gqa=config.model.n_kv_heads < config.model.n_heads,
        sequence_length=config.model.max_seq_len,
        evidence_authorities=deps.authority_map(),
        engineering_blockers=tuple(engineering_blockers),
        authorization_blockers=tuple(authorization_blockers),
        ready_for_authorization_request=not engineering_blockers,
        ready_for_material_compute=not engineering_blockers and not authorization_blockers,
        topology_resource_estimates=_resource_estimates(
            config,
            world_sizes=world_sizes,
            microbatch_size=microbatch_size,
        ),
    )


def meta_parameter_probe(path: str | Path) -> int:
    """Construct the full ~1B module graph on meta and verify exact parameter count."""

    config = validate_scale_1b_candidate(path)
    model = build_meta_decoder(
        config.model,
        config.init,
        activation_checkpointing=True,
    )
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != config.expected_parameters:
        raise RuntimeError(
            f"S6 meta parameter mismatch: expected {config.expected_parameters}, got {actual}"
        )
    if any(parameter.device.type != "meta" for parameter in model.parameters()):
        raise RuntimeError("S6 meta probe materialized parameter storage")
    return actual
