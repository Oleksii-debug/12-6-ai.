"""S4 (~100M) accelerator-readiness contracts and resource preflights.

This module is deliberately small and fail-closed. It does not authorize compute or
freeze S4. It binds a candidate to the current tokenizer/model semantics, provides
transparent memory/checkpoint estimates, proves meta-device construction, and gives
one executable scaled analogue for the existing Trainer.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .distributed.contracts import ModelScaleSpec, ParallelPlan
from .distributed.memory import GIB, MemoryEstimate, estimate_training_memory
from .model import ModelSpec, TwelveSixDecoder, count_trainable_parameters, load_stage_config
from .tokenization.byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer, TrainerConfig

S4_TARGET_PARAMETERS = 100_000_000
S4_DEFAULT_CONTEXT = 4096


@dataclass(frozen=True, slots=True)
class S4RunProfile:
    name: str
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    precision: str = "bf16"

    @property
    def tokens_per_optimizer_step(self) -> int:
        return (
            self.sequence_length
            * self.micro_batch_size
            * self.gradient_accumulation_steps
        )

    @property
    def scheduled_tokens(self) -> int:
        return self.tokens_per_optimizer_step * self.max_steps


@dataclass(frozen=True, slots=True)
class S4ResourceEvidence:
    exact_parameters: int
    parameter_gib_bf16: float
    gradient_gib_bf16: float
    adam_moments_gib_fp32: float
    master_weights_gib_fp32: float
    activation_gib_estimate: float
    total_training_gib_estimate: float
    inference_weight_gib_bf16: float
    training_checkpoint_gib_estimate: float
    tokens_per_optimizer_step: int
    scheduled_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalogueProbeResult:
    parameters: int
    loss: float
    optimizer_step: int
    tokens: int
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("S4 stage config must contain a JSON object")
    return payload


def validate_s4_candidate(path: str | Path):
    """Load and fail-closed validate the current-tokenizer S4 candidate."""

    payload = _read_payload(path)
    config = load_stage_config(path)
    if config.stage != "S4":
        raise ValueError("100M readiness requires stage S4")
    if config.target_parameters != S4_TARGET_PARAMETERS:
        raise ValueError("S4 target must be exactly 100,000,000 parameters")
    if abs(config.expected_parameters - S4_TARGET_PARAMETERS) / S4_TARGET_PARAMETERS > 0.01:
        raise ValueError("S4 candidate must remain within 1% of the 100M target")

    tokenizer = payload.get("tokenizer_contract")
    if not isinstance(tokenizer, dict):
        raise ValueError("S4 candidate must bind an explicit tokenizer_contract")
    expected_tokenizer = {
        "version": BYTE_TOKENIZER_VERSION,
        "vocab_size": ByteTokenizer.vocab_size,
        "config_sha256": BYTE_TOKENIZER_HASH,
        "vocab_sha256": BYTE_VOCAB_HASH,
    }
    for key, expected in expected_tokenizer.items():
        if tokenizer.get(key) != expected:
            raise ValueError(f"S4 tokenizer contract mismatch for {key}")
    if config.model.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("S4 ModelSpec vocabulary must match the current tokenizer")

    runtime = payload.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("S4 candidate must bind an explicit runtime_contract")
    if runtime.get("preferred_precision") != "bf16":
        raise ValueError("S4 accelerator precision must prefer bf16")
    if runtime.get("parallelism") != "single_gpu":
        raise ValueError("S4 is intentionally a single-GPU stage")
    if runtime.get("fsdp_required") is not False:
        raise ValueError("S4 must not require FSDP")

    # Current model.py expands K/V heads with repeat_interleave for GQA before SDPA.
    # Until that path is replaced/proven with native GQA, an accelerator readiness
    # claim would overstate its memory/throughput benefit. Use MHA for S4 instead.
    if config.model.n_kv_heads != config.model.n_heads:
        raise ValueError("S4 accelerator candidate requires MHA under current runtime")
    return config


def meta_parameter_probe(path: str | Path) -> int:
    """Instantiate the full S4 module graph on meta without materializing ~100M weights."""

    config = validate_s4_candidate(path)
    with torch.device("meta"):
        model = TwelveSixDecoder(config.model, config.init)
    actual = count_trainable_parameters(model)
    if actual != config.expected_parameters:
        raise RuntimeError(
            f"meta model count mismatch: expected {config.expected_parameters}, got {actual}"
        )
    return actual


def estimate_s4_resources(
    path: str | Path,
    profile: S4RunProfile,
    *,
    activation_multiplier: float = 8.0,
) -> S4ResourceEvidence:
    """Return conservative single-rank BF16 + AdamW resource evidence."""

    config = validate_s4_candidate(path)
    if profile.sequence_length > config.model.max_seq_len:
        raise ValueError("run sequence length exceeds ModelSpec max_seq_len")
    if profile.precision != "bf16":
        raise ValueError("S4 accelerator profiles are BF16-only by default")
    scale = ModelScaleSpec(
        total_parameters=config.expected_parameters,
        hidden_size=config.model.d_model,
        num_layers=config.model.n_layers,
        num_attention_heads=config.model.n_heads,
        sequence_length=profile.sequence_length,
        micro_batch_size=profile.micro_batch_size,
    )
    estimate: MemoryEstimate = estimate_training_memory(
        scale,
        ParallelPlan(),
        parameter_bytes=2,
        gradient_bytes=2,
        optimizer_bytes_per_parameter=8,
        master_weight_bytes=4,
        activation_bytes=2,
        activation_multiplier=activation_multiplier,
    )
    params = config.expected_parameters
    training_checkpoint_bytes = params * (2 + 8 + 4)
    return S4ResourceEvidence(
        exact_parameters=params,
        parameter_gib_bf16=estimate.parameter_bytes_per_rank / GIB,
        gradient_gib_bf16=estimate.gradient_bytes_per_rank / GIB,
        adam_moments_gib_fp32=estimate.optimizer_bytes_per_rank / GIB,
        master_weights_gib_fp32=estimate.master_weight_bytes_per_rank / GIB,
        activation_gib_estimate=estimate.activation_bytes_per_rank / GIB,
        total_training_gib_estimate=estimate.total_gib_per_rank,
        inference_weight_gib_bf16=(params * 2) / GIB,
        training_checkpoint_gib_estimate=training_checkpoint_bytes / GIB,
        tokens_per_optimizer_step=profile.tokens_per_optimizer_step,
        scheduled_tokens=profile.scheduled_tokens,
    )


def accelerator_preflight() -> dict[str, Any]:
    """Report local accelerator capability without allocating the S4 model."""

    available = torch.cuda.is_available()
    result: dict[str, Any] = {
        "cuda_available": available,
        "torch_version": torch.__version__,
        "bf16_supported": False,
        "device_name": None,
        "device_memory_gib": None,
    }
    if not available:
        return result
    result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    properties = torch.cuda.get_device_properties(0)
    result["device_name"] = properties.name
    result["device_memory_gib"] = properties.total_memory / GIB
    return result


def scaled_analogue_spec() -> ModelSpec:
    """Small geometry-preserving MHA analogue for cheap forward/backward/update probes."""

    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=128,
        d_model=96,
        n_layers=2,
        n_heads=3,
        n_kv_heads=3,
        head_dim=32,
        d_ff=288,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=32,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def run_scaled_analogue(*, sequence_length: int = 64, batch_size: int = 2) -> AnalogueProbeResult:
    """Execute one real CPU optimizer step through the current model + Trainer path."""

    spec = scaled_analogue_spec()
    if sequence_length < 2 or sequence_length > spec.max_seq_len:
        raise ValueError("analogue sequence_length must be in [2, 128]")
    if batch_size < 1:
        raise ValueError("analogue batch_size must be positive")
    torch.manual_seed(404)
    model = TwelveSixDecoder(spec)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=3e-4,
            weight_decay=0.1,
            max_steps=1,
            gradient_accumulation_steps=1,
            precision="fp32",
            seed=404,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )
    generator = torch.Generator(device="cpu").manual_seed(405)
    input_ids = torch.randint(
        low=0,
        high=spec.vocab_size,
        size=(batch_size, sequence_length),
        generator=generator,
    )
    started = time.perf_counter()
    metrics = trainer.train_microbatch({"input_ids": input_ids})
    wall = time.perf_counter() - started
    if not metrics.optimizer_stepped or metrics.optimizer_step != 1:
        raise RuntimeError("analogue did not commit exactly one optimizer step")
    return AnalogueProbeResult(
        parameters=count_trainable_parameters(model),
        loss=metrics.loss,
        optimizer_step=metrics.optimizer_step,
        tokens=metrics.tokens,
        wall_seconds=wall,
    )
