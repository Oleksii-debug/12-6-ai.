"""400M-scale execution helpers that preserve the canonical model/trainer contracts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.checkpoint import checkpoint

from twelve_six.model import CausalLMOutput, InitSpec, ModelSpec, TwelveSixDecoder

from .config import TrainerConfig
from .trainer import Trainer, build_optimizer, build_scheduler

AttentionPolicy = Literal["auto", "flash_required", "math"]
PersistentStateMode = Literal["replicated_fp32_adamw", "fsdp2_fp32_adamw"]


@dataclass(frozen=True, slots=True)
class ScaleRuntimePolicy:
    """Execution knobs that become material at accelerator scale."""

    activation_checkpointing: bool = True
    attention_policy: AttentionPolicy = "flash_required"
    persistent_state_mode: PersistentStateMode = "fsdp2_fp32_adamw"
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.attention_policy not in {"auto", "flash_required", "math"}:
            raise ValueError(f"unsupported attention_policy: {self.attention_policy!r}")
        if self.persistent_state_mode not in {
            "replicated_fp32_adamw",
            "fsdp2_fp32_adamw",
        }:
            raise ValueError(
                f"unsupported persistent_state_mode: {self.persistent_state_mode!r}"
            )
        if not isinstance(self.world_size, int) or isinstance(self.world_size, bool):
            raise TypeError("world_size must be an integer")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.persistent_state_mode == "replicated_fp32_adamw" and self.world_size != 1:
            raise ValueError(
                "replicated_fp32_adamw estimate is intentionally single-rank; "
                "use fsdp2_fp32_adamw for sharded persistent-state estimates"
            )


@dataclass(frozen=True, slots=True)
class ScaleResourceEstimate:
    parameters: int
    embedding_parameters: int
    embedding_fraction: float
    persistent_parameter_bytes_per_rank: int
    persistent_gradient_bytes_per_rank: int
    persistent_optimizer_bytes_per_rank: int
    persistent_total_bytes_per_rank: int
    full_training_checkpoint_bytes: int
    weight_only_checkpoint_bytes: int
    kv_cache_bytes_per_token_per_sequence: int
    estimated_activation_bytes_per_microbatch: int
    estimated_training_flops_per_token: int

    def training_flops(self, tokens: int) -> int:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError("tokens must be a positive integer")
        return self.estimated_training_flops_per_token * tokens


def estimate_scale_resources(
    spec: ModelSpec,
    *,
    sequence_length: int,
    microbatch_size: int = 1,
    activation_checkpointing: bool = True,
    world_size: int = 1,
    fsdp2_sharded: bool = False,
) -> ScaleResourceEstimate:
    """Return an analytical accelerator budget without allocating model tensors.

    The persistent-state model matches the current project semantics: fp32 trainable
    parameters + fp32 gradients + two fp32 AdamW moments. FSDP2 divides those
    persistent tensors across data-parallel ranks. Activation bytes are a planning
    estimate, not a measured CUDA peak.
    """
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
        raise TypeError("sequence_length must be an integer")
    if sequence_length <= 0 or sequence_length > spec.max_seq_len:
        raise ValueError("sequence_length must be in [1, spec.max_seq_len]")
    if not isinstance(microbatch_size, int) or isinstance(microbatch_size, bool):
        raise TypeError("microbatch_size must be an integer")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not fsdp2_sharded and world_size != 1:
        raise ValueError(
            "world_size > 1 requires fsdp2_sharded=True for this persistent-state estimate"
        )

    breakdown = spec.parameter_breakdown()
    parameters = breakdown["total"]
    divisor = world_size if fsdp2_sharded else 1

    # Current Trainer keeps fp32 parameters and uses autocast for bf16/fp16 compute.
    parameter_bytes = (parameters * 4 + divisor - 1) // divisor
    gradient_bytes = (parameters * 4 + divisor - 1) // divisor
    optimizer_bytes = (parameters * 8 + divisor - 1) // divisor

    # A durable training checkpoint does not need gradients. It does need model
    # parameters and both Adam moments; scalar/metadata overhead is deliberately
    # excluded and should be budgeted separately by the checkpoint layer.
    full_checkpoint_bytes = parameters * 12
    weight_only_bytes = parameters * 4

    # K + V, before GQA expansion, fp16/bf16 cache.
    kv_cache_bytes_per_token = (
        2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * 2
    )

    tokens_per_microbatch = sequence_length * microbatch_size
    if activation_checkpointing:
        # Checkpoint boundaries retain one hidden-state-sized tensor per block.
        saved_activation_elements = (
            tokens_per_microbatch * spec.n_layers * spec.d_model
        )
        # Peak recomputation must still hold one block's major intermediates.
        peak_block_elements = tokens_per_microbatch * (
            4 * spec.d_model
            + spec.q_dim
            + 2 * spec.kv_dim
            + 2 * spec.d_ff
        )
        # The dense LM head is material at 32K vocab and must not be omitted.
        logits_elements = tokens_per_microbatch * spec.vocab_size
        activation_elements = (
            saved_activation_elements + peak_block_elements + logits_elements
        )
    else:
        per_layer_elements = tokens_per_microbatch * (
            4 * spec.d_model
            + spec.q_dim
            + 2 * spec.kv_dim
            + 2 * spec.d_ff
        )
        logits_elements = tokens_per_microbatch * spec.vocab_size
        activation_elements = spec.n_layers * per_layer_elements + logits_elements

    # bf16/fp16 saved activations. Kernels may use fp32 accumulators/workspaces,
    # so this is a lower-bound planning estimate rather than a peak-memory claim.
    activation_bytes = activation_elements * 2

    # 6*P covers parameterized forward/backward matrix work. The extra term covers
    # causal QK^T and AV score/value work, which becomes important at 4K context.
    parameterized_flops_per_token = 6 * parameters
    attention_score_flops_per_token = (
        12 * spec.n_layers * sequence_length * spec.q_dim
    )

    return ScaleResourceEstimate(
        parameters=parameters,
        embedding_parameters=breakdown["token_embedding"],
        embedding_fraction=breakdown["token_embedding"] / parameters,
        persistent_parameter_bytes_per_rank=parameter_bytes,
        persistent_gradient_bytes_per_rank=gradient_bytes,
        persistent_optimizer_bytes_per_rank=optimizer_bytes,
        persistent_total_bytes_per_rank=(
            parameter_bytes + gradient_bytes + optimizer_bytes
        ),
        full_training_checkpoint_bytes=full_checkpoint_bytes,
        weight_only_checkpoint_bytes=weight_only_bytes,
        kv_cache_bytes_per_token_per_sequence=kv_cache_bytes_per_token,
        estimated_activation_bytes_per_microbatch=activation_bytes,
        estimated_training_flops_per_token=(
            parameterized_flops_per_token + attention_score_flops_per_token
        ),
    )


@contextmanager
def attention_backend_context(
    policy: AttentionPolicy,
    *,
    device: torch.device,
) -> Iterator[None]:
    """Fail closed when a run requires Flash SDPA but the backend is unavailable."""
    if policy == "auto":
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "explicit SDPA backend selection is unavailable in this PyTorch build"
        ) from exc

    if policy == "flash_required":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("flash_required attention needs an available CUDA device")
        backend = SDPBackend.FLASH_ATTENTION
    elif policy == "math":
        backend = SDPBackend.MATH
    else:
        raise ValueError(f"unsupported attention policy: {policy!r}")

    with sdpa_kernel(backends=[backend]):
        yield


@contextmanager
def _autocast_and_attention(
    autocast_context: object,
    *,
    attention_policy: AttentionPolicy,
    device: torch.device,
) -> Iterator[None]:
    with autocast_context:  # type: ignore[attr-defined]
        with attention_backend_context(attention_policy, device=device):
            yield


class ActivationCheckpointedDecoder(TwelveSixDecoder):
    """State-dict-compatible decoder with blockwise non-reentrant checkpointing."""

    def forward(self, input_ids: Tensor) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] <= 0:
            raise ValueError("input_ids sequence must be non-empty")
        if input_ids.shape[1] > self.spec.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len "
                f"{self.spec.max_seq_len}"
            )

        x = self.token_embedding(input_ids)
        use_checkpointing = self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpointing:
                x = checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                x = block(x)
        x = self.final_norm(x)
        return CausalLMOutput(logits=self.lm_head(x))


def build_meta_decoder(
    spec: ModelSpec,
    init_spec: InitSpec | None = None,
    *,
    activation_checkpointing: bool = True,
) -> TwelveSixDecoder:
    """Instantiate the real decoder architecture on meta without parameter storage."""
    decoder_cls = (
        ActivationCheckpointedDecoder
        if activation_checkpointing
        else TwelveSixDecoder
    )
    with torch.device("meta"):
        model = decoder_cls(spec, init_spec)
    if any(parameter.device.type != "meta" for parameter in model.parameters()):
        raise RuntimeError("meta construction produced a materialized parameter")
    if sum(parameter.numel() for parameter in model.parameters()) != spec.parameter_count():
        raise RuntimeError("meta model parameter count diverges from ModelSpec")
    return model


def reset_materialized_decoder_parameters_(model: TwelveSixDecoder) -> None:
    """Initialize a meta->to_empty materialized decoder with canonical InitSpec rules.

    FSDP2 meta initialization requires materialization after sharding. The canonical
    decoder currently initializes in __init__, which is a no-op on meta tensors, so
    this helper provides the post-materialization initialization seam without
    changing checkpoint keys or model identity.
    """
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise ValueError("materialize meta parameters with to_empty before initialization")

    model.apply(model._init_module)  # project-internal canonical initializer
    residual_std = model.init_spec.residual_std(model.spec.n_layers)
    for block in model.blocks:
        nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
        nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)
    if model.spec.tie_word_embeddings:
        model.lm_head.weight = model.token_embedding.weight

    actual = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if actual != model.spec.parameter_count():
        raise RuntimeError(
            f"materialized parameter count invariant failed: model={actual}, "
            f"spec={model.spec.parameter_count()}"
        )


def prepare_fsdp2_decoder(
    spec: ModelSpec,
    init_spec: InitSpec | None = None,
    *,
    device: str | torch.device,
    activation_checkpointing: bool = True,
) -> TwelveSixDecoder:
    """Meta-build, bottom-up fully-shard, materialize, and initialize a decoder.

    The caller must initialize torch.distributed and select the local CUDA device
    before calling this function. The optimizer must be created only after this
    function returns, so that it owns DTensor/sharded parameters.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("prepare_fsdp2_decoder requires initialized torch.distributed")

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("FSDP2 scale execution currently requires a CUDA device")

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("this PyTorch build does not expose FSDP2 fully_shard") from exc

    model = build_meta_decoder(
        spec,
        init_spec,
        activation_checkpointing=activation_checkpointing,
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    for block in model.blocks:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    # FSDP2 preserves meta tensors through sharding; materialize only the local
    # shards, then apply the canonical scratch initializer.
    model.to_empty(device=target_device)
    reset_materialized_decoder_parameters_(model)
    return model


class ExternallyPlacedTrainer(Trainer):
    """Trainer variant for pre-placed or pre-sharded models.

    Base Trainer intentionally owns model.to(device), which is correct for S0 but
    unsafe for FSDP2: fully_shard must run before optimizer creation and a sharded
    model must not be moved wholesale afterward. This variant keeps all Trainer
    safety/accumulation/resume behavior while making placement ownership explicit.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        device: str | torch.device,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        attention_policy: AttentionPolicy = "auto",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.attention_policy = attention_policy

        if any(parameter.device.type == "meta" for parameter in model.parameters()):
            raise ValueError("ExternallyPlacedTrainer requires materialized parameters")

        self._configure_determinism(config)
        self.optimizer = optimizer or build_optimizer(model, config)
        self.scheduler = (
            scheduler if scheduler is not None else build_scheduler(self.optimizer, config)
        )
        self.scaler = self._build_scaler()

        self.micro_step = 0
        self.optimizer_step = 0
        self.tokens_seen = 0
        self._pending_tokens = 0
        self._pending_loss_sum = 0.0
        self._update_incomplete = False
        self._failure_reason: str | None = None
        self.optimizer.zero_grad(set_to_none=True)

    def _autocast_context(self):
        return _autocast_and_attention(
            super()._autocast_context(),
            attention_policy=self.attention_policy,
            device=self.device,
        )
