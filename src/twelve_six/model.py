from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Versioned checkpoint/inference semantics for a 12-6 Base decoder."""

    schema_version: int
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    activation: str = "swiglu"
    norm_kind: str = "rmsnorm"
    norm_placement: str = "pre"
    norm_eps: float = 1e-5
    position_embedding: str = "rope"
    rope_theta: float = 10_000.0
    rope_rotary_dim: int = 0
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    final_norm: bool = True
    tie_word_embeddings: bool = True
    lm_head_bias: bool = False

    def __post_init__(self) -> None:
        positive_ints = {
            "schema_version": self.schema_version,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "d_ff": self.d_ff,
            "rope_rotary_dim": self.rope_rotary_dim,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.schema_version != 1:
            raise ValueError(f"unsupported ModelSpec schema_version: {self.schema_version}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head_dim")
        if self.rope_rotary_dim > self.head_dim:
            raise ValueError("rope_rotary_dim cannot exceed head_dim")
        if self.rope_rotary_dim % 2 != 0:
            raise ValueError("rope_rotary_dim must be even")
        if self.activation != "swiglu":
            raise ValueError("ModelSpec v1 supports activation='swiglu' only")
        if self.norm_kind != "rmsnorm":
            raise ValueError("ModelSpec v1 supports norm_kind='rmsnorm' only")
        if self.norm_placement != "pre":
            raise ValueError("ModelSpec v1 supports norm_placement='pre' only")
        if self.position_embedding != "rope":
            raise ValueError("ModelSpec v1 supports position_embedding='rope' only")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def tie_embeddings(self) -> bool:
        """Compatibility alias for the pre-v1 field name."""
        return self.tie_word_embeddings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelSpec:
        return cls(**payload)

    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def parameter_breakdown(self) -> dict[str, int]:
        embedding = self.vocab_size * self.d_model
        attention_weights_per_layer = 2 * self.d_model * (self.q_dim + self.kv_dim)
        attention_biases_per_layer = 0
        if self.attention_bias:
            attention_biases_per_layer = self.q_dim + 2 * self.kv_dim + self.d_model

        mlp_weights_per_layer = 3 * self.d_model * self.d_ff
        mlp_biases_per_layer = 2 * self.d_ff + self.d_model if self.mlp_bias else 0
        norms_per_layer = 2 * self.d_model
        attention_per_layer = attention_weights_per_layer + attention_biases_per_layer
        mlp_per_layer = mlp_weights_per_layer + mlp_biases_per_layer
        block_per_layer = attention_per_layer + mlp_per_layer + norms_per_layer
        final_norm = self.d_model if self.final_norm else 0
        lm_head_extra = 0 if self.tie_word_embeddings else self.vocab_size * self.d_model
        if self.lm_head_bias:
            lm_head_extra += self.vocab_size
        total = embedding + self.n_layers * block_per_layer + final_norm + lm_head_extra
        return {
            "token_embedding": embedding,
            "attention_weights_per_layer": attention_weights_per_layer,
            "attention_biases_per_layer": attention_biases_per_layer,
            "attention_per_layer": attention_per_layer,
            "mlp_weights_per_layer": mlp_weights_per_layer,
            "mlp_biases_per_layer": mlp_biases_per_layer,
            "mlp_per_layer": mlp_per_layer,
            "norms_per_layer": norms_per_layer,
            "block_per_layer": block_per_layer,
            "blocks_total": self.n_layers * block_per_layer,
            "final_norm": final_norm,
            "lm_head_extra": lm_head_extra,
            "total": total,
        }

    def parameter_count(self) -> int:
        return self.parameter_breakdown()["total"]


@dataclass(frozen=True, slots=True)
class InitSpec:
    """Scratch-initialization semantics kept separate from ModelSpec identity."""

    schema_version: int = 1
    family: str = "normal"
    std: float = 0.02
    residual_branch_scale: str = "sqrt_2_layers"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported InitSpec schema_version: {self.schema_version}")
        if self.family != "normal":
            raise ValueError("InitSpec v1 supports family='normal' only")
        if self.std <= 0:
            raise ValueError("InitSpec std must be positive")
        if self.residual_branch_scale not in {"none", "sqrt_2_layers"}:
            raise ValueError("unsupported residual_branch_scale")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InitSpec:
        return cls(**payload)

    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def residual_std(self, n_layers: int) -> float:
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.residual_branch_scale == "none":
            return self.std
        return self.std / math.sqrt(2.0 * n_layers)


@dataclass(frozen=True, slots=True)
class StageConfig:
    stage: str
    target_parameters: int
    expected_parameters: int
    canonical_base: str
    expected_model_identity_sha256: str
    expected_init_identity_sha256: str
    model: ModelSpec
    init: InitSpec

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StageConfig:
        return cls(
            stage=str(payload["stage"]),
            target_parameters=int(payload["target_parameters"]),
            expected_parameters=int(payload["expected_parameters"]),
            canonical_base=str(payload["canonical_base"]),
            expected_model_identity_sha256=str(payload["expected_model_identity_sha256"]),
            expected_init_identity_sha256=str(payload["expected_init_identity_sha256"]),
            model=ModelSpec.from_dict(payload["model"]),
            init=InitSpec.from_dict(payload["init"]),
        )


def load_stage_config(path: str | Path) -> StageConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = StageConfig.from_dict(payload)
    actual = config.model.parameter_count()
    if actual != config.expected_parameters:
        raise ValueError(
            "stage config parameter mismatch: "
            f"expected {config.expected_parameters}, formula gives {actual}"
        )
    if config.canonical_base != "random_init":
        raise ValueError("canonical Base stage configs must declare random_init")
    if config.model.identity_sha256() != config.expected_model_identity_sha256:
        raise ValueError("stage config ModelSpec identity hash mismatch")
    if config.init.identity_sha256() != config.expected_init_identity_sha256:
        raise ValueError("stage config InitSpec identity hash mismatch")
    return config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, theta: float) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError("RoPE rotary_dim must be a positive even integer")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if (
            not isinstance(position_offset, int)
            or isinstance(position_offset, bool)
            or position_offset < 0
        ):
            raise ValueError("position_offset must be a non-negative integer")
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=device,
            dtype=torch.float32,
        )
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.repeat_interleave(freqs, 2, dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def _rotate_pairs(x: Tensor) -> Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, rotary_dim: int) -> Tensor:
    rotary = x[..., :rotary_dim]
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1])
    rotated = rotary * cos + _rotate_pairs(rotary) * sin
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


@dataclass(frozen=True, slots=True)
class AttentionKVCache:
    """Ephemeral unexpanded K/V tensors for one decoder attention layer."""

    key: Tensor
    value: Tensor

    @property
    def sequence_length(self) -> int:
        if self.key.ndim != 4:
            raise ValueError("cached key must have shape [batch, kv_heads, sequence, head_dim]")
        return int(self.key.shape[2])


@dataclass(frozen=True, slots=True)
class DecoderKVCache:
    """Ephemeral inference-only cache bound to one ModelSpec identity."""

    model_spec_sha256: str
    sequence_length: int
    batch_size: int
    layers: tuple[AttentionKVCache, ...]


class CausalSelfAttention(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.n_heads = spec.n_heads
        self.n_kv_heads = spec.n_kv_heads
        self.head_dim = spec.head_dim
        self.q_dim = spec.q_dim
        self.kv_dim = spec.kv_dim
        self.rotary_dim = spec.rope_rotary_dim
        self.dropout = spec.attention_dropout

        self.q_proj = nn.Linear(spec.d_model, spec.q_dim, bias=spec.attention_bias)
        self.k_proj = nn.Linear(spec.d_model, spec.kv_dim, bias=spec.attention_bias)
        self.v_proj = nn.Linear(spec.d_model, spec.kv_dim, bias=spec.attention_bias)
        self.out_proj = nn.Linear(spec.q_dim, spec.d_model, bias=spec.attention_bias)
        self.rope = RotaryEmbedding(spec.rope_rotary_dim, spec.rope_theta)

    def _project_qkv(
        self,
        x: Tensor,
        *,
        position_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope.cos_sin(
            seq_len,
            device=x.device,
            dtype=q.dtype,
            position_offset=position_offset,
        )
        q = apply_rope(q, cos, sin, self.rotary_dim)
        k = apply_rope(k, cos, sin, self.rotary_dim)
        return q, k, v

    def _expand_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.n_kv_heads == self.n_heads:
            return k, v
        repeats = self.n_heads // self.n_kv_heads
        return k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, *, is_causal: bool) -> Tensor:
        expanded_k, expanded_v = self._expand_kv(k, v)
        attended = F.scaled_dot_product_attention(
            q,
            expanded_k,
            expanded_v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        batch = q.shape[0]
        seq_len = q.shape[2]
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.q_dim)
        return self.out_proj(attended)

    def _validate_cache(self, cache: AttentionKVCache, x: Tensor) -> int:
        if not isinstance(cache, AttentionKVCache):
            raise TypeError("attention cache must be an AttentionKVCache")
        key = cache.key
        value = cache.value
        expected_prefix = (x.shape[0], self.n_kv_heads)
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError("cached K/V tensors must have identical rank-4 shapes")
        if tuple(key.shape[:2]) != expected_prefix or key.shape[3] != self.head_dim:
            raise ValueError("cached K/V tensor geometry is incompatible with attention")
        if key.shape[2] <= 0:
            raise ValueError("cached K/V sequence must be non-empty")
        if key.device != x.device or value.device != x.device:
            raise ValueError("cached K/V tensors must be on the same device as input")
        if key.dtype != x.dtype or value.dtype != x.dtype:
            raise ValueError("cached K/V tensors must use the same dtype as attention input")
        return int(key.shape[2])

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x, position_offset=0)
        return self._attend(q, k, v, is_causal=True)

    def prefill(self, x: Tensor) -> tuple[Tensor, AttentionKVCache]:
        """Run a normal causal prompt pass while retaining unexpanded K/V state."""
        q, k, v = self._project_qkv(x, position_offset=0)
        output = self._attend(q, k, v, is_causal=True)
        return output, AttentionKVCache(key=k, value=v)

    def decode_one(
        self,
        x: Tensor,
        cache: AttentionKVCache,
    ) -> tuple[Tensor, AttentionKVCache]:
        """Append exactly one position and attend to all cached prior positions."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("cached attention decode requires exactly one input position")
        position_offset = self._validate_cache(cache, x)
        q, new_k, new_v = self._project_qkv(x, position_offset=position_offset)
        key = torch.cat((cache.key, new_k), dim=2)
        value = torch.cat((cache.value, new_v), dim=2)
        output = self._attend(q, key, value, is_causal=False)
        return output, AttentionKVCache(key=key, value=value)


class SwiGLU(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(spec.d_model, spec.d_ff, bias=spec.mlp_bias)
        self.up_proj = nn.Linear(spec.d_model, spec.d_ff, bias=spec.mlp_bias)
        self.down_proj = nn.Linear(spec.d_ff, spec.d_model, bias=spec.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(spec.d_model, spec.norm_eps)
        self.attn = CausalSelfAttention(spec)
        self.mlp_norm = RMSNorm(spec.d_model, spec.norm_eps)
        self.mlp = SwiGLU(spec)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def prefill(self, x: Tensor) -> tuple[Tensor, AttentionKVCache]:
        attention, cache = self.attn.prefill(self.attn_norm(x))
        x = x + attention
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache

    def decode_one(
        self,
        x: Tensor,
        cache: AttentionKVCache,
    ) -> tuple[Tensor, AttentionKVCache]:
        attention, next_cache = self.attn.decode_one(self.attn_norm(x), cache)
        x = x + attention
        x = x + self.mlp(self.mlp_norm(x))
        return x, next_cache


@dataclass(slots=True)
class CausalLMOutput:
    logits: Tensor


class TwelveSixDecoder(nn.Module):
    """Random-initialized decoder-only causal language model for canonical 12-6 Base."""

    def __init__(self, spec: ModelSpec, init_spec: InitSpec | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.init_spec = InitSpec() if init_spec is None else init_spec
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(spec) for _ in range(spec.n_layers))
        self.final_norm = RMSNorm(spec.d_model, spec.norm_eps) if spec.final_norm else nn.Identity()
        self.lm_head = nn.Linear(
            spec.d_model,
            spec.vocab_size,
            bias=spec.lm_head_bias,
        )

        self.apply(self._init_module)
        residual_std = self.init_spec.residual_std(spec.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)
        if spec.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        actual = count_trainable_parameters(self)
        expected = spec.parameter_count()
        if actual != expected:
            raise RuntimeError(f"parameter count invariant failed: model={actual}, spec={expected}")

    def _init_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_spec.std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: Tensor) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] <= 0:
            raise ValueError("input_ids sequence must be non-empty")
        if input_ids.shape[1] > self.spec.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len {self.spec.max_seq_len}"
            )
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return CausalLMOutput(logits=self.lm_head(x))

    @torch.no_grad()
    def prefill_kv_cache(self, input_ids: Tensor) -> tuple[CausalLMOutput, DecoderKVCache]:
        """Run an inference prompt once and return its per-layer K/V cache."""
        if self.training:
            raise RuntimeError("KV-cache inference requires model.eval()")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] <= 0:
            raise ValueError("input_ids sequence must be non-empty")
        if input_ids.shape[1] > self.spec.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len {self.spec.max_seq_len}"
            )
        x = self.token_embedding(input_ids)
        layer_caches: list[AttentionKVCache] = []
        for block in self.blocks:
            x, layer_cache = block.prefill(x)
            layer_caches.append(layer_cache)
        x = self.final_norm(x)
        cache = DecoderKVCache(
            model_spec_sha256=self.spec.identity_sha256(),
            sequence_length=int(input_ids.shape[1]),
            batch_size=int(input_ids.shape[0]),
            layers=tuple(layer_caches),
        )
        return CausalLMOutput(logits=self.lm_head(x)), cache

    @torch.no_grad()
    def decode_one_with_kv_cache(
        self,
        input_ids: Tensor,
        cache: DecoderKVCache,
    ) -> tuple[CausalLMOutput, DecoderKVCache]:
        """Decode exactly one new token from an existing inference cache."""
        if self.training:
            raise RuntimeError("KV-cache inference requires model.eval()")
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("cached decoder input_ids must have shape [batch, 1]")
        if not isinstance(cache, DecoderKVCache):
            raise TypeError("cache must be a DecoderKVCache")
        if cache.model_spec_sha256 != self.spec.identity_sha256():
            raise ValueError("KV cache ModelSpec identity does not match decoder")
        if cache.batch_size != input_ids.shape[0]:
            raise ValueError("KV cache batch size does not match decoder input")
        if cache.sequence_length <= 0:
            raise ValueError("KV cache sequence length must be positive")
        if cache.sequence_length >= self.spec.max_seq_len:
            raise ValueError("KV cache is already at model max_seq_len")
        if len(cache.layers) != len(self.blocks):
            raise ValueError("KV cache layer count does not match decoder")
        if any(layer.sequence_length != cache.sequence_length for layer in cache.layers):
            raise ValueError("KV cache layer sequence lengths are inconsistent")
        x = self.token_embedding(input_ids)
        next_layers: list[AttentionKVCache] = []
        for block, layer_cache in zip(self.blocks, cache.layers, strict=True):
            x, next_cache = block.decode_one(x, layer_cache)
            next_layers.append(next_cache)
        x = self.final_norm(x)
        next_decoder_cache = DecoderKVCache(
            model_spec_sha256=cache.model_spec_sha256,
            sequence_length=cache.sequence_length + 1,
            batch_size=cache.batch_size,
            layers=tuple(next_layers),
        )
        return CausalLMOutput(logits=self.lm_head(x)), next_decoder_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.shape[1] > self.spec.max_seq_len:
            raise ValueError("prompt exceeds max_seq_len")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        generated = input_ids
        steps = min(max_new_tokens, self.spec.max_seq_len - input_ids.shape[1])
        for _ in range(steps):
            logits = self(generated).logits[:, -1, :]
            if do_sample:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    threshold = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
