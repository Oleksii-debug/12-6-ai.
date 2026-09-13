from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .model import ModelSpec


@dataclass(frozen=True, slots=True)
class DenseScalingTemplate:
    """A non-canonical architecture template used to search near a parameter target."""

    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff_multiple: int = 64
    rope_theta: float = 10_000.0
    rope_rotary_dim: int | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    final_norm: bool = True
    tie_word_embeddings: bool = True
    lm_head_bias: bool = False

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "d_ff_multiple": self.d_ff_multiple,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.rope_rotary_dim is not None:
            if not isinstance(self.rope_rotary_dim, int) or isinstance(self.rope_rotary_dim, bool):
                raise TypeError("rope_rotary_dim must be an integer or None")
            if self.rope_rotary_dim <= 0 or self.rope_rotary_dim > self.head_dim:
                raise ValueError("rope_rotary_dim must be in [1, head_dim]")
            if self.rope_rotary_dim % 2 != 0:
                raise ValueError("rope_rotary_dim must be even")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def attention_variant(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "mha"
        if self.n_kv_heads == 1:
            return "mqa"
        return "gqa"

    def model_spec(self, d_ff: int) -> ModelSpec:
        if not isinstance(d_ff, int) or isinstance(d_ff, bool):
            raise TypeError("d_ff must be an integer")
        if d_ff <= 0 or d_ff % self.d_ff_multiple != 0:
            raise ValueError("d_ff must be a positive multiple of d_ff_multiple")
        rotary_dim = self.head_dim if self.rope_rotary_dim is None else self.rope_rotary_dim
        return ModelSpec(
            schema_version=1,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            d_ff=d_ff,
            activation="swiglu",
            norm_kind="rmsnorm",
            norm_placement="pre",
            norm_eps=1e-5,
            position_embedding="rope",
            rope_theta=self.rope_theta,
            rope_rotary_dim=rotary_dim,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            attention_dropout=0.0,
            final_norm=self.final_norm,
            tie_word_embeddings=self.tie_word_embeddings,
            lm_head_bias=self.lm_head_bias,
        )


@dataclass(frozen=True, slots=True)
class DenseParameterBreakdown:
    """Exact analytic trainable-parameter decomposition for a dense ModelSpec v1."""

    token_embedding: int
    attention_weights_per_layer: int
    attention_biases_per_layer: int
    attention_per_layer: int
    mlp_weights_per_layer: int
    mlp_biases_per_layer: int
    mlp_per_layer: int
    norms_per_layer: int
    block_per_layer: int
    blocks_total: int
    final_norm: int
    lm_head_extra: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def dense_parameter_breakdown(
    template: DenseScalingTemplate,
    d_ff: int,
) -> DenseParameterBreakdown:
    """Count dense parameters analytically without materializing model tensors."""

    if not isinstance(d_ff, int) or isinstance(d_ff, bool):
        raise TypeError("d_ff must be an integer")
    if d_ff <= 0 or d_ff % template.d_ff_multiple != 0:
        raise ValueError("d_ff must be a positive multiple of d_ff_multiple")

    embedding = template.vocab_size * template.d_model
    attention_weights = 2 * template.d_model * (template.q_dim + template.kv_dim)
    attention_biases = 0
    if template.attention_bias:
        attention_biases = template.q_dim + 2 * template.kv_dim + template.d_model
    attention = attention_weights + attention_biases

    mlp_weights = 3 * template.d_model * d_ff
    mlp_biases = 2 * d_ff + template.d_model if template.mlp_bias else 0
    mlp = mlp_weights + mlp_biases
    norms = 2 * template.d_model
    block = attention + mlp + norms
    blocks_total = template.n_layers * block
    final_norm = template.d_model if template.final_norm else 0

    lm_head_extra = 0
    if not template.tie_word_embeddings:
        lm_head_extra += template.vocab_size * template.d_model
    if template.lm_head_bias:
        lm_head_extra += template.vocab_size

    total = embedding + blocks_total + final_norm + lm_head_extra
    return DenseParameterBreakdown(
        token_embedding=embedding,
        attention_weights_per_layer=attention_weights,
        attention_biases_per_layer=attention_biases,
        attention_per_layer=attention,
        mlp_weights_per_layer=mlp_weights,
        mlp_biases_per_layer=mlp_biases,
        mlp_per_layer=mlp,
        norms_per_layer=norms,
        block_per_layer=block,
        blocks_total=blocks_total,
        final_norm=final_norm,
        lm_head_extra=lm_head_extra,
        total=total,
    )


@dataclass(frozen=True, slots=True)
class DenseScalingCandidate:
    """Exact-count result from a target search; it is not a stage-promotion decision."""

    target_parameters: int
    spec: ModelSpec

    @property
    def exact_parameters(self) -> int:
        return self.spec.parameter_count()

    @property
    def parameter_delta(self) -> int:
        return self.exact_parameters - self.target_parameters

    @property
    def relative_error(self) -> float:
        return abs(self.parameter_delta) / self.target_parameters

    @property
    def model_identity_sha256(self) -> str:
        return self.spec.identity_sha256()

    @property
    def ffn_ratio(self) -> float:
        return self.spec.d_ff / self.spec.d_model

    @property
    def q_width_ratio(self) -> float:
        return self.spec.q_dim / self.spec.d_model

    @property
    def kv_width_ratio(self) -> float:
        return self.spec.kv_dim / self.spec.d_model

    @property
    def attention_variant(self) -> str:
        if self.spec.n_kv_heads == self.spec.n_heads:
            return "mha"
        if self.spec.n_kv_heads == 1:
            return "mqa"
        return "gqa"

    def to_dict(self) -> dict[str, object]:
        return {
            "target_parameters": self.target_parameters,
            "exact_parameters": self.exact_parameters,
            "parameter_delta": self.parameter_delta,
            "relative_error": self.relative_error,
            "model_identity_sha256": self.model_identity_sha256,
            "attention_variant": self.attention_variant,
            "q_width": self.spec.q_dim,
            "kv_width": self.spec.kv_dim,
            "q_width_ratio": self.q_width_ratio,
            "kv_width_ratio": self.kv_width_ratio,
            "ffn_ratio": self.ffn_ratio,
            "model": self.spec.to_dict(),
        }


def _dense_affine_terms(template: DenseScalingTemplate) -> tuple[int, int]:
    """Return fixed and d_ff-linear terms for exact dense parameter accounting."""

    embedding = template.vocab_size * template.d_model
    attention = 2 * template.d_model * (template.q_dim + template.kv_dim)
    if template.attention_bias:
        attention += template.q_dim + 2 * template.kv_dim + template.d_model

    block_fixed = attention + 2 * template.d_model
    if template.mlp_bias:
        block_fixed += template.d_model

    fixed = embedding + template.n_layers * block_fixed
    if template.final_norm:
        fixed += template.d_model
    if not template.tie_word_embeddings:
        fixed += template.vocab_size * template.d_model
    if template.lm_head_bias:
        fixed += template.vocab_size

    slope_per_layer = 3 * template.d_model + (2 if template.mlp_bias else 0)
    slope = template.n_layers * slope_per_layer
    return fixed, slope


def _estimate_d_ff(target_parameters: int, template: DenseScalingTemplate) -> float:
    """Solve the exact affine parameter equation for d_ff."""

    fixed, slope = _dense_affine_terms(template)
    return (target_parameters - fixed) / slope


def solve_dense_scaling_candidates(
    target_parameters: int,
    templates: tuple[DenseScalingTemplate, ...],
    *,
    max_results: int = 8,
    max_relative_error: float = 0.05,
) -> tuple[DenseScalingCandidate, ...]:
    """Return deterministic exact-count candidates nearest the requested dense target."""

    if not isinstance(target_parameters, int) or isinstance(target_parameters, bool):
        raise TypeError("target_parameters must be an integer")
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if not templates:
        raise ValueError("at least one scaling template is required")
    if not isinstance(max_results, int) or isinstance(max_results, bool):
        raise TypeError("max_results must be an integer")
    if max_results <= 0:
        raise ValueError("max_results must be positive")
    if isinstance(max_relative_error, bool) or not isinstance(max_relative_error, (int, float)):
        raise TypeError("max_relative_error must be numeric")
    if not 0.0 <= max_relative_error < 1.0:
        raise ValueError("max_relative_error must be in [0, 1)")

    candidates: dict[str, DenseScalingCandidate] = {}
    for template in templates:
        raw_d_ff = _estimate_d_ff(target_parameters, template)
        if raw_d_ff <= 0:
            continue
        multiple = template.d_ff_multiple
        scaled = raw_d_ff / multiple
        nearby = {
            math.floor(scaled) * multiple,
            round(scaled) * multiple,
            math.ceil(scaled) * multiple,
        }
        for d_ff in nearby:
            if d_ff <= 0:
                continue
            spec = template.model_spec(d_ff)
            analytic = dense_parameter_breakdown(template, d_ff)
            if analytic.total != spec.parameter_count():
                raise RuntimeError("analytic parameter algebra drifted from ModelSpec")
            candidate = DenseScalingCandidate(target_parameters=target_parameters, spec=spec)
            if candidate.relative_error > max_relative_error:
                continue
            candidates[candidate.model_identity_sha256] = candidate

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.relative_error,
            abs(candidate.parameter_delta),
            candidate.exact_parameters,
            candidate.model_identity_sha256,
        ),
    )
    return tuple(ordered[:max_results])
