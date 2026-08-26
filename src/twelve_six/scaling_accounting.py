"""Exact parameter accounting and explicit dominant-matmul FLOP planning.

This module is intentionally stdlib-only and planning-only.  It does not build a
model, fit a tokenizer, allocate accelerators, or authorize training.

The FLOP estimator exposes components instead of presenting ``6 * N`` as exact.
It counts dense projection matmuls and the QK/AV attention-context matmuls.  It
omits elementwise activation/norm/softmax/RoPE work, embedding lookup, optimizer
updates, communication, and kernel inefficiency, so callers must treat it as a
planning estimate rather than measured runtime cost.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


class ScalingAccountingError(ValueError):
    """Raised when a decoder geometry cannot be accounted for safely."""


@dataclass(frozen=True)
class DecoderGeometry:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    final_norm: bool = True
    lm_head_bias: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecoderGeometry":
        required = (
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "d_ff",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ScalingAccountingError(f"missing geometry fields: {missing}")
        geometry = cls(
            vocab_size=int(value["vocab_size"]),
            max_seq_len=int(value["max_seq_len"]),
            d_model=int(value["d_model"]),
            n_layers=int(value["n_layers"]),
            n_heads=int(value["n_heads"]),
            n_kv_heads=int(value["n_kv_heads"]),
            head_dim=int(value["head_dim"]),
            d_ff=int(value["d_ff"]),
            tie_word_embeddings=bool(value.get("tie_word_embeddings", True)),
            attention_bias=bool(value.get("attention_bias", False)),
            mlp_bias=bool(value.get("mlp_bias", False)),
            final_norm=bool(value.get("final_norm", True)),
            lm_head_bias=bool(value.get("lm_head_bias", False)),
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "d_ff": self.d_ff,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ScalingAccountingError(f"{name} must be positive, got {value}")
        if self.n_heads * self.head_dim != self.d_model:
            raise ScalingAccountingError(
                "n_heads * head_dim must equal d_model for this decoder geometry"
            )
        if self.n_kv_heads > self.n_heads:
            raise ScalingAccountingError("n_kv_heads cannot exceed n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ScalingAccountingError("n_heads must be divisible by n_kv_heads for GQA")

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def parameter_breakdown(geometry: DecoderGeometry) -> dict[str, int]:
    """Return exact learned-parameter counts for the repository decoder geometry."""
    geometry.validate()
    d_model = geometry.d_model
    kv_dim = geometry.kv_dim

    token_embedding = geometry.vocab_size * d_model
    attention_weights_per_layer = (
        d_model * d_model  # Q
        + d_model * kv_dim  # K
        + d_model * kv_dim  # V
        + d_model * d_model  # O
    )
    attention_biases_per_layer = 0
    if geometry.attention_bias:
        attention_biases_per_layer = 2 * d_model + 2 * kv_dim

    # SwiGLU: gate, up, and down matrices.
    mlp_weights_per_layer = 3 * d_model * geometry.d_ff
    mlp_biases_per_layer = 0
    if geometry.mlp_bias:
        mlp_biases_per_layer = 2 * geometry.d_ff + d_model

    # Two pre-RMSNorm vectors per decoder block.
    norm_parameters_per_layer = 2 * d_model
    block_parameters_per_layer = (
        attention_weights_per_layer
        + attention_biases_per_layer
        + mlp_weights_per_layer
        + mlp_biases_per_layer
        + norm_parameters_per_layer
    )
    blocks_total = geometry.n_layers * block_parameters_per_layer
    final_norm = d_model if geometry.final_norm else 0
    untied_lm_head = 0 if geometry.tie_word_embeddings else geometry.vocab_size * d_model
    lm_head_bias = geometry.vocab_size if geometry.lm_head_bias else 0
    total = token_embedding + blocks_total + final_norm + untied_lm_head + lm_head_bias

    return {
        "token_embedding": token_embedding,
        "attention_weights_per_layer": attention_weights_per_layer,
        "attention_biases_per_layer": attention_biases_per_layer,
        "mlp_weights_per_layer": mlp_weights_per_layer,
        "mlp_biases_per_layer": mlp_biases_per_layer,
        "norm_parameters_per_layer": norm_parameters_per_layer,
        "block_parameters_per_layer": block_parameters_per_layer,
        "blocks_total": blocks_total,
        "final_norm": final_norm,
        "untied_lm_head": untied_lm_head,
        "lm_head_bias": lm_head_bias,
        "total_parameters": total,
    }


def dominant_matmul_flops_per_token(
    geometry: DecoderGeometry,
    *,
    context_tokens: int,
) -> dict[str, int]:
    """Estimate dominant forward/training matmul FLOPs for one token/query.

    ``context_tokens`` means the number of KV positions attended by the query.
    For a full causal sequence, callers should use an average effective context
    (roughly half the packed sequence length) when estimating sequence-average
    cost, rather than blindly substituting max_seq_len.
    """
    geometry.validate()
    if context_tokens <= 0:
        raise ScalingAccountingError("context_tokens must be positive")
    if context_tokens > geometry.max_seq_len:
        raise ScalingAccountingError(
            f"context_tokens {context_tokens} exceeds max_seq_len {geometry.max_seq_len}"
        )

    d_model = geometry.d_model
    kv_dim = geometry.kv_dim
    attention_projection_weights = 2 * d_model * d_model + 2 * d_model * kv_dim
    mlp_projection_weights = 3 * d_model * geometry.d_ff
    vocab_projection_weights = geometry.vocab_size * d_model

    attention_projection_forward = 2 * attention_projection_weights
    mlp_projection_forward = 2 * mlp_projection_weights
    vocab_projection_forward = 2 * vocab_projection_weights

    # Per query: QK^T and attention-weighted V each cost ~2*d_model*context.
    attention_context_forward = 4 * d_model * context_tokens
    forward_total = (
        attention_projection_forward
        + mlp_projection_forward
        + vocab_projection_forward
        + attention_context_forward
    )

    # Dense matmul training rule-of-thumb: forward + two backward matmuls.
    training_total = 3 * forward_total
    return {
        "context_tokens": context_tokens,
        "attention_projection_forward": attention_projection_forward,
        "attention_context_forward": attention_context_forward,
        "mlp_projection_forward": mlp_projection_forward,
        "vocab_projection_forward": vocab_projection_forward,
        "forward_dominant_matmul_total": forward_total,
        "training_dominant_matmul_estimate": training_total,
    }


def closest_d_ff_for_target(
    geometry: DecoderGeometry,
    *,
    target_parameters: int,
    multiple_of: int = 8,
) -> tuple[DecoderGeometry, int]:
    """Adjust only d_ff to approach a fixed total-parameter target.

    This makes the vocabulary-size confound visible: changing vocabulary at
    fixed geometry changes total parameters, while holding total parameters
    approximately fixed requires changing transformer capacity elsewhere.
    """
    geometry.validate()
    if target_parameters <= 0:
        raise ScalingAccountingError("target_parameters must be positive")
    if multiple_of <= 0:
        raise ScalingAccountingError("multiple_of must be positive")

    zero_ff = replace(geometry, d_ff=1)
    # Remove the contribution of d_ff=1 to recover the affine intercept while
    # keeping validation simple (d_ff itself is required to be positive).
    one_total = parameter_breakdown(zero_ff)["total_parameters"]
    slope = geometry.n_layers * 3 * geometry.d_model
    if geometry.mlp_bias:
        slope += geometry.n_layers * 2
    intercept = one_total - slope
    ideal = (target_parameters - intercept) / slope
    if ideal <= 0:
        raise ScalingAccountingError("target is too small for this fixed geometry")

    lower = max(multiple_of, int(ideal // multiple_of) * multiple_of)
    upper = lower + multiple_of
    candidates = [replace(geometry, d_ff=lower), replace(geometry, d_ff=upper)]
    best = min(
        candidates,
        key=lambda item: abs(parameter_breakdown(item)["total_parameters"] - target_parameters),
    )
    delta = parameter_breakdown(best)["total_parameters"] - target_parameters
    return best, delta


def geometry_as_dict(geometry: DecoderGeometry) -> dict[str, Any]:
    return {
        "vocab_size": geometry.vocab_size,
        "max_seq_len": geometry.max_seq_len,
        "d_model": geometry.d_model,
        "n_layers": geometry.n_layers,
        "n_heads": geometry.n_heads,
        "n_kv_heads": geometry.n_kv_heads,
        "head_dim": geometry.head_dim,
        "d_ff": geometry.d_ff,
        "tie_word_embeddings": geometry.tie_word_embeddings,
        "attention_bias": geometry.attention_bias,
        "mlp_bias": geometry.mlp_bias,
        "final_norm": geometry.final_norm,
        "lm_head_bias": geometry.lm_head_bias,
    }


def build_planning_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the fail-closed R01 accounting report from a machine-readable config."""
    if config.get("schema_version") != 1:
        raise ScalingAccountingError("unsupported accounting config schema")
    if config.get("planning_only") is not True:
        raise ScalingAccountingError("accounting config must remain planning_only=true")
    boundaries = config.get("hard_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ScalingAccountingError("hard_boundaries must be an object")
    forbidden_true = (
        "training_authorized",
        "paid_compute_authorized",
        "tokenizer_fit_authorized",
        "stage_promotion_authorized",
    )
    for key in forbidden_true:
        if boundaries.get(key) is not False:
            raise ScalingAccountingError(f"hard boundary must be false: {key}")

    baseline = DecoderGeometry.from_mapping(config["baseline_model"])
    expected = int(config["expected_baseline_parameters"])
    baseline_params = parameter_breakdown(baseline)
    if baseline_params["total_parameters"] != expected:
        raise ScalingAccountingError(
            "baseline parameter identity drift: "
            f"expected {expected}, got {baseline_params['total_parameters']}"
        )

    contexts = config.get("context_probe_tokens", [])
    if not isinstance(contexts, list) or not contexts:
        raise ScalingAccountingError("context_probe_tokens must be a non-empty list")
    context_reports = [
        dominant_matmul_flops_per_token(baseline, context_tokens=int(context))
        for context in contexts
    ]

    vocab_sizes = config.get("tokenizer_candidate_vocab_sizes", [])
    if not isinstance(vocab_sizes, list) or not vocab_sizes:
        raise ScalingAccountingError("tokenizer_candidate_vocab_sizes must be non-empty")
    fixed_geometry = []
    fixed_total = []
    for vocab_size_raw in vocab_sizes:
        vocab_size = int(vocab_size_raw)
        variant = replace(baseline, vocab_size=vocab_size)
        variant.validate()
        variant_params = parameter_breakdown(variant)
        variant_flops = dominant_matmul_flops_per_token(
            variant,
            context_tokens=int(contexts[-1]),
        )
        fixed_geometry.append(
            {
                "vocab_size": vocab_size,
                "d_ff": variant.d_ff,
                "total_parameters": variant_params["total_parameters"],
                "parameter_delta_vs_baseline": (
                    variant_params["total_parameters"] - expected
                ),
                "vocab_projection_forward_flops_per_token": variant_flops[
                    "vocab_projection_forward"
                ],
            }
        )
        adjusted, delta = closest_d_ff_for_target(
            variant,
            target_parameters=expected,
            multiple_of=int(config.get("d_ff_multiple_of", 8)),
        )
        fixed_total.append(
            {
                "vocab_size": vocab_size,
                "d_ff": adjusted.d_ff,
                "total_parameters": parameter_breakdown(adjusted)["total_parameters"],
                "delta_from_parameter_target": delta,
                "transformer_capacity_changed": adjusted.d_ff != baseline.d_ff,
            }
        )

    candidates = []
    for candidate in config.get("candidate_geometries", []):
        if not isinstance(candidate, Mapping):
            raise ScalingAccountingError("candidate_geometries entries must be objects")
        geometry = DecoderGeometry.from_mapping(candidate["model"])
        breakdown = parameter_breakdown(geometry)
        candidates.append(
            {
                "id": str(candidate["id"]),
                "target_parameters": int(candidate["target_parameters"]),
                "geometry": geometry_as_dict(geometry),
                "parameter_breakdown": breakdown,
                "target_delta": breakdown["total_parameters"]
                - int(candidate["target_parameters"]),
                "candidate_only": True,
                "model_spec_frozen": False,
                "training_authorized": False,
            }
        )

    return {
        "schema_version": 1,
        "report_id": "R01-FLOPS-VOCAB-ACCOUNTING-V1",
        "planning_only": True,
        "baseline": {
            "geometry": geometry_as_dict(baseline),
            "parameter_breakdown": baseline_params,
            "dominant_matmul_flops": context_reports,
        },
        "vocabulary_sensitivity": {
            "fixed_geometry": fixed_geometry,
            "fixed_total_parameters_via_d_ff_only": fixed_total,
        },
        "candidate_geometries": candidates,
        "truth_boundary": {
            "parameter_count_is_compute_cost": False,
            "flop_estimate_is_measured_runtime": False,
            "candidate_geometry_is_frozen_model_spec": False,
            "training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
