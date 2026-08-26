"""Exact parameter and planning FLOP accounting for 12-6 decoder candidates.

The parameter formula matches the repository decoder architecture: bias-free GQA,
SwiGLU, two pre-RMSNorm weights per block, one final RMSNorm, and optional tied
input/output embeddings. FLOP estimates intentionally cover dominant matrix
multiplications only; they are planning metrics, not hardware-runtime claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ScaleAccountingError(ValueError):
    """Raised when a proposed scale geometry is internally inconsistent."""


@dataclass(frozen=True)
class DecoderScaleSpec:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    tie_word_embeddings: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DecoderScaleSpec":
        required = {
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "d_ff",
            "tie_word_embeddings",
        }
        missing = sorted(required - set(value))
        unexpected = sorted(set(value) - required)
        if missing or unexpected:
            raise ScaleAccountingError(
                f"scale spec keys differ: missing={missing}, unexpected={unexpected}"
            )
        spec = cls(**value)
        spec.validate()
        return spec

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "d_ff",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ScaleAccountingError(f"{name} must be a positive integer")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ScaleAccountingError("tie_word_embeddings must be boolean")
        if self.n_heads * self.head_dim != self.d_model:
            raise ScaleAccountingError("n_heads * head_dim must equal d_model")
        if self.n_kv_heads > self.n_heads:
            raise ScaleAccountingError("n_kv_heads cannot exceed n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ScaleAccountingError("n_heads must be divisible by n_kv_heads for GQA")


def parameter_breakdown(spec: DecoderScaleSpec) -> dict[str, int]:
    """Return the exact parameter breakdown for the repository decoder geometry."""

    spec.validate()
    kv_width = spec.n_kv_heads * spec.head_dim
    attention_per_layer = (
        spec.d_model * spec.d_model
        + 2 * spec.d_model * kv_width
        + spec.d_model * spec.d_model
    )
    mlp_per_layer = 3 * spec.d_model * spec.d_ff
    norm_per_layer = 2 * spec.d_model
    transformer = spec.n_layers * (
        attention_per_layer + mlp_per_layer + norm_per_layer
    )
    input_embeddings = spec.vocab_size * spec.d_model
    output_embeddings = 0 if spec.tie_word_embeddings else input_embeddings
    final_norm = spec.d_model
    total = transformer + input_embeddings + output_embeddings + final_norm
    return {
        "attention": spec.n_layers * attention_per_layer,
        "mlp": spec.n_layers * mlp_per_layer,
        "block_norms": spec.n_layers * norm_per_layer,
        "input_embeddings": input_embeddings,
        "output_embeddings": output_embeddings,
        "final_norm": final_norm,
        "transformer_non_embedding": transformer + final_norm,
        "total": total,
    }


def dominant_matmul_flops_per_token(spec: DecoderScaleSpec) -> dict[str, float]:
    """Estimate dominant matmul FLOPs/token and expose the source of each cost.

    Attention context uses a dense full-context QK + AV estimate of
    ``4 * sequence_length * d_model`` per layer per token. Training is reported
    as three times the forward dominant-matmul estimate (forward + backward).
    Norm, activation, softmax, optimizer, communication, and kernel overhead are
    intentionally excluded, so this is not a wall-clock estimator.
    """

    spec.validate()
    kv_width = spec.n_kv_heads * spec.head_dim
    projection_weights_per_layer = (
        2 * spec.d_model * spec.d_model
        + 2 * spec.d_model * kv_width
        + 3 * spec.d_model * spec.d_ff
    )
    transformer_projection_forward = (
        2 * spec.n_layers * projection_weights_per_layer
    )
    attention_context_forward = (
        4 * spec.n_layers * spec.max_seq_len * spec.d_model
    )
    vocabulary_projection_forward = 2 * spec.d_model * spec.vocab_size
    forward = (
        transformer_projection_forward
        + attention_context_forward
        + vocabulary_projection_forward
    )
    training = 3 * forward
    total_parameters = parameter_breakdown(spec)["total"]
    six_n_proxy = 6 * total_parameters
    return {
        "transformer_projection_forward": float(transformer_projection_forward),
        "attention_context_forward": float(attention_context_forward),
        "vocabulary_projection_forward": float(vocabulary_projection_forward),
        "dominant_forward_total": float(forward),
        "dominant_training_total": float(training),
        "six_n_parameter_proxy": float(six_n_proxy),
        "training_to_six_n_ratio": training / six_n_proxy,
    }


def vocabulary_sensitivity(
    baseline: DecoderScaleSpec,
    vocab_sizes: list[int],
) -> list[dict[str, int | float]]:
    """Measure fixed-geometry parameter/FLOP drift across vocabulary sizes."""

    baseline.validate()
    rows: list[dict[str, int | float]] = []
    for vocab_size in vocab_sizes:
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
            raise ScaleAccountingError("vocabulary sweep values must be positive integers")
        candidate = DecoderScaleSpec(
            vocab_size=vocab_size,
            max_seq_len=baseline.max_seq_len,
            d_model=baseline.d_model,
            n_layers=baseline.n_layers,
            n_heads=baseline.n_heads,
            n_kv_heads=baseline.n_kv_heads,
            head_dim=baseline.head_dim,
            d_ff=baseline.d_ff,
            tie_word_embeddings=baseline.tie_word_embeddings,
        )
        params = parameter_breakdown(candidate)
        flops = dominant_matmul_flops_per_token(candidate)
        rows.append(
            {
                "vocab_size": vocab_size,
                "total_parameters": params["total"],
                "embedding_parameters": (
                    params["input_embeddings"] + params["output_embeddings"]
                ),
                "embedding_fraction": (
                    params["input_embeddings"] + params["output_embeddings"]
                )
                / params["total"],
                "dominant_training_flops_per_token": flops["dominant_training_total"],
            }
        )
    return rows


def assess_candidate(
    spec: DecoderScaleSpec,
    *,
    target_parameters: int | None = None,
) -> dict[str, Any]:
    """Return one self-contained planning record for a decoder candidate."""

    params = parameter_breakdown(spec)
    flops = dominant_matmul_flops_per_token(spec)
    record: dict[str, Any] = {
        "spec": {
            "vocab_size": spec.vocab_size,
            "max_seq_len": spec.max_seq_len,
            "d_model": spec.d_model,
            "n_layers": spec.n_layers,
            "n_heads": spec.n_heads,
            "n_kv_heads": spec.n_kv_heads,
            "head_dim": spec.head_dim,
            "d_ff": spec.d_ff,
            "tie_word_embeddings": spec.tie_word_embeddings,
        },
        "parameters": params,
        "dominant_matmul_flops_per_token": flops,
    }
    if target_parameters is not None:
        if (
            not isinstance(target_parameters, int)
            or isinstance(target_parameters, bool)
            or target_parameters <= 0
        ):
            raise ScaleAccountingError("target_parameters must be a positive integer")
        record["target_parameters"] = target_parameters
        record["target_relative_error"] = (
            params["total"] - target_parameters
        ) / target_parameters
    return record
