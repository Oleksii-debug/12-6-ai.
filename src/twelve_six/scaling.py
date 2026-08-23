from __future__ import annotations

import math
from dataclasses import dataclass

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
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    def model_spec(self, d_ff: int) -> ModelSpec:
        if d_ff <= 0 or d_ff % self.d_ff_multiple != 0:
            raise ValueError("d_ff must be a positive multiple of d_ff_multiple")
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
            rope_rotary_dim=self.head_dim,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            final_norm=True,
            tie_word_embeddings=True,
            lm_head_bias=False,
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

    def to_dict(self) -> dict[str, object]:
        return {
            "target_parameters": self.target_parameters,
            "exact_parameters": self.exact_parameters,
            "parameter_delta": self.parameter_delta,
            "relative_error": self.relative_error,
            "model_identity_sha256": self.model_identity_sha256,
            "model": self.spec.to_dict(),
        }


def _estimate_d_ff(target_parameters: int, template: DenseScalingTemplate) -> float:
    """Solve the affine parameter equation for d_ff under the template assumptions."""

    # All template-controlled terms except SwiGLU width.
    q_dim = template.n_heads * template.head_dim
    kv_dim = template.n_kv_heads * template.head_dim
    fixed = template.vocab_size * template.d_model
    fixed += template.n_layers * (
        2 * template.d_model * (q_dim + kv_dim) + 2 * template.d_model
    )
    fixed += template.d_model
    slope = 3 * template.d_model * template.n_layers
    return (target_parameters - fixed) / slope


def solve_dense_scaling_candidates(
    target_parameters: int,
    templates: tuple[DenseScalingTemplate, ...],
    *,
    max_results: int = 8,
    max_relative_error: float = 0.05,
) -> tuple[DenseScalingCandidate, ...]:
    """Return deterministic exact-count candidates nearest the requested dense target.

    The solver varies only SwiGLU width around the analytic optimum for each supplied
    architecture template. The caller owns the template search space and any later
    benchmark decision; this function never promotes or freezes a stage.
    """

    if not isinstance(target_parameters, int) or isinstance(target_parameters, bool):
        raise ValueError("target_parameters must be an integer")
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if not templates:
        raise ValueError("at least one scaling template is required")
    if max_results <= 0:
        raise ValueError("max_results must be positive")
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
