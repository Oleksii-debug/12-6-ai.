"""Stdlib-only parameter and dominant-matmul FLOP accounting for R01.

This module is planning infrastructure. It does not import torch, allocate tensors,
fit a tokenizer, mutate a model, or authorize training. FLOP estimates explicitly
separate dense projections, causal attention-context work, and vocabulary logits
so callers do not mistake a simple 6*N proxy for exact small-model compute.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "12-6.r01-scale-vocab-accounting.v1"
REPORT_SCHEMA = "12-6.r01-scale-vocab-accounting-report.v1"
DEFAULT_TRAINING_MULTIPLIER = 3


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class DecoderGeometry:
    """Tensor-free decoder geometry matching the repository's ModelSpec algebra."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    attention_bias: bool = False
    mlp_bias: bool = False
    final_norm: bool = True
    tie_word_embeddings: bool = True
    lm_head_bias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "d_ff",
        ):
            _positive_int(name, getattr(self, name))
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for the repository RoPE contract")

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DecoderGeometry":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown geometry fields: {sorted(unknown)}")
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parameter_breakdown(geometry: DecoderGeometry) -> dict[str, int]:
    """Return exact parameter counts using the repository decoder algebra."""
    d_model = geometry.d_model
    token_embedding = geometry.vocab_size * d_model

    q_weight = d_model * geometry.q_dim
    k_weight = d_model * geometry.kv_dim
    v_weight = d_model * geometry.kv_dim
    out_weight = geometry.q_dim * d_model
    attention_weights_per_layer = q_weight + k_weight + v_weight + out_weight

    attention_biases_per_layer = 0
    if geometry.attention_bias:
        attention_biases_per_layer = (
            geometry.q_dim + 2 * geometry.kv_dim + d_model
        )

    mlp_weights_per_layer = 3 * d_model * geometry.d_ff
    mlp_biases_per_layer = 0
    if geometry.mlp_bias:
        mlp_biases_per_layer = 2 * geometry.d_ff + d_model

    norms_per_layer = 2 * d_model
    block_per_layer = (
        attention_weights_per_layer
        + attention_biases_per_layer
        + mlp_weights_per_layer
        + mlp_biases_per_layer
        + norms_per_layer
    )
    blocks_total = geometry.n_layers * block_per_layer
    final_norm = d_model if geometry.final_norm else 0

    untied_lm_head_weight = 0
    if not geometry.tie_word_embeddings:
        untied_lm_head_weight = geometry.vocab_size * d_model
    lm_head_bias = geometry.vocab_size if geometry.lm_head_bias else 0
    lm_head_extra = untied_lm_head_weight + lm_head_bias

    vocabulary_parameters = token_embedding + lm_head_extra
    transformer_body_parameters = blocks_total + final_norm
    total = vocabulary_parameters + transformer_body_parameters

    return {
        "token_embedding": token_embedding,
        "attention_q_weight_per_layer": q_weight,
        "attention_k_weight_per_layer": k_weight,
        "attention_v_weight_per_layer": v_weight,
        "attention_out_weight_per_layer": out_weight,
        "attention_weights_per_layer": attention_weights_per_layer,
        "attention_biases_per_layer": attention_biases_per_layer,
        "mlp_weights_per_layer": mlp_weights_per_layer,
        "mlp_biases_per_layer": mlp_biases_per_layer,
        "norms_per_layer": norms_per_layer,
        "block_per_layer": block_per_layer,
        "blocks_total": blocks_total,
        "final_norm": final_norm,
        "untied_lm_head_weight": untied_lm_head_weight,
        "lm_head_bias": lm_head_bias,
        "lm_head_extra": lm_head_extra,
        "vocabulary_parameters": vocabulary_parameters,
        "transformer_body_parameters": transformer_body_parameters,
        "total": total,
    }


def dominant_matmul_flops_per_token(
    geometry: DecoderGeometry,
    *,
    sequence_length: int,
    training_multiplier: int = DEFAULT_TRAINING_MULTIPLIER,
) -> dict[str, int | str]:
    """Estimate dominant dense-matmul FLOPs per causal token.

    Convention:
    - one multiply-add is two FLOPs;
    - a fully occupied causal sequence is assumed, so each query sees an average
      of (L + 1) / 2 key/value positions;
    - QK^T and attention-value products are both included;
    - logits over the full vocabulary are included even when embeddings are tied;
    - softmax, RoPE, RMSNorm, SwiGLU elementwise work, optimizer work, data I/O,
      communication, checkpointing, and padding inefficiency are excluded;
    - training is a planning estimate equal to ``training_multiplier`` times the
      forward dominant-matmul total, not an exact hardware/runtime measurement.
    """
    sequence_length = _positive_int("sequence_length", sequence_length)
    training_multiplier = _positive_int("training_multiplier", training_multiplier)
    counts = parameter_breakdown(geometry)

    attention_projection = (
        2 * counts["attention_weights_per_layer"] * geometry.n_layers
    )
    mlp_projection = 2 * counts["mlp_weights_per_layer"] * geometry.n_layers

    # Exact triangular-pair average for a fully occupied causal sequence:
    # two matmuls (QK^T and A@V), 2 FLOPs/MAC, across q_dim channels.
    attention_context = (
        2 * geometry.q_dim * (sequence_length + 1) * geometry.n_layers
    )

    # Embedding lookup is not a dense matmul, but producing training logits is.
    vocabulary_projection = 2 * geometry.d_model * geometry.vocab_size

    forward_total = (
        attention_projection
        + mlp_projection
        + attention_context
        + vocabulary_projection
    )
    training_total = training_multiplier * forward_total
    six_n_proxy = 6 * counts["total"]

    return {
        "convention": "dominant_matmul_2flop_mac_full_causal_v1",
        "sequence_length": sequence_length,
        "attention_projection_forward": attention_projection,
        "mlp_projection_forward": mlp_projection,
        "attention_context_forward": attention_context,
        "vocabulary_projection_forward": vocabulary_projection,
        "forward_total": forward_total,
        "training_multiplier": training_multiplier,
        "training_total_estimate": training_total,
        "six_n_parameter_proxy": six_n_proxy,
        "training_minus_six_n": training_total - six_n_proxy,
    }


def _unique_positive_ints(values: Iterable[object], *, name: str) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = _positive_int(name, value)
        if item not in seen:
            result.append(item)
            seen.add(item)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def fixed_geometry_vocabulary_sweep(
    geometry: DecoderGeometry,
    vocabulary_sizes: Iterable[object],
    *,
    sequence_length: int,
) -> list[dict[str, Any]]:
    """Change only vocabulary size and expose resulting capacity/compute drift."""
    baseline = parameter_breakdown(geometry)
    rows: list[dict[str, Any]] = []
    for vocab_size in _unique_positive_ints(vocabulary_sizes, name="vocab_size"):
        candidate = replace(geometry, vocab_size=vocab_size)
        counts = parameter_breakdown(candidate)
        flops = dominant_matmul_flops_per_token(
            candidate, sequence_length=sequence_length
        )
        rows.append(
            {
                "vocab_size": vocab_size,
                "d_ff": candidate.d_ff,
                "total_parameters": counts["total"],
                "delta_parameters_vs_baseline": counts["total"] - baseline["total"],
                "vocabulary_parameters": counts["vocabulary_parameters"],
                "transformer_body_parameters": counts["transformer_body_parameters"],
                "vocabulary_parameter_share_ppm": (
                    counts["vocabulary_parameters"] * 1_000_000 // counts["total"]
                ),
                "forward_flops_per_token": flops["forward_total"],
                "training_flops_per_token_estimate": flops["training_total_estimate"],
            }
        )
    return rows


def _nearest_d_ff_for_target(
    geometry: DecoderGeometry,
    *,
    target_parameters: int,
    d_ff_multiple: int,
) -> DecoderGeometry:
    target_parameters = _positive_int("target_parameters", target_parameters)
    d_ff_multiple = _positive_int("d_ff_multiple", d_ff_multiple)

    # Parameter count is affine in d_ff. Derive the exact integer coefficient
    # instead of embedding a second hand-maintained formula.
    at_one = parameter_breakdown(replace(geometry, d_ff=1))["total"]
    at_two = parameter_breakdown(replace(geometry, d_ff=2))["total"]
    coefficient = at_two - at_one
    intercept = at_one - coefficient
    if coefficient <= 0:
        raise AssertionError("d_ff parameter coefficient must be positive")

    numerator = target_parameters - intercept
    floor_multiple = max(1, numerator // (coefficient * d_ff_multiple))
    multiples = {
        max(1, floor_multiple - 1),
        floor_multiple,
        floor_multiple + 1,
        floor_multiple + 2,
    }
    candidates = [
        replace(geometry, d_ff=multiple * d_ff_multiple) for multiple in multiples
    ]
    return min(
        candidates,
        key=lambda item: (
            abs(parameter_breakdown(item)["total"] - target_parameters),
            parameter_breakdown(item)["total"] > target_parameters,
            item.d_ff,
        ),
    )


def fixed_total_vocabulary_sweep(
    geometry: DecoderGeometry,
    vocabulary_sizes: Iterable[object],
    *,
    target_parameters: int,
    d_ff_multiple: int,
    sequence_length: int,
) -> list[dict[str, Any]]:
    """Hold the total parameter target approximately fixed by adjusting d_ff.

    This deliberately demonstrates the confound: a larger vocabulary forces a
    smaller transformer MLP budget when total parameters are held fixed.
    """
    target_parameters = _positive_int("target_parameters", target_parameters)
    rows: list[dict[str, Any]] = []
    for vocab_size in _unique_positive_ints(vocabulary_sizes, name="vocab_size"):
        vocab_geometry = replace(geometry, vocab_size=vocab_size)
        candidate = _nearest_d_ff_for_target(
            vocab_geometry,
            target_parameters=target_parameters,
            d_ff_multiple=d_ff_multiple,
        )
        counts = parameter_breakdown(candidate)
        flops = dominant_matmul_flops_per_token(
            candidate, sequence_length=sequence_length
        )
        rows.append(
            {
                "vocab_size": vocab_size,
                "d_ff": candidate.d_ff,
                "total_parameters": counts["total"],
                "parameter_error_vs_target": counts["total"] - target_parameters,
                "vocabulary_parameters": counts["vocabulary_parameters"],
                "transformer_body_parameters": counts["transformer_body_parameters"],
                "forward_flops_per_token": flops["forward_total"],
                "training_flops_per_token_estimate": flops["training_total_estimate"],
            }
        )
    return rows


def geometry_report(
    geometry: DecoderGeometry,
    *,
    target_parameters: int | None,
    vocabulary_sizes: Iterable[object],
    context_lengths: Iterable[object],
    d_ff_multiple: int,
) -> dict[str, Any]:
    contexts = _unique_positive_ints(context_lengths, name="context_length")
    counts = parameter_breakdown(geometry)
    report: dict[str, Any] = {
        "geometry": geometry.to_dict(),
        "parameter_breakdown": counts,
        "fixed_geometry_vocabulary_sweep": fixed_geometry_vocabulary_sweep(
            geometry,
            vocabulary_sizes,
            sequence_length=contexts[0],
        ),
        "flops_by_context": {
            str(context): dominant_matmul_flops_per_token(
                geometry, sequence_length=context
            )
            for context in contexts
        },
    }
    if target_parameters is not None:
        report["target_parameters"] = _positive_int(
            "target_parameters", target_parameters
        )
        report["parameter_error_vs_target"] = counts["total"] - target_parameters
        report["fixed_total_parameter_vocabulary_sweep"] = (
            fixed_total_vocabulary_sweep(
                geometry,
                vocabulary_sizes,
                target_parameters=target_parameters,
                d_ff_multiple=d_ff_multiple,
                sequence_length=contexts[0],
            )
        )
    return report


def build_report(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")

    authority = config.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError("authority must be an object")
    model341 = authority.get("model341")
    if not isinstance(model341, Mapping):
        raise ValueError("authority.model341 must be an object")
    geometry_payload = model341.get("geometry")
    if not isinstance(geometry_payload, Mapping):
        raise ValueError("authority.model341.geometry must be an object")
    incumbent = DecoderGeometry.from_mapping(geometry_payload)
    expected_parameters = _positive_int(
        "authority.model341.expected_parameters", model341.get("expected_parameters")
    )
    observed_parameters = parameter_breakdown(incumbent)["total"]
    if observed_parameters != expected_parameters:
        raise ValueError(
            "MODEL-341 exact parameter drift: "
            f"expected {expected_parameters}, observed {observed_parameters}"
        )

    vocabulary_sizes = config.get("vocabulary_sizes")
    context_lengths = config.get("context_lengths")
    if not isinstance(vocabulary_sizes, list):
        raise ValueError("vocabulary_sizes must be a list")
    if not isinstance(context_lengths, list):
        raise ValueError("context_lengths must be a list")
    d_ff_multiple = _positive_int("d_ff_multiple", config.get("d_ff_multiple"))

    candidate_payloads = config.get("candidate_geometries")
    if not isinstance(candidate_payloads, list) or not candidate_payloads:
        raise ValueError("candidate_geometries must be a non-empty list")
    candidate_reports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for payload in candidate_payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("candidate geometry entries must be objects")
        candidate_id = payload.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        geometry_data = payload.get("geometry")
        if not isinstance(geometry_data, Mapping):
            raise ValueError(f"{candidate_id}.geometry must be an object")
        geometry = DecoderGeometry.from_mapping(geometry_data)
        target = _positive_int(
            f"{candidate_id}.target_parameters", payload.get("target_parameters")
        )
        candidate_reports.append(
            {
                "candidate_id": candidate_id,
                "status": "PLANNING_CANDIDATE_ONLY",
                **geometry_report(
                    geometry,
                    target_parameters=target,
                    vocabulary_sizes=vocabulary_sizes,
                    context_lengths=context_lengths,
                    d_ff_multiple=d_ff_multiple,
                ),
            }
        )

    return {
        "schema": REPORT_SCHEMA,
        "truth_boundary": {
            "planning_only": True,
            "tokenizer_fit_performed": False,
            "training_authorized": False,
            "modelspec_frozen": False,
            "parameter_count_is_not_compute_equivalence": True,
        },
        "authority": dict(authority),
        "model341": geometry_report(
            incumbent,
            target_parameters=expected_parameters,
            vocabulary_sizes=vocabulary_sizes,
            context_lengths=context_lengths,
            d_ff_multiple=d_ff_multiple,
        ),
        "candidate_geometries": candidate_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/research/r01_scale_vocab_accounting_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config root must be an object")
    report = build_report(payload)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
