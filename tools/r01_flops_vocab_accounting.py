#!/usr/bin/env python3
"""Exact parameter accounting and planning-only FLOP decomposition for R01.

No training is performed. FLOP values are dominant-matmul planning estimates,
not hardware profiler measurements or material-compute authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class AccountingError(ValueError):
    """Raised when a decoder geometry or planning contract is invalid."""


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
                raise AccountingError(f"{name} must be a positive integer")
        if not isinstance(self.tie_word_embeddings, bool):
            raise AccountingError("tie_word_embeddings must be boolean")
        if self.d_model != self.n_heads * self.head_dim:
            raise AccountingError("d_model must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise AccountingError("n_heads must be divisible by n_kv_heads for GQA")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DecoderGeometry":
        geometry = cls(**value)
        geometry.validate()
        return geometry


def parameter_breakdown(geometry: DecoderGeometry) -> dict[str, int]:
    """Count repository decoder parameters exactly under the bound architecture.

    Assumptions: bias-free Q/K/V/O linears, bias-free SwiGLU gate/up/down
    linears, two RMSNorm scales per block, one final RMSNorm scale, RoPE with
    no learned parameters, and optionally tied token embedding / LM-head weights.
    """

    geometry.validate()
    d = geometry.d_model
    q_dim = geometry.n_heads * geometry.head_dim
    kv_dim = geometry.n_kv_heads * geometry.head_dim

    q_proj = d * q_dim
    k_proj = d * kv_dim
    v_proj = d * kv_dim
    o_proj = q_dim * d
    attention = q_proj + k_proj + v_proj + o_proj
    swiglu = 3 * d * geometry.d_ff
    block_norms = 2 * d
    block = attention + swiglu + block_norms
    all_blocks = geometry.n_layers * block
    token_embedding = geometry.vocab_size * d
    untied_lm_head = 0 if geometry.tie_word_embeddings else geometry.vocab_size * d
    final_rmsnorm = d
    transformer_non_vocab = all_blocks + final_rmsnorm
    vocabulary_parameters = token_embedding + untied_lm_head
    total = transformer_non_vocab + vocabulary_parameters

    return {
        "token_embedding": token_embedding,
        "q_proj_per_layer": q_proj,
        "k_proj_per_layer": k_proj,
        "v_proj_per_layer": v_proj,
        "o_proj_per_layer": o_proj,
        "attention_projection_per_layer": attention,
        "swiglu_mlp_per_layer": swiglu,
        "rmsnorm_per_layer": block_norms,
        "block_per_layer": block,
        "all_blocks": all_blocks,
        "final_rmsnorm": final_rmsnorm,
        "untied_lm_head": untied_lm_head,
        "transformer_non_vocab_parameters": transformer_non_vocab,
        "vocabulary_parameters": vocabulary_parameters,
        "total_parameters": total,
    }


def flop_breakdown(geometry: DecoderGeometry, *, seq_len: int | None = None) -> dict[str, int]:
    """Return dominant-matmul FLOP/token planning estimates.

    Convention: one multiply-add = 2 FLOPs. The attention-context term counts
    dense QK^T and AV work as 4 * layers * q_dim * sequence_length per token.
    Training is estimated as 3x forward dominant-matmul work (forward plus two
    backward matmul families). Real kernels, causal masking, fused operations,
    recomputation, optimizer work and hardware utilization require measurement.
    """

    geometry.validate()
    if seq_len is None:
        seq_len = geometry.max_seq_len
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
        raise AccountingError("seq_len must be a positive integer")
    if seq_len > geometry.max_seq_len:
        raise AccountingError("seq_len cannot exceed max_seq_len")

    p = parameter_breakdown(geometry)
    attention_projection_forward = 2 * geometry.n_layers * p["attention_projection_per_layer"]
    mlp_forward = 2 * geometry.n_layers * p["swiglu_mlp_per_layer"]
    transformer_projection_forward = attention_projection_forward + mlp_forward
    q_dim = geometry.n_heads * geometry.head_dim
    attention_context_forward = 4 * geometry.n_layers * q_dim * seq_len
    vocabulary_projection_forward = 2 * geometry.d_model * geometry.vocab_size
    forward_total = (
        transformer_projection_forward
        + attention_context_forward
        + vocabulary_projection_forward
    )
    training_estimate = 3 * forward_total
    six_n_proxy = 6 * p["total_parameters"]

    return {
        "sequence_length": seq_len,
        "attention_projection_forward_flops_per_token": attention_projection_forward,
        "mlp_forward_flops_per_token": mlp_forward,
        "transformer_projection_forward_flops_per_token": transformer_projection_forward,
        "attention_context_forward_flops_per_token": attention_context_forward,
        "vocabulary_projection_forward_flops_per_token": vocabulary_projection_forward,
        "forward_dominant_matmul_flops_per_token": forward_total,
        "training_dominant_matmul_estimate_flops_per_token": training_estimate,
        "six_n_training_proxy_flops_per_token": six_n_proxy,
        "training_estimate_minus_six_n": training_estimate - six_n_proxy,
    }


def fixed_geometry_vocab_sweep(
    geometry: DecoderGeometry,
    vocab_sizes: list[int],
    *,
    seq_len: int | None = None,
) -> list[dict[str, int]]:
    geometry.validate()
    baseline = parameter_breakdown(geometry)["total_parameters"]
    rows: list[dict[str, int]] = []
    for vocab_size in vocab_sizes:
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
            raise AccountingError("vocab sizes must be positive integers")
        candidate = replace(geometry, vocab_size=vocab_size)
        p = parameter_breakdown(candidate)
        f = flop_breakdown(candidate, seq_len=seq_len)
        rows.append(
            {
                "vocab_size": vocab_size,
                "total_parameters": p["total_parameters"],
                "parameter_delta_vs_baseline": p["total_parameters"] - baseline,
                "vocabulary_parameters": p["vocabulary_parameters"],
                "vocabulary_projection_forward_flops_per_token": f[
                    "vocabulary_projection_forward_flops_per_token"
                ],
                "forward_dominant_matmul_flops_per_token": f[
                    "forward_dominant_matmul_flops_per_token"
                ],
                "training_dominant_matmul_estimate_flops_per_token": f[
                    "training_dominant_matmul_estimate_flops_per_token"
                ],
            }
        )
    return rows


def nearest_dff_for_fixed_total(
    geometry: DecoderGeometry,
    *,
    target_total_parameters: int,
    vocab_size: int,
    d_ff_multiple: int = 1,
) -> dict[str, int | float]:
    """Hold all geometry except vocab+d_ff and find the nearest integer d_ff.

    This is an accounting aid only. It does not claim that changing d_ff creates
    a scientifically matched architecture because depth/width/optimizer dynamics
    remain independent experimental variables.
    """

    geometry.validate()
    for name, value in (
        ("target_total_parameters", target_total_parameters),
        ("vocab_size", vocab_size),
        ("d_ff_multiple", d_ff_multiple),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AccountingError(f"{name} must be a positive integer")

    d = geometry.d_model
    q_dim = geometry.n_heads * geometry.head_dim
    kv_dim = geometry.n_kv_heads * geometry.head_dim
    attention = d * q_dim + d * kv_dim + d * kv_dim + q_dim * d
    per_layer_non_mlp = attention + 2 * d
    embedding_multiplier = 1 if geometry.tie_word_embeddings else 2
    non_mlp_total = (
        geometry.n_layers * per_layer_non_mlp
        + d
        + embedding_multiplier * vocab_size * d
    )
    mlp_coefficient = geometry.n_layers * 3 * d
    raw_d_ff = (target_total_parameters - non_mlp_total) / mlp_coefficient
    if raw_d_ff <= 0:
        raise AccountingError("target leaves no positive d_ff capacity")

    rounded_units = max(1, round(raw_d_ff / d_ff_multiple))
    chosen_d_ff = rounded_units * d_ff_multiple
    candidate = replace(geometry, vocab_size=vocab_size, d_ff=chosen_d_ff)
    total = parameter_breakdown(candidate)["total_parameters"]
    return {
        "vocab_size": vocab_size,
        "target_total_parameters": target_total_parameters,
        "raw_d_ff": raw_d_ff,
        "chosen_d_ff": chosen_d_ff,
        "d_ff_multiple": d_ff_multiple,
        "result_total_parameters": total,
        "parameter_delta_from_target": total - target_total_parameters,
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "12-6.r01-flops-vocab-accounting.v1":
        raise AccountingError("unexpected accounting schema")
    authority = config.get("authority")
    if not isinstance(authority, dict):
        raise AccountingError("authority must be a mapping")
    if authority.get("main_sha") != "a73ab38026cb7849f478cc13ad58b93534a76e2f":
        raise AccountingError("main authority drift")
    if authority.get("model341_sha") != "e4ff486fd90802fc123bebf60eed4e59196a98df":
        raise AccountingError("MODEL-341 authority drift")
    if authority.get("model341_parameter_count") != 20_613_440:
        raise AccountingError("MODEL-341 parameter authority drift")

    claims = config.get("claims")
    expected_claims = {
        "training_authorized": False,
        "paid_compute_authorized": False,
        "tokenizer_fit_authorized": False,
        "model100m_frozen": False,
        "candidate_geometries_are_planning_only": True,
    }
    if claims != expected_claims:
        raise AccountingError("authority claims drifted")

    baseline = DecoderGeometry.from_mapping(config["baseline_geometry"])
    baseline_params = parameter_breakdown(baseline)
    if baseline_params["total_parameters"] != 20_613_440:
        raise AccountingError("baseline arithmetic does not reproduce MODEL-341")

    vocab_sizes = config.get("vocab_sensitivity_sizes")
    if not isinstance(vocab_sizes, list) or not vocab_sizes:
        raise AccountingError("vocab_sensitivity_sizes must be a non-empty list")

    candidates: list[dict[str, Any]] = []
    for item in config.get("candidate_geometries", []):
        if not isinstance(item, dict):
            raise AccountingError("candidate geometry entry must be a mapping")
        geometry = DecoderGeometry.from_mapping(item["geometry"])
        p = parameter_breakdown(geometry)
        lower, upper = item["parameter_band"]
        if not (lower <= p["total_parameters"] <= upper):
            raise AccountingError(f"candidate {item['id']} fell outside parameter band")
        candidates.append(
            {
                "id": item["id"],
                "status": "PLANNING_ONLY_NOT_MODELSPEC",
                "geometry": item["geometry"],
                "parameters": p,
                "flops_at_max_seq": flop_breakdown(geometry),
            }
        )

    fixed_total = [
        nearest_dff_for_fixed_total(
            baseline,
            target_total_parameters=20_613_440,
            vocab_size=vocab,
        )
        for vocab in vocab_sizes
    ]

    report = {
        "schema": "12-6.r01-flops-vocab-accounting-report.v1",
        "status": "PASS_PLANNING_ACCOUNTING_ONLY",
        "config_sha256": canonical_sha256(config),
        "baseline_parameters": baseline_params,
        "baseline_flops_at_max_seq": flop_breakdown(baseline),
        "fixed_geometry_vocab_sweep": fixed_geometry_vocab_sweep(baseline, vocab_sizes),
        "fixed_total_parameter_vocab_sweep": fixed_total,
        "candidate_geometries": candidates,
        "truth_boundary": expected_claims,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/research/r01_flops_vocab_accounting_v1.json"),
    )
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = build_report(config)
    except (OSError, json.JSONDecodeError, AccountingError, KeyError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
