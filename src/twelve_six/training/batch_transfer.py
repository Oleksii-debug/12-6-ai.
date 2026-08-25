"""TRAIN-128 effective-batch transfer over the existing TRAIN-53 experiment path.

This module is orchestration only. It deliberately reuses TRAIN-53 real-source intake,
packing, Trainer/optimizer, TrainingObserver, validation and non-mutating gradient-noise
diagnostics rather than creating competing implementations.
"""
from __future__ import annotations

import math
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six.checkpoint import hash_json
from twelve_six.model import InitSpec, ModelSpec
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.s0_evidence_contract import validate_locked_environment_evidence

from .batch_noise_probe import (
    ACCUMULATION_CANDIDATES,
    BASE_LOSS_TOKENS,
    BATCH_SIZE_EXAMPLES,
    DIAGNOSTIC_SAMPLES,
    SEQUENCE_LENGTH,
    TOTAL_MICROBATCHES,
    _candidate_run,
    _real_corpus_records,
    _sample_indices,
    _tensor_batches_from_records,
    _trainer_config,
)

SCHEMA_VERSION = "12-6.train128-batch-transfer.v1"
AUTHORITY = "LOCAL_FREE_BATCH_TRANSFER_EVIDENCE_NOT_CRITICAL_BATCH_OR_PROMOTION"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _spec(*, d_model: int, n_layers: int, n_heads: int, head_dim: int, d_ff: int) -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        head_dim=head_dim,
        d_ff=d_ff,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=head_dim,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def transfer_specs() -> dict[str, ModelSpec]:
    """Return the already-established RESEARCH41 fixed-vocab ~500K/~1M members."""
    specs = {
        "500k": _spec(d_model=96, n_layers=4, n_heads=6, head_dim=16, d_ff=256),
        "1m": _spec(d_model=128, n_layers=5, n_heads=8, head_dim=16, d_ff=352),
    }
    expected = {"500k": 467_808, "1m": 1_037_696}
    counts = {name: spec.parameter_count() for name, spec in specs.items()}
    if counts != expected:
        raise RuntimeError(f"TRAIN-128 controlled geometry drift: {counts!r} != {expected!r}")
    return specs


def _final_point(candidate: Mapping[str, Any]) -> dict[str, Any]:
    final = candidate["checkpoints"][-1]
    return {
        "gradient_accumulation_steps": int(candidate["gradient_accumulation_steps"]),
        "effective_loss_tokens_per_update": int(candidate["effective_loss_tokens_per_update"]),
        "optimizer_steps": int(candidate["optimizer_steps"]),
        "optimized_tokens": int(candidate["optimized_tokens"]),
        "final_validation_loss": float(final["validation_loss"]),
        "final_validation_bpb": float(final["validation_loss"]) / math.log(2.0),
        "training_wall_seconds": float(candidate["training_wall_seconds"]),
        "optimized_tokens_per_training_second": float(candidate["optimized_tokens_per_training_second"]),
    }


def run_batch_transfer(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    """Run the fixed-real-corpus matched-token batch matrix at ~500K and ~1M."""
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise ValueError("source SHA must be a full lowercase Git SHA")
    root = Path(root).resolve()
    environment = validate_locked_environment_evidence(locked_environment_evidence, source_sha=source_sha)
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    if tokenizer.vocab_size != 256:
        raise RuntimeError("byte tokenizer vocabulary drift")

    init_spec = InitSpec()
    specs = transfer_specs()
    with tempfile.TemporaryDirectory(prefix="train128-real-intake-") as intake_tmp:
        train_records, validation_records, real_data = _real_corpus_records(root, Path(intake_tmp))
        train_batches = _tensor_batches_from_records(
            train_records,
            split="train",
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE_EXAMPLES,
            sequence_length=SEQUENCE_LENGTH,
            full_only=True,
        )
        validation_batches = _tensor_batches_from_records(
            validation_records,
            split="validation",
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE_EXAMPLES,
            sequence_length=SEQUENCE_LENGTH,
            full_only=False,
        )
        if len(train_batches) < DIAGNOSTIC_SAMPLES:
            raise RuntimeError("real corpus yields too few full microbatches")
        trace_indices = _sample_indices(
            seed=seed + 1_001,
            population=len(train_batches),
            count=TOTAL_MICROBATCHES,
        )

        matrices: dict[str, list[dict[str, Any]]] = {}
        for name, spec in specs.items():
            matrices[name] = [
                _candidate_run(
                    source_sha=source_sha,
                    environment=environment,
                    dataset_identity_sha256=real_data["dataset_identity_sha256"],
                    spec=spec,
                    init_spec=init_spec,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    trace_indices=trace_indices,
                    accumulation=accumulation,
                    seed=seed,
                    collect_diagnostics=accumulation in (1, 4),
                )
                for accumulation in ACCUMULATION_CANDIDATES
            ]

    measured = {name: [_final_point(candidate) for candidate in candidates] for name, candidates in matrices.items()}
    selected: dict[str, dict[str, Any]] = {}
    for name, points in measured.items():
        best_loss = min(point["final_validation_loss"] for point in points)
        eligible = [point for point in points if point["final_validation_loss"] <= best_loss * 1.005]
        selected[name] = min(eligible, key=lambda point: point["effective_loss_tokens_per_update"])

    # Both measured transfer points determine the exponent. If they select the same
    # batch, alpha is exactly zero; that is an empirical result, not a theory claim.
    p0, p1 = specs["500k"].parameter_count(), specs["1m"].parameter_count()
    b0 = selected["500k"]["effective_loss_tokens_per_update"]
    b1 = selected["1m"]["effective_loss_tokens_per_update"]
    alpha = math.log(b1 / b0) / math.log(p1 / p0) if b0 != b1 else 0.0

    identity = {
        "repository": "Oleksii-debug/12-6-ai.",
        "source_sha": source_sha,
        "schema_version": SCHEMA_VERSION,
        "model_identity_sha256": {name: spec.identity_sha256() for name, spec in specs.items()},
        "parameter_count": {name: spec.parameter_count() for name, spec in specs.items()},
        "dataset_identity_sha256": real_data["dataset_identity_sha256"],
        "training_trace_sha256": hash_json({"indices": trace_indices}),
        "seed": seed,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": hash_json(identity),
        "runtime": {"device": "cpu", "precision": "fp32", "torch_threads": torch_threads, "paid_compute": False},
        "data": real_data,
        "controls": {
            "tokenizer": "s0-byte-v1",
            "microbatch_examples": BATCH_SIZE_EXAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
            "microbatch_loss_tokens": BASE_LOSS_TOKENS,
            "accumulation_candidates": list(ACCUMULATION_CANDIDATES),
            "fixed_microbatches": TOTAL_MICROBATCHES,
            "fixed_optimized_loss_tokens": TOTAL_MICROBATCHES * BASE_LOSS_TOKENS,
            "optimizer_recipe": asdict(_trainer_config(accumulation=1, max_steps=TOTAL_MICROBATCHES, seed=seed)),
            "same_training_trace": True,
            "learning_rate_scaled_with_batch": False,
        },
        "models": {
            name: {
                "spec": specs[name].to_dict(),
                "parameter_count": specs[name].parameter_count(),
                "candidates": matrices[name],
                "final_comparison": measured[name],
                "selected_by_0_5pct_quality_rule": selected[name],
            }
            for name in specs
        },
        "heuristic": {
            "fit_form": "B(P)=B500k*(P/P500k)^alpha inside the measured 500K-1M interval",
            "b500k_loss_tokens": b0,
            "b1m_loss_tokens": b1,
            "alpha": alpha,
            "exact_theoretical_critical_batch_claim": False,
            "ten_million_preregistered_loss_token_candidates": [504, 1008],
            "ten_million_note": "Validate both before any larger-batch or LR-scaling decision.",
        },
        "claims": {
            "representative_broad_corpus": False,
            "paid_compute_used": False,
            "canonical_batch_default_changed": False,
            "exact_critical_batch_size": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    return report
