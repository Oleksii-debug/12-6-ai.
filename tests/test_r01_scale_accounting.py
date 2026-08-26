from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.scale_accounting import (
    DecoderScaleSpec,
    ScaleAccountingError,
    dominant_matmul_flops_per_token,
    parameter_breakdown,
    vocabulary_sensitivity,
)


BASELINE = DecoderScaleSpec(
    vocab_size=256,
    max_seq_len=1024,
    d_model=320,
    n_layers=16,
    n_heads=10,
    n_kv_heads=2,
    head_dim=32,
    d_ff=1080,
    tie_word_embeddings=True,
)


def test_model341_parameter_formula_is_exact() -> None:
    assert parameter_breakdown(BASELINE)["total"] == 20_613_440


def test_gqa_head_divisibility_fails_closed() -> None:
    broken = DecoderScaleSpec(
        vocab_size=256,
        max_seq_len=1024,
        d_model=320,
        n_layers=16,
        n_heads=10,
        n_kv_heads=4,
        head_dim=32,
        d_ff=1080,
    )
    with pytest.raises(ScaleAccountingError, match="divisible"):
        broken.validate()


def test_untied_embeddings_add_exact_output_matrix() -> None:
    untied = DecoderScaleSpec(**{**BASELINE.__dict__, "tie_word_embeddings": False})
    tied_total = parameter_breakdown(BASELINE)["total"]
    untied_total = parameter_breakdown(untied)["total"]
    assert untied_total - tied_total == BASELINE.vocab_size * BASELINE.d_model


def test_vocabulary_sweep_exposes_fixed_geometry_parameter_drift() -> None:
    rows = vocabulary_sensitivity(BASELINE, [256, 8192])
    expected_delta = (8192 - 256) * BASELINE.d_model
    assert rows[1]["total_parameters"] - rows[0]["total_parameters"] == expected_delta
    assert rows[1]["embedding_fraction"] > 0.10


def test_context_aware_planning_compute_is_not_six_n_identity() -> None:
    flops = dominant_matmul_flops_per_token(BASELINE)
    assert flops["attention_context_forward"] > 0
    assert flops["training_to_six_n_ratio"] > 1.4


def test_planning_config_candidates_stay_within_one_percent() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/research/r01_scale_accounting_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for row in config["candidate_geometries"]:
        spec = DecoderScaleSpec.from_mapping(row["spec"])
        total = parameter_breakdown(spec)["total"]
        target = row["target_parameters"]
        assert abs(total - target) / target <= 0.01
