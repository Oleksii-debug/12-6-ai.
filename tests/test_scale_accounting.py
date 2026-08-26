from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.scale_accounting import (
    DecoderGeometry,
    build_report,
    dominant_matmul_flops_per_token,
    fixed_geometry_vocabulary_sweep,
    fixed_total_vocabulary_sweep,
    parameter_breakdown,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/research/r01_scale_vocab_accounting_v1.json").read_text(
        encoding="utf-8"
    )
)


def _model341() -> DecoderGeometry:
    return DecoderGeometry.from_mapping(CONFIG["authority"]["model341"]["geometry"])


def test_model341_exact_parameter_count_is_reproduced_without_torch() -> None:
    counts = parameter_breakdown(_model341())
    assert counts["token_embedding"] == 81_920
    assert counts["attention_weights_per_layer"] == 245_760
    assert counts["mlp_weights_per_layer"] == 1_036_800
    assert counts["block_per_layer"] == 1_283_200
    assert counts["total"] == 20_613_440


def test_gqa_head_divisibility_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DecoderGeometry(
            vocab_size=256,
            d_model=320,
            n_layers=16,
            n_heads=10,
            n_kv_heads=3,
            head_dim=32,
            d_ff=1080,
        )


def test_unknown_geometry_field_is_rejected() -> None:
    payload = dict(CONFIG["authority"]["model341"]["geometry"])
    payload["pretend_authorized"] = True
    with pytest.raises(ValueError, match="unknown geometry fields"):
        DecoderGeometry.from_mapping(payload)


def test_tied_vs_untied_embeddings_are_counted_exactly() -> None:
    tied = _model341()
    untied = DecoderGeometry(**{**tied.to_dict(), "tie_word_embeddings": False})
    tied_counts = parameter_breakdown(tied)
    untied_counts = parameter_breakdown(untied)
    assert untied_counts["total"] - tied_counts["total"] == 256 * 320
    assert untied_counts["untied_lm_head_weight"] == 256 * 320


def test_fixed_geometry_vocab_growth_changes_parameters_and_logits_compute() -> None:
    geometry = _model341()
    rows = fixed_geometry_vocabulary_sweep(
        geometry,
        [256, 512],
        sequence_length=1024,
    )
    baseline, larger = rows
    assert baseline["total_parameters"] == 20_613_440
    assert larger["delta_parameters_vs_baseline"] == 256 * 320
    assert (
        larger["forward_flops_per_token"] - baseline["forward_flops_per_token"]
        == 2 * 320 * 256
    )
    assert larger["transformer_body_parameters"] == baseline["transformer_body_parameters"]


def test_fixed_total_vocab_growth_reduces_transformer_mlp_capacity() -> None:
    geometry = _model341()
    rows = fixed_total_vocabulary_sweep(
        geometry,
        [256, 4096],
        target_parameters=20_613_440,
        d_ff_multiple=64,
        sequence_length=1024,
    )
    small_vocab, large_vocab = rows
    assert large_vocab["d_ff"] < small_vocab["d_ff"]
    assert large_vocab["vocabulary_parameters"] > small_vocab["vocabulary_parameters"]
    assert (
        large_vocab["transformer_body_parameters"]
        < small_vocab["transformer_body_parameters"]
    )


def test_context_cost_is_separate_from_parameter_count() -> None:
    geometry = _model341()
    short = dominant_matmul_flops_per_token(geometry, sequence_length=1024)
    long = dominant_matmul_flops_per_token(geometry, sequence_length=4096)
    assert short["attention_projection_forward"] == long["attention_projection_forward"]
    assert short["mlp_projection_forward"] == long["mlp_projection_forward"]
    assert short["vocabulary_projection_forward"] == long["vocabulary_projection_forward"]
    assert long["attention_context_forward"] > short["attention_context_forward"]
    assert parameter_breakdown(geometry)["total"] == 20_613_440


def test_candidate_only_50m_and_100m_surfaces_have_exact_known_counts() -> None:
    report = build_report(CONFIG)
    candidates = {
        item["candidate_id"]: item for item in report["candidate_geometries"]
    }
    assert (
        candidates["R01-50M-GQA-A"]["parameter_breakdown"]["total"]
        == 49_726_976
    )
    assert (
        candidates["R01-100M-GQA-A"]["parameter_breakdown"]["total"]
        == 99_753_216
    )
    assert (
        candidates["R01-100M-MHA-EXISTING-CONTROL"]["parameter_breakdown"]["total"]
        == 99_897_600
    )
    assert all(item["status"] == "PLANNING_CANDIDATE_ONLY" for item in candidates.values())
    assert report["truth_boundary"]["training_authorized"] is False
    assert report["truth_boundary"]["modelspec_frozen"] is False


def test_model341_parameter_drift_invalidates_report() -> None:
    config = copy.deepcopy(CONFIG)
    config["authority"]["model341"]["expected_parameters"] += 1
    with pytest.raises(ValueError, match="MODEL-341 exact parameter drift"):
        build_report(config)


def test_training_flop_estimate_does_not_claim_six_n_is_exact() -> None:
    flops = dominant_matmul_flops_per_token(_model341(), sequence_length=1024)
    assert flops["training_multiplier"] == 3
    assert flops["training_total_estimate"] != flops["six_n_parameter_proxy"]
    assert flops["training_minus_six_n"] != 0
