from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.scaling_accounting import (
    DecoderGeometry,
    ScalingAccountingError,
    build_planning_report,
    closest_d_ff_for_target,
    dominant_matmul_flops_per_token,
    parameter_breakdown,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/r01_flops_vocab_accounting_v1.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _baseline() -> DecoderGeometry:
    return DecoderGeometry.from_mapping(_config()["baseline_model"])


def test_model341_exact_parameter_identity() -> None:
    breakdown = parameter_breakdown(_baseline())

    assert breakdown["token_embedding"] == 81_920
    assert breakdown["attention_weights_per_layer"] == 245_760
    assert breakdown["mlp_weights_per_layer"] == 1_036_800
    assert breakdown["norm_parameters_per_layer"] == 640
    assert breakdown["block_parameters_per_layer"] == 1_283_200
    assert breakdown["blocks_total"] == 20_531_200
    assert breakdown["final_norm"] == 320
    assert breakdown["total_parameters"] == 20_613_440


def test_untied_embeddings_are_counted_not_silently_ignored() -> None:
    geometry = _baseline()
    untied = replace(geometry, tie_word_embeddings=False)

    assert parameter_breakdown(untied)["total_parameters"] == (
        parameter_breakdown(geometry)["total_parameters"]
        + geometry.vocab_size * geometry.d_model
    )


def test_gqa_geometry_fails_closed_on_invalid_head_relationships() -> None:
    geometry = _baseline()

    with pytest.raises(ScalingAccountingError, match="n_heads \* head_dim"):
        replace(geometry, n_heads=8).validate()
    with pytest.raises(ScalingAccountingError, match="divisible"):
        replace(geometry, n_kv_heads=3).validate()


def test_dominant_flops_expose_context_and_vocab_costs() -> None:
    geometry = _baseline()
    report = dominant_matmul_flops_per_token(geometry, context_tokens=512)

    assert report["attention_projection_forward"] == 491_520
    assert report["attention_context_forward"] == 655_360
    assert report["mlp_projection_forward"] == 2_073_600
    assert report["vocab_projection_forward"] == 163_840
    assert report["forward_dominant_matmul_total"] == 3_384_320
    assert report["training_dominant_matmul_estimate"] == 10_152_960


def test_vocab_change_at_fixed_geometry_changes_parameters_and_output_compute() -> None:
    baseline = _baseline()
    larger_vocab = replace(baseline, vocab_size=512)

    baseline_params = parameter_breakdown(baseline)["total_parameters"]
    larger_params = parameter_breakdown(larger_vocab)["total_parameters"]
    assert larger_params - baseline_params == (512 - 256) * baseline.d_model

    baseline_flops = dominant_matmul_flops_per_token(baseline, context_tokens=512)
    larger_flops = dominant_matmul_flops_per_token(larger_vocab, context_tokens=512)
    assert larger_flops["vocab_projection_forward"] > baseline_flops[
        "vocab_projection_forward"
    ]


def test_fixed_parameter_comparison_requires_transformer_capacity_change() -> None:
    baseline = _baseline()
    larger_vocab = replace(baseline, vocab_size=512)
    adjusted, delta = closest_d_ff_for_target(
        larger_vocab,
        target_parameters=20_613_440,
        multiple_of=8,
    )

    assert adjusted.d_ff != baseline.d_ff
    assert adjusted.d_ff == 1072
    assert abs(delta) <= 61_440


def test_candidate_surfaces_are_close_to_target_but_never_frozen_or_authorized() -> None:
    report = build_planning_report(_config())
    candidates = {item["id"]: item for item in report["candidate_geometries"]}

    assert candidates["R01-CANDIDATE-50M-A"]["parameter_breakdown"][
        "total_parameters"
    ] == 50_009_472
    assert candidates["R01-CANDIDATE-100M-A"]["parameter_breakdown"][
        "total_parameters"
    ] == 99_998_080
    for candidate in candidates.values():
        assert candidate["candidate_only"] is True
        assert candidate["model_spec_frozen"] is False
        assert candidate["training_authorized"] is False


def test_report_fails_closed_on_model341_parameter_drift() -> None:
    config = copy.deepcopy(_config())
    config["expected_baseline_parameters"] = 20_000_000

    with pytest.raises(ScalingAccountingError, match="baseline parameter identity drift"):
        build_planning_report(config)


def test_report_fails_closed_if_planning_boundary_is_weakened() -> None:
    config = copy.deepcopy(_config())
    config["hard_boundaries"]["training_authorized"] = True

    with pytest.raises(ScalingAccountingError, match="hard boundary must be false"):
        build_planning_report(config)
