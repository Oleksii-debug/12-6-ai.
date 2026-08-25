from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "evidence" / "research138" / "observed_experiments.json"
SPEC = importlib.util.spec_from_file_location("research138_scaling_fit", ROOT / "tools" / "research138_scaling_fit.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TARGET_PARAMETERS = MODULE.TARGET_PARAMETERS
_validate_input = MODULE._validate_input
build_report = MODULE.build_report


def _input() -> dict:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def test_observed_experiment_identity_is_hash_bound_and_balanced() -> None:
    doc = _input()
    _validate_input(doc)
    core = doc["core_fit_observations"]
    assert len(core) == 12
    assert sorted({row["parameters"] for row in core}) == [95568, 267912, 467808, 1037696]
    assert sorted({row["optimized_tokens"] for row in core}) == [4284, 16632, 65772]
    assert doc["shared_identity"]["tokenizer_id"] == "s0-byte-v1"
    assert doc["shared_identity"]["context"] == 256


def test_leave_one_scale_out_selects_different_interpolation_and_extrapolation_forms() -> None:
    report = build_report(_input())
    assert report["model_selection"]["best_average_loso_interpolator"] == "linear_log"
    assert report["model_selection"]["selected_extrapolator"] == "log_power"
    linear = report["candidate_models"]["linear_log"]
    power = report["candidate_models"]["log_power"]
    assert linear["loso"]["loso_rmse_nats"] == pytest.approx(0.164509, abs=1e-6)
    assert power["loso"]["largest_scale_holdout_rmse_nats"] == pytest.approx(0.073429, abs=1e-6)
    assert linear["extrapolation_admissible"] is False
    assert power["extrapolation_admissible"] is True
    assert len(power["loso"]["folds"]) == 4


def test_long_horizon_stress_is_larger_than_seed_noise() -> None:
    stress = build_report(_input())["same_identity_stress_diagnostics"]
    assert stress["beyond_core_max_abs_residual_nats"] > 1.2
    assert stress["max_seed_delta_nats"] < 0.16
    assert stress["beyond_core_max_abs_residual_nats"] > 5 * stress["max_seed_delta_nats"]


def test_10m_predictions_are_wide_extrapolation_stress_bands() -> None:
    report = build_report(_input())
    target = report["predictions_10m_parameter_target"]
    assert target["target_parameters"] == TARGET_PARAMETERS == 10_000_640
    assert target["applicability_to_current_s3_gqa_context1024"] is False
    predictions = target["predictions"]
    assert [row["optimized_tokens"] for row in predictions] == [16632, 65772, 131292, 262332]
    assert all(row["coverage_guarantee"] is False for row in predictions)
    assert predictions[-1]["empirical_90_interval_bpb"][0] == 0.0
    assert predictions[-1]["empirical_90_interval_bpb"][1] > 5.7
    assert predictions[-1]["long_horizon_structural_penalty_nats"] > 1.2


def test_next_experiment_bridges_scale_instead_of_jumping_to_10m() -> None:
    recommendation = build_report(_input())["next_experiment"]
    assert 3_000_000 < recommendation["ideal_parameters"] < 3_500_000
    assert recommendation["ideal_parameters"] < TARGET_PARAMETERS
    assert recommendation["final_optimized_tokens"] == 131292
    assert recommendation["heldout_checkpoints"] == [16632, 65772, 131292]
