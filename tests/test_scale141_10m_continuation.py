from __future__ import annotations

import math
from pathlib import Path

from twelve_six.scale141_10m_continuation import (
    EXPECTED_MODEL_SHA,
    EXPECTED_OPTIMIZED_TOKENS,
    EXPECTED_PARAMETERS,
    MAX_STEPS,
    TRAIN_CORPUS_BYTES,
    _model,
    _scaling_prediction,
    _select_cadence,
)


def test_fallback_budget_is_material_and_does_not_replay_data25() -> None:
    assert EXPECTED_OPTIMIZED_TOKENS == 2_046_000
    assert EXPECTED_OPTIMIZED_TOKENS > 1_000_000
    assert EXPECTED_OPTIMIZED_TOKENS < TRAIN_CORPUS_BYTES
    assert 0.10 < EXPECTED_OPTIMIZED_TOKENS / TRAIN_CORPUS_BYTES < 0.11
    assert MAX_STEPS == 2000


def test_exact_s3_geometry_and_prepared_optimizer_lineage() -> None:
    repo = Path(__file__).resolve().parents[1]
    spec, init, prepared = _model(repo)
    assert spec.parameter_count() == EXPECTED_PARAMETERS
    assert spec.identity_sha256() == EXPECTED_MODEL_SHA
    assert spec.vocab_size == 256
    assert spec.max_seq_len == 1024
    assert spec.n_layers == 12
    assert spec.n_heads == 8
    assert spec.n_kv_heads == 2
    assert prepared["training"]["optimizer"] == "AdamW"
    assert prepared["training"]["learning_rate"] == 3e-4
    assert prepared["training"]["weight_decay"] == 0.1
    assert init.identity_sha256() == prepared["candidate"].get("init_spec_sha256", init.identity_sha256())


def test_train56_cadence_selects_jointly_feasible_envelope() -> None:
    value = _select_cadence([0.9, 1.0, 1.1], [0.1, 0.12, 0.11])
    assert value["five_second_and_overhead_constraints_jointly_feasible"] is True
    assert value["checkpoint_every_optimizer_steps"] == 5
    assert value["projected_lost_work_seconds"] <= 5.0
    assert value["projected_checkpoint_overhead_fraction"] <= 0.05


def test_train56_cadence_reports_infeasible_joint_constraint_without_lying() -> None:
    value = _select_cadence([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    assert value["five_second_and_overhead_constraints_jointly_feasible"] is False
    assert value["minimum_overhead_bound_steps"] == 10
    assert value["checkpoint_every_optimizer_steps"] == 10
    assert value["projected_lost_work_seconds"] > 5.0
    assert value["projected_checkpoint_overhead_fraction"] <= 0.05


def test_scaling_prediction_is_explicitly_out_of_domain() -> None:
    prediction = _scaling_prediction(EXPECTED_OPTIMIZED_TOKENS)
    assert prediction is not None
    assert prediction["predicted_loss"] > 0.0
    assert math.isfinite(prediction["predicted_bits_per_byte"])
    assert prediction["fit"]["out_of_domain_for_10m"] is True
    assert prediction["interpretation"] == "OUT_OF_DOMAIN_DIAGNOSTIC_NOT_A_SUCCESS_GATE"
