from __future__ import annotations

import math

import pytest

from twelve_six.scaling_experiment import (
    AUTHORITY,
    SCHEMA,
    TOKENIZER_ID,
    _canonical_hash,
    _fit_log_plane,
    _make_batch,
    controlled_specs,
    validate_report,
)


def test_controlled_specs_hold_tokenizer_context_and_expected_sizes() -> None:
    specs = controlled_specs()
    assert [spec.parameter_count() for spec in specs] == [
        95_568,
        267_912,
        467_808,
        1_037_696,
    ]
    assert {spec.vocab_size for spec in specs} == {256}
    assert {spec.max_seq_len for spec in specs} == {256}


def test_research_batch_schedule_is_deterministic_and_byte_bounded() -> None:
    stream = "English\nУкраїнська\n".encode()
    first = _make_batch(stream, step=7, batch_size=4, sequence_length=16)
    second = _make_batch(stream, step=7, batch_size=4, sequence_length=16)
    assert first.equal(second)
    assert first.shape == (4, 16)
    assert first.min().item() >= 0
    assert first.max().item() <= 255


def test_log_plane_recovers_known_local_relationship() -> None:
    points = []
    for parameters in (100_000, 300_000, 1_000_000):
        for tokens in (4_000, 16_000, 64_000):
            loss = math.exp(5.0) * parameters**-0.12 * tokens**-0.27
            points.append(
                {
                    "parameters": parameters,
                    "optimized_tokens": tokens,
                    "validation_loss": loss,
                }
            )
    fit = _fit_log_plane(points)
    coefficients = fit["coefficients"]
    assert coefficients["b0"] == pytest.approx(5.0, abs=1e-10)
    assert coefficients["b_parameters"] == pytest.approx(-0.12, abs=1e-10)
    assert coefficients["b_tokens"] == pytest.approx(-0.27, abs=1e-10)
    assert fit["r_squared_log_space"] == pytest.approx(1.0, abs=1e-12)
    assert fit["extrapolation_authorized"] is False


def _valid_report() -> dict:
    parameter_counts = (95_568, 267_912, 467_808, 1_037_696)
    budgets = (4_096, 16_384, 65_536)
    observations = []
    for parameters in parameter_counts:
        for budget in budgets:
            optimized_tokens = budget + 188
            observations.append(
                {
                    "parameters": parameters,
                    "requested_token_budget": budget,
                    "optimized_tokens": optimized_tokens,
                    "optimizer_steps": 1,
                    "compute_proxy": 6 * parameters * optimized_tokens,
                    "validation_loss": 2.0,
                    "last_train_loss": 2.0,
                    "last_grad_norm": 1.0,
                }
            )
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": "a" * 40,
        },
        "runtime": {"paid_compute": False},
        "controls": {
            "tokenizer_id": TOKENIZER_ID,
            "vocab_size": 256,
            "model_max_seq_len": 256,
        },
        "data": {"train_validation_record_overlap": []},
        "init_spec": {},
        "init_identity_sha256": "b" * 64,
        "model_runs": [],
        "observations": observations,
        "fit": {},
        "decision_signals": {},
        "truth_boundary": {
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
            "one_trillion_parameter_extrapolation": False,
            "euro_optimum_claim": False,
            "fit_valid_only_inside_observed_box": True,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def test_report_validator_rejects_rehashed_overclaim() -> None:
    report = _valid_report()
    validate_report(report, expected_source_sha="a" * 40)
    report["truth_boundary"]["euro_optimum_claim"] = True
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    report["report_sha256"] = _canonical_hash(unsigned)
    with pytest.raises(ValueError, match="truth boundary"):
        validate_report(report, expected_source_sha="a" * 40)
