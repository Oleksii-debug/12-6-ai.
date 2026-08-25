from __future__ import annotations

import copy
import math

import pytest

from twelve_six.fixed_token_research import (
    DEPTH_WIDTH_SCHEMA,
    EXACT_TOKEN_BUDGETS,
    FIXED_SCHEMA,
    TOKENS_PER_UPDATE,
    _canonical_hash,
    _rank_fixed_runs,
    depth_width_specs,
    validate_exact_token_budgets,
    validate_report,
)


def test_strict_fixed_token_budget_rejects_research41_round_labels() -> None:
    with pytest.raises(ValueError, match="not reachable"):
        validate_exact_token_budgets((4_096, 16_384, 65_536))
    assert validate_exact_token_budgets() == EXACT_TOKEN_BUDGETS
    assert [budget // TOKENS_PER_UPDATE for budget in EXACT_TOKEN_BUDGETS] == [17, 66, 261]


def test_depth_width_family_is_tied_and_nearly_iso_parameter() -> None:
    candidates = depth_width_specs()
    assert [label for label, _ in candidates] == [
        "shallow_wide",
        "mid_shallow",
        "balanced",
        "deep_narrow",
        "very_deep_narrow",
    ]
    counts = [spec.parameter_count() for _, spec in candidates]
    assert counts == [496_808, 502_544, 495_456, 497_680, 503_496]
    assert max(counts) - min(counts) < 0.017 * 500_000
    assert [spec.n_layers for _, spec in candidates] == [2, 3, 4, 6, 8]
    assert [spec.d_model for _, spec in candidates] == [136, 112, 96, 80, 72]
    for _, spec in candidates:
        breakdown = spec.parameter_breakdown()
        assert spec.tie_word_embeddings is True
        assert breakdown["token_embedding"] == 256 * spec.d_model
        assert breakdown["lm_head_extra"] == 0
        assert breakdown["total"] == spec.parameter_count()


def _fake_fixed_run(label: str, parameters: int, initial: float, final: float, wall: float) -> dict:
    return {
        "label": label,
        "parameters": parameters,
        "initial_validation_loss": initial,
        "timing": {"end_to_end_wall_seconds": wall},
        "checkpoints": [
            {
                "validation_loss": final,
                "compute_proxy": 6 * parameters * EXACT_TOKEN_BUDGETS[-1],
            }
        ],
    }


def test_research_vehicle_rule_prefers_smallest_model_within_five_percent() -> None:
    runs = [
        _fake_fixed_run("100k", 100_000, 5.5, 2.50, 1.0),
        _fake_fixed_run("250k", 250_000, 5.5, 2.20, 2.0),
        _fake_fixed_run("500k", 500_000, 5.5, 2.08, 3.0),
        _fake_fixed_run("1m", 1_000_000, 5.5, 2.00, 5.0),
    ]
    ranking = _rank_fixed_runs(runs)
    assert ranking["best_validation"][0]["label"] == "1m"
    assert ranking["recommended_primary_small_model"]["label"] == "500k"


def _minimal_report(schema: str) -> dict:
    if schema == FIXED_SCHEMA:
        counts = [95_568, 267_912, 467_808, 1_037_696]
        layers = [None] * 4
    else:
        counts = [496_808, 502_544, 495_456, 497_680, 503_496]
        layers = [2, 3, 4, 6, 8]
    runs = []
    for index, parameters in enumerate(counts):
        checkpoints = []
        for budget in EXACT_TOKEN_BUDGETS:
            loss = 2.0 + index * 0.01
            checkpoints.append(
                {
                    "requested_token_budget": budget,
                    "optimized_tokens": budget,
                    "compute_proxy": 6 * parameters * budget,
                    "validation_loss": loss,
                    "bits_per_byte": loss / math.log(2.0),
                    "evaluation_guard": {
                        "optimized_validation_tokens": 0,
                        "trainer_tokens_unchanged": True,
                        "optimizer_step_unchanged": True,
                        "model_state_unchanged": True,
                        "trainer_state_unchanged": True,
                    },
                }
            )
        n_layers = layers[index]
        runs.append(
            {
                "parameters": parameters,
                "resume_proof": {"passed": True, "optimized_tokens": EXACT_TOKEN_BUDGETS[-2]},
                "checkpoints": checkpoints,
                "model_spec": {"n_layers": n_layers} if n_layers is not None else {},
                "parameter_breakdown": {"lm_head_extra": 0},
                "layer_summary": (
                    [{"layer": layer} for layer in range(n_layers)] if n_layers is not None else None
                ),
            }
        )
    report = {
        "schema": schema,
        "source": {"git_sha": "a" * 40},
        "runtime": {"paid_compute": False},
        "controls": {
            "exact_token_budgets": list(EXACT_TOKEN_BUDGETS),
            "valid_causal_loss_tokens_per_update": TOKENS_PER_UPDATE,
        },
        "runs": runs,
        "truth_boundary": {
            "held_out_generalization_measured": True,
            "train_loss_used_as_generalization": False,
            "evaluation_tokens_optimized": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


@pytest.mark.parametrize("schema", [FIXED_SCHEMA, DEPTH_WIDTH_SCHEMA])
def test_validator_rejects_rehashed_token_accounting_drift(schema: str) -> None:
    report = _minimal_report(schema)
    validate_report(report, expected_source_sha="a" * 40)
    tampered = copy.deepcopy(report)
    tampered["runs"][0]["checkpoints"][0]["optimized_tokens"] += 1
    unsigned = dict(tampered)
    unsigned.pop("report_sha256")
    tampered["report_sha256"] = _canonical_hash(unsigned)
    with pytest.raises(ValueError, match="optimized tokens"):
        validate_report(tampered, expected_source_sha="a" * 40)
