from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from twelve_six.training.lr_range_experiment import (
    _aggregate_summary,
    _classify_model_results,
    controlled_family,
    load_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/runs/train42_lr_range.experimental.json"


def _result(lr: float, initial: float, final: float, update_max: float = 0.01) -> dict:
    return {
        "learning_rate": lr,
        "status": "PASS",
        "initial_validation_loss": initial,
        "final_validation_loss": final,
        "relative_update_l2_max": update_max,
    }


def test_plan_is_small_log_spaced_and_keeps_cosine_horizon_long() -> None:
    plan = load_plan(PLAN)
    assert plan["learning_rates"] == [0.0001, 0.0003, 0.001, 0.003, 0.01]
    assert plan["current_default_learning_rate"] == 0.0003
    execution = plan["execution"]
    assert execution["schedule_horizon_steps"] >= 3 * execution["execution_steps"]
    assert plan["model_family"]["canonical_stage_configs_modified"] is False


def test_controlled_family_is_exact_100k_to_500k_research41_subset() -> None:
    specs = controlled_family()
    assert tuple(spec.parameter_count() for spec in specs) == (95_568, 267_912, 467_808)
    assert {spec.vocab_size for spec in specs} == {256}
    assert {spec.max_seq_len for spec in specs} == {256}


def test_classification_distinguishes_slow_healthy_and_unstable() -> None:
    plan = load_plan(PLAN)
    results = [
        _result(0.0001, 6.0, 5.8),
        _result(0.0003, 6.0, 5.0),
        _result(0.001, 6.0, 4.9),
        _result(0.003, 6.0, 4.8),
        _result(0.01, 6.0, 4.7, update_max=0.051),
    ]
    _classify_model_results(results, default_lr=0.0003, plan=plan)
    assert [result["classification"] for result in results] == [
        "too_slow",
        "healthy",
        "healthy",
        "healthy",
        "unstable",
    ]


def test_aggregate_transfer_never_promotes_above_healthy_default() -> None:
    base = [
        {"learning_rate": 0.0001, "classification": "too_slow"},
        {"learning_rate": 0.0003, "classification": "healthy"},
        {"learning_rate": 0.001, "classification": "healthy"},
        {"learning_rate": 0.003, "classification": "healthy"},
        {"learning_rate": 0.01, "classification": "unstable"},
    ]
    model_runs = [{"results": deepcopy(base)} for _ in range(3)]
    summary = _aggregate_summary(model_runs, 0.0003)
    assert summary["provisional_1m_learning_rate"] == 0.0003
    assert summary["aggressive_lr_promotion_authorized"] is False
