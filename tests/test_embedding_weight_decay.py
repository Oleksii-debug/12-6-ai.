from __future__ import annotations

from pathlib import Path

import torch

from twelve_six.model import InitSpec
from twelve_six.training.embedding_weight_decay import (
    EXPECTED_PARAMETERS,
    fixed_model_spec,
    load_plan,
    recommendation,
    resume_regression,
)
from twelve_six.training.lr_range_experiment import controlled_family


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/runs/train44_embedding_weight_decay.experimental.json"


def _synthetic_batch() -> dict[str, torch.Tensor]:
    values = torch.arange(64, dtype=torch.long).reshape(2, 32) % 256
    return {"input_ids": values, "labels": values.clone()}


def test_plan_changes_only_embedding_decay_membership() -> None:
    plan = load_plan(PLAN)
    assert plan["learning_rate"] == 3e-4
    assert plan["optimizer"]["weight_decay"] == 0.1
    assert plan["optimizer"]["betas"] == [0.9, 0.95]
    assert plan["optimizer"]["scheduler"] == "constant"
    assert plan["conditions"] == ["all_parameters_decayed", "embedding_excluded"]
    assert plan["execution_steps"] >= 256


def test_fixed_model_is_tied_468k_control() -> None:
    spec = fixed_model_spec()
    assert spec.parameter_count() == EXPECTED_PARAMETERS
    assert spec.tie_word_embeddings is True
    assert spec.vocab_size == 256


def test_parameter_group_checkpoint_resume_is_exact_for_both_conditions() -> None:
    plan = load_plan(PLAN)
    spec = controlled_family()[0]
    batch = _synthetic_batch()
    all_decay = resume_regression(
        spec=spec,
        init_spec=InitSpec(),
        plan=plan,
        train_batches=[batch],
        exclude_embedding=False,
    )
    excluded = resume_regression(
        spec=spec,
        init_spec=InitSpec(),
        plan=plan,
        train_batches=[batch],
        exclude_embedding=True,
    )
    assert all_decay["passed"] is True
    assert all_decay["group_weight_decays_after_load"] == [0.1]
    assert excluded["passed"] is True
    assert excluded["group_weight_decays_after_load"] == [0.1, 0.0]


def test_recommendation_prefers_exclusion_within_predeclared_bpb_tolerance() -> None:
    results = [
        {"condition": "all_parameters_decayed", "final_validation_bpb": 3.0},
        {"condition": "embedding_excluded", "final_validation_bpb": 3.01},
    ]
    resume = {
        "all_parameters_decayed": {"passed": True},
        "embedding_excluded": {"passed": True},
    }
    result = recommendation(results, resume)
    assert result["selected_default_grouping"] == "embedding_excluded"


def test_recommendation_can_select_all_decay_for_material_bpb_gain() -> None:
    results = [
        {"condition": "all_parameters_decayed", "final_validation_bpb": 3.0},
        {"condition": "embedding_excluded", "final_validation_bpb": 3.03},
    ]
    resume = {
        "all_parameters_decayed": {"passed": True},
        "embedding_excluded": {"passed": True},
    }
    result = recommendation(results, resume)
    assert result["selected_default_grouping"] == "all_parameters_decayed"
