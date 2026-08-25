from __future__ import annotations

import copy

import pytest
import torch

from twelve_six.fixed_token_experiment import (
    AUTHORITY,
    CANDIDATE_SCHEMA,
    _canonical_hash,
    _model_from_geometry,
    make_exact_causal_batch,
    strict_step_plan,
    validate_candidate,
)
from twelve_six.scaling_experiment import controlled_specs


def test_strict_step_plan_lands_exactly_on_every_budget() -> None:
    budgets = (4096, 16_384, 65_536)
    plan = strict_step_plan(budgets, capacity=4 * 63)
    boundaries = [
        int(item["cumulative_optimized_tokens"])
        for item in plan
        if bool(item["budget_boundary"])
    ]
    assert boundaries == list(budgets)
    assert sum(int(item["valid_loss_tokens"]) for item in plan) == 65_536
    assert int(plan[-1]["cumulative_optimized_tokens"]) == 65_536
    assert all(1 <= int(item["valid_loss_tokens"]) <= 252 for item in plan)
    assert len(plan) == 262


def test_exact_causal_batch_counts_only_real_next_token_pairs() -> None:
    stream = bytes(range(1, 120))
    batch = make_exact_causal_batch(
        stream,
        step=3,
        batch_size=4,
        sequence_length=16,
        valid_loss_tokens=37,
    )
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    loss_mask = batch["loss_mask"]
    assert int(loss_mask.sum().item()) == 37
    assert not bool(loss_mask[:, -1].any().item())
    assert torch.equal(target_ids[:, :-1], input_ids[:, 1:])
    valid = loss_mask.bool()
    assert int(valid.sum().item()) == 37


def test_model10_gqa_reallocates_kv_savings_with_tight_parameter_match() -> None:
    mha = controlled_specs()[-1]
    gqa = _model_from_geometry(
        {
            "d_model": 128,
            "n_layers": 5,
            "n_heads": 8,
            "n_kv_heads": 4,
            "head_dim": 16,
            "d_ff": 395,
        }
    )
    assert mha.parameter_count() == 1_037_696
    assert gqa.parameter_count() == 1_038_336
    assert abs(gqa.parameter_count() - mha.parameter_count()) / mha.parameter_count() < 0.001
    assert gqa.n_heads == mha.n_heads == 8
    assert gqa.n_kv_heads == 4
    assert gqa.d_ff > mha.d_ff
    assert (
        gqa.parameter_breakdown()["attention_per_layer"]
        < mha.parameter_breakdown()["attention_per_layer"]
    )
    assert gqa.parameter_breakdown()["mlp_per_layer"] > mha.parameter_breakdown()["mlp_per_layer"]


def _minimal_valid_candidate() -> dict:
    parameters = 95_568
    checkpoint = {
        "requested_token_budget": 65_536,
        "optimized_tokens": 65_536,
        "optimizer_steps": 1,
        "compute_proxy": 6 * parameters * 65_536,
        "validation_loss": 2.0,
        "validation_bpb": 2.0,
        "validation_loss_tokens": 10,
        "evaluation_optimized_tokens": 0,
        "last_train_loss": 1.9,
        "last_grad_norm": 1.0,
    }
    report = {
        "schema": CANDIDATE_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": "a" * 40,
        "candidate_label": "P100K_MHA",
        "model": {"parameters": parameters},
        "controls": {},
        "data": {},
        "token_accounting": {
            "optimized_tokens_final": 65_536,
            "expected_optimized_tokens_final": 65_536,
            "evaluation_loss_tokens_per_pass": 10,
            "evaluation_optimized_tokens_total": 0,
            "all_budget_boundaries_exact": True,
            "batch_trace_sha256": "b" * 64,
            "resume_control_trace_exact": True,
        },
        "heldout": {"checkpoints": [checkpoint]},
        "optimization": {},
        "timing": {},
        "memory": {},
        "resume": {
            "model_state_exact_vs_uninterrupted": True,
            "trainer_optimizer_state_exact_vs_uninterrupted": True,
        },
        "attention_probe": None,
        "truth_boundary": {
            "broad_corpus_generalization_claim": False,
            "gpu_performance_claim": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def test_candidate_validator_fails_closed_on_token_overshoot() -> None:
    report = _minimal_valid_candidate()
    validate_candidate(report, expected_source_sha="a" * 40)
    bad = copy.deepcopy(report)
    bad["heldout"]["checkpoints"][0]["optimized_tokens"] = 65_537
    bad["heldout"]["checkpoints"][0]["compute_proxy"] = 6 * 95_568 * 65_537
    unsigned = dict(bad)
    unsigned.pop("report_sha256")
    bad["report_sha256"] = _canonical_hash(unsigned)
    with pytest.raises(ValueError, match="budget drift"):
        validate_candidate(bad, expected_source_sha="a" * 40)


def test_candidate_validator_fails_closed_if_eval_tokens_are_optimized() -> None:
    bad = _minimal_valid_candidate()
    bad["token_accounting"]["evaluation_optimized_tokens_total"] = 10
    unsigned = dict(bad)
    unsigned.pop("report_sha256")
    bad["report_sha256"] = _canonical_hash(unsigned)
    with pytest.raises(ValueError, match="evaluation tokens"):
        validate_candidate(bad, expected_source_sha="a" * 40)
