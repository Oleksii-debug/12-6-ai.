from __future__ import annotations

from pathlib import Path

import pytest
import torch

from twelve_six.training.stability_schedule_experiments import (
    _augment_result,
    _load_plan,
    _recipe,
    _select_clip,
)


def _fake_result(*, clip: float | None, final_validation_loss: float, grad_norm: float):
    recipe = {
        "gradient_clip_norm": clip,
        "warmup_steps": 4,
        "warmup_fraction": 0.05,
    }
    return {
        "recipe": recipe,
        "summary": {
            "status": "PASS",
            "initial_validation_loss": 5.0,
            "final_validation_loss": final_validation_loss,
        },
        "progression": [
            {
                "step": 1,
                "loss": 5.5,
                "grad_norm": grad_norm,
                "relative_update_l2": 0.01,
            },
            {
                "step": 2,
                "loss": 4.9,
                "grad_norm": grad_norm / 4,
                "relative_update_l2": 0.005,
            },
        ],
    }


def test_committed_plan_locks_lr_optimizer_and_decoupled_horizons() -> None:
    plan = _load_plan(Path("configs/runs/train43_45_stability.experimental.json"))["raw"]
    assert plan["incumbent"]["learning_rate"] == 3e-4
    assert plan["incumbent"]["betas"] == [0.9, 0.95]
    assert plan["incumbent"]["weight_decay"] == 0.1
    assert plan["warmup"]["fractions"] == [0.0, 0.01, 0.05, 0.10]
    assert plan["clipping"]["thresholds"] == [None, 1.0, 3.0, 10.0]
    for stage in plan["stages"]:
        assert stage["schedule_horizon_steps"] > stage["execution_steps"]


def test_recipe_changes_only_warmup_or_clip_on_locked_incumbent() -> None:
    incumbent = _load_plan(Path("configs/runs/train43_45_stability.experimental.json"))["raw"]["incumbent"]
    recipe = _recipe(
        "probe",
        incumbent,
        warmup_fraction=0.05,
        gradient_clip_norm=3.0,
    )
    config, materialized = recipe.materialize(schedule_horizon_steps=200, seed=1337)
    assert config.learning_rate == 3e-4
    assert config.betas == (0.9, 0.95)
    assert config.weight_decay == 0.1
    assert config.max_steps == 200
    assert config.warmup_steps == 10
    assert config.gradient_clip_norm == 3.0
    assert materialized["schedule_horizon_steps"] == 200


def test_augmentation_records_recovery_tokens_and_clip_factor() -> None:
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    batch = {"input_ids": labels.clone(), "labels": labels}
    result = _fake_result(clip=1.0, final_validation_loss=4.8, grad_norm=2.0)
    augmented = _augment_result(
        result,
        train_batches=[batch],
        execution_steps=2,
        early_window_steps=2,
        gradient_clip_norm=1.0,
    )
    summary = augmented["summary"]
    assert summary["initialization_recovery_needed"] is True
    assert summary["initialization_recovery_tokens"] == 6
    assert summary["early_loss_spike_absolute"] == pytest.approx(0.5)
    assert summary["clip_frequency"] == 0.5
    assert augmented["progression"][0]["post_clip_factor"] == pytest.approx(0.5, rel=1e-5)
    assert augmented["progression"][1]["post_clip_factor"] == pytest.approx(1.0, rel=1e-5)


def test_clip_selection_rejects_every_step_clip_when_no_clip_is_quality_equivalent() -> None:
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    batch = {"input_ids": labels.clone(), "labels": labels}
    results = {}
    for name, clip, val, grad in (
        ("clip_none", None, 4.80, 8.0),
        ("clip_1", 1.0, 4.79, 8.0),
        ("clip_3", 3.0, 4.80, 8.0),
        ("clip_10", 10.0, 4.81, 8.0),
    ):
        results[name] = _augment_result(
            _fake_result(clip=clip, final_validation_loss=val, grad_norm=grad),
            train_batches=[batch],
            execution_steps=2,
            early_window_steps=2,
            gradient_clip_norm=clip,
        )
    assert _select_clip(results) == "clip_10"
