from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.optimization_experiments import (
    OptimizationExperimentError,
    _run_recipe,
    _tensor_batches,
    load_experiment_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/runs/optimizer_experiments.experimental.json"


def test_plan_is_small_explicit_and_transfer_is_unexecuted() -> None:
    plan = load_experiment_plan(PLAN)
    assert set(plan["recipes"]) == {
        "adamw_baseline_constant",
        "adamw_wd01_constant",
        "adamw_wd01_warmup_constant",
        "adamw_wd01_warmup_cosine",
        "adamw_wd01_warmup_cosine_beta999",
        "adamw_wd01_warmup_cosine_no_clip",
        "adamw_wd01_warmup_cosine_lr1e3",
    }
    assert len(plan["raw"]["stages"][0]["recipes"]) == 7
    assert all(
        target["executed"] is False
        for target in plan["raw"]["unexecuted_transfer_targets"]
    )


def test_warmup_fraction_materializes_against_schedule_horizon() -> None:
    plan = load_experiment_plan(PLAN)
    recipe = plan["recipes"]["adamw_wd01_warmup_cosine"]
    config, materialized = recipe.materialize(schedule_horizon_steps=80, seed=1337)
    assert config.max_steps == 80
    assert config.warmup_steps == 4
    assert materialized["warmup_steps"] == 4
    assert materialized["trainer_config_sha256"]


def test_plan_rejects_unapproved_muon_without_composite_contract(tmp_path: Path) -> None:
    raw = json.loads(PLAN.read_text(encoding="utf-8"))
    raw["recipes"]["adamw_baseline_constant"]["optimizer"] = "muon"
    changed = tmp_path / "plan.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(OptimizationExperimentError, match="only AdamW"):
        load_experiment_plan(changed)


def test_real_s1_probe_measures_updates_gradients_time_and_optimizer_memory() -> None:
    plan = load_experiment_plan(PLAN)
    recipe = plan["recipes"]["adamw_wd01_warmup_cosine"]
    tokenizer = ByteTokenizer()
    train_batches, _, _, _ = _tensor_batches(
        ROOT,
        split="train",
        tokenizer=tokenizer,
        batch_size=3,
        sequence_length=128,
    )
    validation_batches, _, _, _ = _tensor_batches(
        ROOT,
        split="validation",
        tokenizer=tokenizer,
        batch_size=3,
        sequence_length=128,
    )
    result = _run_recipe(
        ROOT / "configs/stages/s1_100k.json",
        recipe=recipe,
        execution_steps=2,
        schedule_horizon_steps=8,
        train_batches=train_batches,
        validation_batches=validation_batches,
        seed=1337,
    )
    summary = result["summary"]
    assert summary["status"] == "PASS"
    assert summary["steps_completed"] == 2
    assert summary["optimizer_state_tensor_bytes_final"] > 0
    assert summary["relative_update_l2_median"] > 0.0
    assert summary["step_wall_seconds_median"] > 0.0
    assert math.isfinite(summary["gradient_norm_max"])
    assert len(result["progression"]) == 2
    assert all(item["model_parameters_finite"] for item in result["progression"])
