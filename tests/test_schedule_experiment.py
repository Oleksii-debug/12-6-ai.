from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from twelve_six.scaling_500k_evidence import TARGET_PARAMETERS, _target_spec
from twelve_six.training import build_scheduler
from twelve_six.training.schedule_experiment import (
    SCHEDULES,
    load_plan,
    time_to_quality_for_seed,
    trainer_config_for,
)

PLAN_PATH = Path("configs/runs/train50_schedule_500k.experimental.json")


def test_plan_is_exact_full_horizon_467808_schedule_comparison() -> None:
    plan = load_plan(PLAN_PATH)
    assert _target_spec().parameter_count() == TARGET_PARAMETERS == 467_808
    assert tuple(plan["schedules"]) == SCHEDULES
    assert plan["planned_optimizer_steps"] == 1040
    assert plan["tokens_per_optimizer_step"] == 252
    assert plan["planned_optimized_tokens"] == 262_080
    assert plan["warmup_steps"] == 52
    assert plan["warmup_steps"] * 20 == plan["planned_optimizer_steps"]
    assert plan["validation_every_steps"] == 52


def test_schedule_arms_change_only_scheduler_family() -> None:
    plan = load_plan(PLAN_PATH)
    constant = trainer_config_for(
        plan,
        schedule="constant_with_warmup",
        seed=1337,
    )
    cosine = trainer_config_for(
        plan,
        schedule="cosine_with_warmup",
        seed=1337,
    )
    assert constant.learning_rate == cosine.learning_rate == 3e-4
    assert constant.weight_decay == cosine.weight_decay == 0.0
    assert constant.betas == cosine.betas == (0.9, 0.95)
    assert constant.eps == cosine.eps == 1e-8
    assert constant.gradient_clip_norm == cosine.gradient_clip_norm == 1.0
    assert constant.precision == cosine.precision == "fp32"
    assert constant.max_steps == cosine.max_steps == 1040
    assert constant.warmup_steps == cosine.warmup_steps == 52
    assert constant.scheduler == "constant"
    assert cosine.scheduler == "cosine"


def _lr_trace(schedule: str) -> list[float]:
    plan = load_plan(PLAN_PATH)
    config = trainer_config_for(plan, schedule=schedule, seed=1337)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = build_scheduler(optimizer, config)
    assert scheduler is not None
    trace: list[float] = []
    for _ in range(config.max_steps):
        trace.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return trace


def test_warmup_trace_matches_and_cosine_decays_over_full_planned_horizon() -> None:
    constant = _lr_trace("constant_with_warmup")
    cosine = _lr_trace("cosine_with_warmup")
    assert constant[:52] == cosine[:52]
    assert constant[0] == pytest.approx(3e-4 / 52)
    assert constant[51] == pytest.approx(3e-4)
    assert constant[-1] == pytest.approx(3e-4)
    assert cosine[52] == pytest.approx(3e-4)
    assert 0.0 <= cosine[-1] < 3e-8


def test_time_to_quality_uses_common_reachable_thresholds() -> None:
    def run(final: float, mid: float) -> dict[str, object]:
        return {
            "validation_curve": [
                {
                    "optimizer_step": 0,
                    "optimized_tokens": 0,
                    "validation_bpb": 5.0,
                    "training_wall_seconds": 0.0,
                },
                {
                    "optimizer_step": 10,
                    "optimized_tokens": 2520,
                    "validation_bpb": mid,
                    "training_wall_seconds": 1.0,
                },
                {
                    "optimizer_step": 20,
                    "optimized_tokens": 5040,
                    "validation_bpb": final,
                    "training_wall_seconds": 2.0,
                },
            ]
        }

    result = time_to_quality_for_seed(
        {
            "constant_with_warmup": run(3.0, 4.0),
            "cosine_with_warmup": run(2.5, 3.5),
        },
        [0.5, 1.0],
    )
    assert result["common_final_target_bpb"] == 3.0
    assert result["levels"][0]["threshold_validation_bpb"] == 4.0
    assert (
        result["levels"][0]["arrivals"]["constant_with_warmup"][
            "optimizer_step"
        ]
        == 10
    )
    assert (
        result["levels"][0]["arrivals"]["cosine_with_warmup"][
            "optimizer_step"
        ]
        == 10
    )
    assert (
        result["levels"][1]["arrivals"]["constant_with_warmup"][
            "optimizer_step"
        ]
        == 20
    )
    assert (
        result["levels"][1]["arrivals"]["cosine_with_warmup"][
            "optimizer_step"
        ]
        == 20
    )


def test_committed_plan_has_no_third_schedule_family() -> None:
    raw = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert raw["schedules"] == [
        "constant_with_warmup",
        "cosine_with_warmup",
    ]
