from __future__ import annotations

import math
from pathlib import Path

from twelve_six.model import load_stage_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.adam_epsilon_experiment import (
    _compare_to_baseline,
    _run_candidate,
    fixed_500k_model_spec,
    load_plan,
)
from twelve_six.training.optimization_experiments import _tensor_batches

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/runs/adam_epsilon_500k.experimental.json"


def test_plan_is_epsilon_only_and_scheduler_horizon_is_independent() -> None:
    plan = load_plan(PLAN)["raw"]
    assert plan["epsilon_values"] == [1e-8, 1e-6, 1e-4]
    assert plan["precisions"] == ["fp32", "bf16"]
    controls = plan["controls"]
    assert controls["learning_rate"] == 3e-4
    assert controls["betas"] == [0.9, 0.95]
    assert controls["weight_decay"] == 0.0
    assert controls["scheduler"] == "constant"
    assert controls["warmup_steps"] == 0
    assert controls["gradient_clip_norm"] == 1.0
    assert controls["execution_steps"] < controls["schedule_horizon_steps"]


def test_fixed_model_is_exact_research41_467808_member() -> None:
    spec = fixed_500k_model_spec()
    assert spec.parameter_count() == 467_808
    assert spec.vocab_size == 256
    assert spec.max_seq_len == 256
    assert spec.d_model == 96
    assert spec.n_layers == 4
    assert spec.n_heads == 6
    assert spec.n_kv_heads == 6
    assert spec.head_dim == 16
    assert spec.d_ff == 256


def test_two_step_fp32_probe_measures_loss_gradients_updates_and_finite_state() -> None:
    raw = load_plan(PLAN)["raw"]
    controls = dict(raw["controls"])
    controls["execution_steps"] = 2
    controls["schedule_horizon_steps"] = 8
    tokenizer = ByteTokenizer()
    train_batches, _, _, _ = _tensor_batches(
        ROOT,
        split="train",
        tokenizer=tokenizer,
        batch_size=controls["batch_size"],
        sequence_length=controls["sequence_length"],
    )
    validation_batches, _, _, _ = _tensor_batches(
        ROOT,
        split="validation",
        tokenizer=tokenizer,
        batch_size=controls["batch_size"],
        sequence_length=controls["sequence_length"],
    )
    init = load_stage_config(ROOT / "configs/stages/s1_100k.json").init
    result = _run_candidate(
        spec=fixed_500k_model_spec(),
        init=init,
        train_batches=train_batches,
        validation_batches=validation_batches,
        controls=controls,
        epsilon=1e-8,
        precision="fp32",
    )
    summary = result["summary"]
    assert summary["status"] == "PASS"
    assert summary["steps_completed"] == 2
    assert math.isfinite(summary["final_validation_loss"])
    assert math.isfinite(summary["gradient_norm_median"])
    assert summary["relative_update_l2_median"] > 0.0
    assert summary["all_model_states_finite"] is True
    assert summary["all_optimizer_states_finite"] is True
    assert summary["second_moment"]["state_elements"] > 0
    assert len(result["progression"]) == 2


def _synthetic_result(loss: float, update: float, grad: float) -> dict:
    return {
        "summary": {
            "status": "PASS",
            "final_validation_loss": loss,
            "training_loss_last": loss,
            "relative_update_l2_median": update,
            "gradient_norm_median": grad,
            "all_model_states_finite": True,
            "all_optimizer_states_finite": True,
        },
        "progression": [{"loss": loss}, {"loss": loss}],
    }


def test_materiality_rule_is_precommitted_and_not_exact_equality() -> None:
    thresholds = load_plan(PLAN)["raw"]["materiality_thresholds"]
    baseline = _synthetic_result(2.0, 0.01, 1.0)
    near = _synthetic_result(2.001, 0.0101, 1.01)
    far = _synthetic_result(2.1, 0.012, 1.2)
    assert _compare_to_baseline(baseline, near, thresholds)["material"] is False
    assert _compare_to_baseline(baseline, far, thresholds)["material"] is True
