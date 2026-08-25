from __future__ import annotations

import copy

import torch

from twelve_six.microbatch_experiment import (
    BETAS,
    EFFECTIVE_BATCH_SIZE,
    EXECUTION_UPDATES,
    LEARNING_RATE,
    MICROBATCH_SIZES,
    SCHEDULER_HORIZON_STEPS,
    WEIGHT_DECAY,
    _canonical_hash,
    _config,
    _summarize,
    validate_report,
)
from twelve_six.scaling_experiment import _make_batch


def test_effective_batch_is_exactly_reconstructed_for_every_microbatch() -> None:
    stream = bytes(range(256)) * 8
    effective = _make_batch(
        stream,
        step=5,
        batch_size=EFFECTIVE_BATCH_SIZE,
        sequence_length=64,
    )
    for microbatch_size in MICROBATCH_SIZES:
        pieces = [
            effective[offset : offset + microbatch_size]
            for offset in range(0, EFFECTIVE_BATCH_SIZE, microbatch_size)
        ]
        assert torch.equal(torch.cat(pieces, dim=0), effective)


def test_optimizer_and_horizon_controls_are_fixed() -> None:
    config = _config(accumulation_steps=4)
    assert config.learning_rate == LEARNING_RATE == 3e-4
    assert config.betas == BETAS == (0.9, 0.95)
    assert config.weight_decay == WEIGHT_DECAY == 0.0
    assert config.max_steps == SCHEDULER_HORIZON_STEPS == 512
    assert config.max_steps > EXECUTION_UPDATES
    assert config.gradient_accumulation_steps == 4


def _fake_run(size: int, speed: float, delta: float) -> dict[str, object]:
    return {
        "parameters": 1_037_696,
        "model_identity_sha256": "b" * 64,
        "microbatch_size": size,
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // size,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "effective_predicted_tokens_per_update": 504,
        "optimizer_steps": 2,
        "optimized_tokens": 1008,
        "initial_validation_loss": 5.0,
        "final_validation_loss": 4.0 + delta,
        "mean_gradient_norm": 1.0,
        "max_gradient_norm": 2.0,
        "mean_update_to_weight_ratio": 0.01,
        "mean_step_wall_seconds": 1.0,
        "tokens_per_second": speed,
        "rss_max_sampled_bytes": 1,
        "rss_end_bytes": 1,
        "rss_scope": "same_process_current_RSS_samples_not_fresh_process_peak",
        "trace": [],
        "_final_model_vector": torch.tensor([1.0 + delta, 2.0]),
        "_optimizer_tensors": [torch.tensor([3.0 + delta])],
    }


def test_summary_attributes_numeric_drift_not_data_order() -> None:
    runs = [
        _fake_run(8, 100.0, 0.0),
        _fake_run(4, 125.0, 1e-6),
        _fake_run(2, 115.0, 2e-6),
        _fake_run(1, 90.0, 3e-6),
    ]
    public, decision = _summarize(runs)
    assert all(
        item["equivalence_vs_microbatch_8"]["same_effective_batch_order"]
        for item in public
    )
    assert decision["local_free_microbatch_size"] == 4
    assert decision["local_free_gradient_accumulation_steps"] == 2
    assert decision["target_gpu_status"] == "NOT_TESTED"


def _minimal_report() -> dict[str, object]:
    runs = []
    for size in MICROBATCH_SIZES:
        runs.append(
            {
                "microbatch_size": size,
                "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // size,
                "equivalence_vs_microbatch_8": {"same_effective_batch_order": True},
            }
        )
    report: dict[str, object] = {
        "schema": "12-6.train47-microbatch-experiment.v1",
        "authority": "LOCAL_FREE_CPU_MICROBATCH_EVIDENCE_PROVISIONAL",
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": "a" * 40},
        "controls": {
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "same_effective_batch_order": True,
            "scheduler_horizon_steps": SCHEDULER_HORIZON_STEPS,
            "execution_updates": EXECUTION_UPDATES,
        },
        "runs": runs,
        "truth_boundary": {"cpu_only": True, "gpu_behavior_claimed": False},
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def test_validator_rejects_effective_batch_drift() -> None:
    report = _minimal_report()
    validate_report(report, expected_source_sha="a" * 40)
    broken = copy.deepcopy(report)
    broken["runs"][1]["gradient_accumulation_steps"] = 3
    unsigned = {key: value for key, value in broken.items() if key != "report_sha256"}
    broken["report_sha256"] = _canonical_hash(unsigned)
    try:
        validate_report(broken, expected_source_sha="a" * 40)
    except ValueError as exc:
        assert "effective batch drift" in str(exc)
    else:
        raise AssertionError("effective-batch drift must fail closed")
