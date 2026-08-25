from __future__ import annotations

import copy

import torch

from twelve_six.scaling_experiment import _make_batch
from twelve_six.schedule_batch_experiment import (
    BASE_LR,
    BETAS,
    EFFECTIVE_BATCH_SIZE,
    SCHEDULER_HORIZON_STEPS,
    WARMUP_UPDATES,
    WEIGHT_DECAY,
    _hash_payload,
    _microbatch_summary,
    _trainer_config,
    _update_ratio,
    validate_report,
)


def test_warmup_scheduler_horizon_is_not_short_experiment_length() -> None:
    config = _trainer_config(warmup_steps=32, accumulation=1, scheduler="cosine")
    assert config.max_steps == SCHEDULER_HORIZON_STEPS == 512
    assert WARMUP_UPDATES == 128
    assert config.max_steps > WARMUP_UPDATES
    assert config.learning_rate == BASE_LR == 3e-4
    assert config.betas == BETAS == (0.9, 0.95)
    assert config.weight_decay == WEIGHT_DECAY == 0.0


def test_microbatch_slicing_preserves_exact_effective_batch_order() -> None:
    stream = bytes(range(256)) * 8
    effective = _make_batch(
        stream,
        step=7,
        batch_size=EFFECTIVE_BATCH_SIZE,
        sequence_length=64,
    )
    for microbatch_size in (8, 4, 2, 1):
        pieces = [
            effective[offset : offset + microbatch_size]
            for offset in range(0, EFFECTIVE_BATCH_SIZE, microbatch_size)
        ]
        reconstructed = torch.cat(pieces, dim=0)
        assert torch.equal(reconstructed, effective)


def test_update_ratio_is_zero_then_positive_after_weight_change() -> None:
    layer = torch.nn.Linear(2, 2, bias=False)
    before = [layer.weight.detach().clone()]
    assert _update_ratio(before, layer) == 0.0
    with torch.no_grad():
        layer.weight.add_(0.01)
    assert _update_ratio(before, layer) > 0.0


def _fake_micro_run(size: int, speed: float, delta: float) -> dict[str, object]:
    accumulation = EFFECTIVE_BATCH_SIZE // size
    return {
        "parameters": 1_037_696,
        "microbatch_size": size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "effective_tokens_per_update": 504,
        "optimizer_steps": 2,
        "optimized_tokens": 1008,
        "initial_validation_loss": 5.0,
        "final_validation_loss": 4.0 + delta,
        "mean_grad_norm": 1.0,
        "mean_update_ratio": 0.01,
        "mean_step_wall_seconds": 1.0,
        "tokens_per_second": speed,
        "rss_before_run_bytes": 1,
        "rss_max_sampled_bytes": 2,
        "rss_end_bytes": 2,
        "rss_measurement_scope": "same_process_current_RSS_samples_not_fresh_process_peak",
        "final_model_vector": torch.tensor([1.0 + delta, 2.0]),
        "optimizer_tensors": [torch.tensor([3.0 + delta])],
        "trace": [],
    }


def test_microbatch_summary_separates_numeric_drift_from_data_order() -> None:
    runs = [
        _fake_micro_run(8, 100.0, 0.0),
        _fake_micro_run(4, 120.0, 1e-6),
        _fake_micro_run(2, 110.0, 2e-6),
        _fake_micro_run(1, 90.0, 3e-6),
    ]
    summarized, recommendation = _microbatch_summary(runs)
    assert all(r["equivalence_vs_fullbatch"]["same_effective_batch_order"] for r in summarized)
    assert recommendation["local_free_microbatch_size"] == 4
    assert recommendation["local_free_accumulation_steps"] == 2
    assert recommendation["gpu_evidence_status"] == "NOT_TESTED"


def test_report_validator_rejects_collapsed_horizon() -> None:
    micro_runs = []
    for size in (8, 4, 2, 1):
        micro_runs.append(
            {
                "microbatch_size": size,
                "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // size,
                "equivalence_vs_fullbatch": {"same_effective_batch_order": True},
            }
        )
    report = {
        "schema": "12-6.train43-train47-experiment.v1",
        "authority": "LOCAL_FREE_CPU_EXPERIMENTAL_EVIDENCE",
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": "a" * 40,
        },
        "controls": {
            "learning_rate": BASE_LR,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "scheduler_horizon_steps": SCHEDULER_HORIZON_STEPS,
            "warmup_experiment_updates": WARMUP_UPDATES,
        },
        "warmup": {"scales": [95_568, 1_037_696]},
        "microbatch": {"runs": micro_runs},
        "truth_boundary": {"cpu_only": True, "gpu_behavior_claimed": False},
    }
    report["report_sha256"] = _hash_payload(report)
    validate_report(report, expected_source_sha="a" * 40)

    broken = copy.deepcopy(report)
    broken["controls"]["scheduler_horizon_steps"] = WARMUP_UPDATES
    broken["report_sha256"] = _hash_payload({k: v for k, v in broken.items() if k != "report_sha256"})
    try:
        validate_report(broken, expected_source_sha="a" * 40)
    except ValueError as exc:
        assert "scheduler horizon collapsed" in str(exc)
    else:
        raise AssertionError("collapsed scheduler horizon should fail closed")
