from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from twelve_six.training.observability import (
    PhaseTimings,
    RankMetadata,
    TrainingObserver,
    aggregate_rank_summaries,
    paid_compute_decision_support,
)
from twelve_six.training.trainer import StepMetrics


def _identity() -> dict[str, object]:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "source_sha": "a" * 40,
        "stage": "S1",
        "modelspec_sha256": "b" * 64,
        "training": {"seed": 1337, "max_steps": 8},
    }


def _metrics(step: int, *, tokens: int = 128, grad_norm: float = 0.5) -> StepMetrics:
    return StepMetrics(
        micro_step=step,
        optimizer_step=step,
        loss=5.0 - step * 0.1,
        update_loss=5.0 - step * 0.1,
        learning_rate=1e-3,
        grad_norm=grad_norm,
        tokens=tokens,
        optimizer_stepped=True,
    )


def test_run_identity_hash_is_stable_across_different_timing_observations() -> None:
    left = TrainingObserver(_identity(), max_step_samples=8)
    right = TrainingObserver(_identity(), max_step_samples=8)

    left.record_step(_metrics(1), data_wait_seconds=0.01, step_seconds=0.10)
    right.record_step(_metrics(1), data_wait_seconds=9.0, step_seconds=17.0)

    assert left.run_identity_sha256 == right.run_identity_sha256
    assert left.summary()["timing"] != right.summary()["timing"]
    assert left.summary()["determinism_boundary"] == {
        "run_identity_hash_contains_telemetry": False,
        "timing_or_resource_metrics_are_training_state": False,
        "telemetry_collection_initializes_distributed_process_group": False,
    }


@pytest.mark.parametrize(
    "key",
    ["wall_seconds", "step_duration", "tokens_per_second", "gpu_utilization_percent"],
)
def test_run_identity_rejects_nondeterministic_telemetry_fields(key: str) -> None:
    identity = _identity()
    identity[key] = 1.0
    with pytest.raises(ValueError, match="must not enter run identity"):
        TrainingObserver(identity)


def test_observer_captures_required_step_fields_without_trainer_semantic_copy() -> None:
    observer = TrainingObserver(_identity(), max_step_samples=8)
    observed = observer.record_step(
        _metrics(1, tokens=256, grad_norm=1.25),
        data_wait_seconds=0.02,
        step_seconds=0.08,
        phases=PhaseTimings(
            forward_seconds=0.02,
            backward_seconds=0.04,
            update_seconds=0.02,
        ),
    )
    summary = observer.summary()

    assert observed.train_tokens_per_second == pytest.approx(2560.0)
    assert observed.compute_tokens_per_second == pytest.approx(3200.0)
    assert summary["counters"]["optimized_tokens"] == 256
    assert summary["optimization"]["loss_final"] == pytest.approx(4.9)
    assert summary["optimization"]["learning_rate_final"] == pytest.approx(1e-3)
    assert summary["optimization"]["gradient_norm_max"] == pytest.approx(1.25)
    assert summary["phase_timing"]["forward"]["status"] == "MEASURED"
    assert summary["phase_timing"]["backward"]["seconds_total"] == pytest.approx(0.04)
    assert summary["phase_timing"]["update"]["seconds_total"] == pytest.approx(0.02)


def test_current_whole_step_path_marks_phase_timings_unavailable() -> None:
    observer = TrainingObserver(_identity())
    observer.record_step(_metrics(1), data_wait_seconds=0.01, step_seconds=0.20)
    phases = observer.summary()["phase_timing"]

    assert phases["forward"]["status"] == "UNAVAILABLE_NOT_RECORDED"
    assert phases["backward"]["status"] == "UNAVAILABLE_NOT_RECORDED"
    assert phases["update"]["status"] == "UNAVAILABLE_NOT_RECORDED"
    assert "WHOLE_MICROBATCH_ONLY" in phases["current_d02_trainer_contract"]


def test_bottleneck_classifier_distinguishes_data_and_compute_dominance() -> None:
    data_bound = TrainingObserver(_identity())
    compute_bound = TrainingObserver(_identity())
    for step in range(1, 5):
        data_bound.record_step(
            _metrics(step),
            data_wait_seconds=0.30,
            step_seconds=0.70,
        )
        compute_bound.record_step(
            _metrics(step),
            data_wait_seconds=0.01,
            step_seconds=0.99,
        )

    assert data_bound.summary()["bottleneck"]["classification"] == "DATA_BOUND"
    assert (
        compute_bound.summary()["bottleneck"]["classification"]
        == "COMPUTE_BOUND_OR_RUNTIME_BOUND"
    )


def test_step_sample_retention_is_bounded_without_losing_aggregate_totals() -> None:
    observer = TrainingObserver(_identity(), max_step_samples=4)
    for step in range(1, 101):
        observer.record_step(
            _metrics(step, tokens=10),
            data_wait_seconds=0.001,
            step_seconds=0.009,
        )
    summary = observer.summary()

    assert len(observer.step_samples) <= 4
    assert summary["counters"]["observed_microbatches"] == 100
    assert summary["counters"]["optimized_tokens"] == 1000
    assert summary["throughput"]["train_tokens_per_second"] == pytest.approx(1000.0)
    assert summary["counters"]["retained_step_stride"] > 1


def test_jsonl_is_post_run_structured_and_identity_record_is_timing_free(tmp_path: Path) -> None:
    observer = TrainingObserver(_identity())
    observer.record_step(_metrics(1), data_wait_seconds=0.01, step_seconds=0.09)
    output = tmp_path / "telemetry.jsonl"
    observer.write_jsonl(output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert records[0]["record_type"] == "run_identity"
    assert "step_seconds" not in json.dumps(records[0]["run_identity"])
    assert any(record["record_type"] == "step" for record in records)
    assert records[-1]["record_type"] == "summary"


def test_rank_metadata_reads_launcher_environment_without_initializing_dist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    rank = RankMetadata.detect()

    assert rank == RankMetadata(
        rank=2,
        local_rank=0,
        world_size=4,
        distributed_initialized=False,
        backend=None,
    )


def test_rank_aggregation_uses_sum_tokens_over_slowest_rank_wall() -> None:
    first = TrainingObserver(
        _identity(),
        rank=RankMetadata(0, 0, 2, True, "gloo"),
    )
    second = TrainingObserver(
        _identity(),
        rank=RankMetadata(1, 1, 2, True, "gloo"),
    )
    first.record_step(_metrics(1, tokens=100), data_wait_seconds=0.0, step_seconds=1.0)
    second.record_step(_metrics(1, tokens=100), data_wait_seconds=0.0, step_seconds=2.0)

    aggregate = aggregate_rank_summaries([first.summary(), second.summary()])

    assert aggregate["world_size"] == 2
    assert aggregate["optimized_tokens_sum"] == 200
    assert aggregate["critical_path_training_seconds"] == pytest.approx(2.0)
    assert aggregate["global_train_tokens_per_second"] == pytest.approx(100.0)
    assert aggregate["rank_time_skew_ratio"] == pytest.approx(2.0)


def test_paid_compute_gate_refuses_cpu_only_capacity_claim() -> None:
    observer = TrainingObserver(_identity(), device="cpu")
    observer.record_step(_metrics(1), data_wait_seconds=0.01, step_seconds=0.09)

    decision = paid_compute_decision_support(observer.summary())

    assert decision["euro_2000_gate"] == "BLOCKED_PENDING_TARGET_GPU_CALIBRATION"
    assert (
        decision["euro_10000_gate"]
        == "BLOCKED_PENDING_TARGET_GPU_AND_DISTRIBUTED_CALIBRATION"
    )
    assert decision["telemetry_alone_authorizes_spend"] is False
