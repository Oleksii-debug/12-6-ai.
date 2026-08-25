from __future__ import annotations

from collections import Counter
from pathlib import Path

from twelve_six.curriculum_experiment import (
    MODALITIES,
    _load_config,
    _trace_multiset_identity,
    build_incumbent_plan,
    materialize_incumbent_trace,
    order_trace,
    run_candidate,
)


CONFIG = Path("configs/experiments/data35_curriculum_v1.json")


def _fixture():
    config = _load_config(CONFIG)
    batch_size = int(config["base_control"]["batch_size"])
    sequence_length = int(config["base_control"]["sequence_length"])
    capacity = batch_size * sequence_length
    final_tokens = int(config["budget"]["optimized_loss_tokens"])
    prefix_tokens = int(config["budget"]["curriculum_prefix_tokens"])
    plan = build_incumbent_plan(config)
    trace = materialize_incumbent_trace(
        plan,
        steps=final_tokens // capacity,
        tokens_per_step=capacity,
    )
    return config, capacity, trace, prefix_tokens // capacity


def test_curricula_are_permutations_of_one_incumbent_trace() -> None:
    _config, _capacity, trace, prefix_steps = _fixture()
    baseline_identity = _trace_multiset_identity(trace)
    baseline_counts = Counter(entry.source for entry in trace)

    for candidate in (
        "fully_mixed",
        "quality_first_then_mixed",
        "ukrainian_first_then_mixed",
    ):
        ordered = order_trace(trace, candidate=candidate, prefix_steps=prefix_steps)
        assert _trace_multiset_identity(ordered) == baseline_identity
        assert Counter(entry.source for entry in ordered) == baseline_counts


def test_quality_first_improves_training_only_prefix_proxy() -> None:
    _config, _capacity, trace, prefix_steps = _fixture()
    baseline = order_trace(trace, candidate="fully_mixed", prefix_steps=prefix_steps)
    quality = order_trace(
        trace,
        candidate="quality_first_then_mixed",
        prefix_steps=prefix_steps,
    )
    baseline_mean = sum(entry.quality_score for entry in baseline[:prefix_steps]) / prefix_steps
    quality_mean = sum(entry.quality_score for entry in quality[:prefix_steps]) / prefix_steps
    assert quality_mean >= baseline_mean


def test_ukrainian_first_prefix_is_bounded_and_compensated() -> None:
    _config, _capacity, trace, prefix_steps = _fixture()
    ordered = order_trace(
        trace,
        candidate="ukrainian_first_then_mixed",
        prefix_steps=prefix_steps,
    )
    assert {entry.source for entry in ordered[:prefix_steps]} == {"uk"}
    assert Counter(entry.identity() for entry in ordered) == Counter(
        entry.identity() for entry in trace
    )


def test_final_modality_token_counts_are_candidate_invariant() -> None:
    _config, capacity, trace, prefix_steps = _fixture()
    expected = {
        modality: sum(entry.source == modality for entry in trace) * capacity
        for modality in MODALITIES
    }
    assert sum(expected.values()) == len(trace) * capacity
    for candidate in (
        "fully_mixed",
        "quality_first_then_mixed",
        "ukrainian_first_then_mixed",
    ):
        ordered = order_trace(trace, candidate=candidate, prefix_steps=prefix_steps)
        observed = {
            modality: sum(entry.source == modality for entry in ordered) * capacity
            for modality in MODALITIES
        }
        assert observed == expected


def test_near_250k_candidate_executes_exact_aligned_token_smoke() -> None:
    _config, capacity, trace, _prefix_steps = _fixture()
    smoke_order = trace[:2]
    result = run_candidate(
        order=smoke_order,
        seed=1337,
        final_tokens=2 * capacity,
        checkpoint_tokens=(capacity, 2 * capacity),
        batch_size=4,
        sequence_length=64,
    )
    assert result["parameters"] == 267_912
    assert result["optimized_tokens"] == 2 * capacity
    assert sum(result["modality_optimized_tokens"].values()) == 2 * capacity
    assert result["checkpoints"][-1]["optimized_tokens"] == 2 * capacity
