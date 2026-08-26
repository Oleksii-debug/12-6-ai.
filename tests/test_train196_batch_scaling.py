from __future__ import annotations

import torch

from twelve_six.train196_batch_scaling import (
    BASE_MICROBATCH_LOSS_TOKENS,
    CANDIDATES,
    SEQUENCE_LENGTH,
    TOTAL_OPTIMIZED_LOSS_TOKENS,
    _bootstrap_ci,
    _group_base_trace,
    _selection,
)


def _dummy_batch(start: int) -> dict[str, torch.Tensor]:
    values = torch.arange(start, start + 4 * SEQUENCE_LENGTH, dtype=torch.long).reshape(
        4, SEQUENCE_LENGTH
    )
    return {"input_ids": values, "labels": values.clone()}


def test_statistical_grid_uses_existing_accumulation_path_and_exact_budget() -> None:
    statistical = [item for item in CANDIDATES if item["role"] == "statistical"]
    assert [item["effective_loss_tokens"] for item in statistical] == [504, 1008, 2016]
    for item in statistical:
        assert item["microbatch_examples"] == 4
        expected = (
            item["microbatch_examples"]
            * (SEQUENCE_LENGTH - 1)
            * item["accumulation"]
        )
        assert expected == item["effective_loss_tokens"]
        assert TOTAL_OPTIMIZED_LOSS_TOKENS % item["effective_loss_tokens"] == 0
    assert BASE_MICROBATCH_LOSS_TOKENS == 252
    assert TOTAL_OPTIMIZED_LOSS_TOKENS == 64_512


def test_hardware_control_preserves_underlying_example_order() -> None:
    base = [_dummy_batch(0), _dummy_batch(1000), _dummy_batch(2000), _dummy_batch(3000)]
    grouped = _group_base_trace(base, 8)
    assert len(grouped) == 2
    assert torch.equal(grouped[0]["input_ids"][:4], base[0]["input_ids"])
    assert torch.equal(grouped[0]["input_ids"][4:], base[1]["input_ids"])
    assert torch.equal(grouped[1]["labels"][:4], base[2]["labels"])
    assert torch.equal(grouped[1]["labels"][4:], base[3]["labels"])


def test_bootstrap_is_deterministic() -> None:
    values = [0.0, 0.01, -0.02]
    first = _bootstrap_ci(values, seed=196, replicates=1000)
    second = _bootstrap_ci(values, seed=196, replicates=1000)
    assert first == second
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def _record(seed: int, bpb: float) -> dict[str, object]:
    return {"seed": seed, "final_heldout": {"bpb": bpb}}


def _aggregate(median: float) -> dict[str, object]:
    return {"final_bpb": {"median": median}}


def test_selection_chooses_smaller_batch_inside_practical_tie() -> None:
    statistical = {
        "batch-504": [_record(1, 3.000), _record(2, 3.010), _record(3, 2.990)],
        "batch-1008": [_record(1, 2.995), _record(2, 3.005), _record(3, 2.985)],
        "batch-2016": [_record(1, 3.050), _record(2, 3.060), _record(3, 3.040)],
    }
    aggregates = {
        "batch-504": _aggregate(3.000),
        "batch-1008": _aggregate(2.995),
        "batch-2016": _aggregate(3.050),
    }
    decision = _selection(statistical, aggregates)
    assert decision["best_raw_median_bpb"] == "batch-1008"
    assert decision["selected"] == "batch-504"
    assert decision["selected_effective_loss_tokens"] == 504
    assert decision["grid_edge"] is False


def test_selection_marks_largest_grid_winner_provisional() -> None:
    statistical = {
        "batch-504": [_record(1, 3.2), _record(2, 3.2), _record(3, 3.2)],
        "batch-1008": [_record(1, 3.1), _record(2, 3.1), _record(3, 3.1)],
        "batch-2016": [_record(1, 3.0), _record(2, 3.0), _record(3, 3.0)],
    }
    aggregates = {
        "batch-504": _aggregate(3.2),
        "batch-1008": _aggregate(3.1),
        "batch-2016": _aggregate(3.0),
    }
    decision = _selection(statistical, aggregates)
    assert decision["selected"] == "batch-2016"
    assert decision["grid_edge"] is True
    assert decision["status"] == "PROVISIONAL_GRID_EDGE_REQUIRES_LARGER_BATCH_CONTROL"
