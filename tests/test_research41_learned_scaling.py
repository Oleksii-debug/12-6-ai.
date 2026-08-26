from __future__ import annotations

import math

from tools.research41_learned_scaling import (
    BUDGETS,
    COUNTS,
    TOK_VOCAB,
    batch,
    efficiencies,
    specs,
)


def test_learned_control_family_keeps_target_scale_and_fixed_vocab_context() -> None:
    family = specs()
    assert tuple(item.parameter_count() for item in family) == COUNTS
    assert {item.vocab_size for item in family} == {TOK_VOCAB}
    assert {item.max_seq_len for item in family} == {256}


def test_cyclic_bpe_batch_is_deterministic_and_uses_nonbyte_ids() -> None:
    stream = [0, 1, 255, 256, 471, 17, 91]
    first = batch(stream, 7)
    second = batch(stream, 7)
    assert first.equal(second)
    assert tuple(first.shape) == (4, 64)
    assert first.max().item() == 471


def test_efficiency_report_selects_one_observed_model_at_common_budget() -> None:
    cells = []
    for model_index, parameters in enumerate(COUNTS):
        observations = [{
            "requested_token_budget": 0,
            "optimized_tokens": 0,
            "validation_loss": 6.0,
            "compute_proxy": 0,
        }]
        for checkpoint_index, budget in enumerate(BUDGETS, 1):
            tokens = budget + 100
            loss = 6.0 - 0.1 * checkpoint_index - 0.01 * model_index
            observations.append({
                "requested_token_budget": budget,
                "optimized_tokens": tokens,
                "validation_loss": loss,
                "compute_proxy": 6 * parameters * tokens,
            })
        cells.append({"parameters": parameters, "observations": observations})
    result = efficiencies(cells)
    winner = result["best_validation_improvement_per_compute"]["winner"]
    assert winner["parameters"] in COUNTS
    assert winner["requested_token_budget"] == BUDGETS[-1]
    assert math.isfinite(winner["validation_improvement_per_compute_proxy"])
