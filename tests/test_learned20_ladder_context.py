from __future__ import annotations

from twelve_six.learned20_ladder_context import validate_smaller_ladder_context


def _context(*, mode: str = "CONTEXT_ONLY") -> dict:
    comparison = {
        "authority_identity": "milestone150-learned-base-ladder-v1",
        "comparison_mode": mode,
        "budget_caveat": (
            "100K/500K/1M ladder used a different historical corpus/evaluation and fixed "
            "1000-step CPU budget; values are contextual controls, not a direct 20M quality ordering."
        ),
        "direct_quality_ordering_claimed": False,
        "rungs": [
            {
                "model_identity": "m150-100k",
                "parameter_count": 95_568,
                "optimized_target_exposure": 256_000,
                "best_bpb": 6.1,
                "evaluation_identity": "m150-eval-v1",
                "data_identity": "m150-data-v1",
                "tokenizer_identity": "s0-byte-v1",
            },
            {
                "model_identity": "m150-500k",
                "parameter_count": 467_808,
                "optimized_target_exposure": 256_000,
                "best_bpb": 5.7,
                "evaluation_identity": "m150-eval-v1",
                "data_identity": "m150-data-v1",
                "tokenizer_identity": "s0-byte-v1",
            },
            {
                "model_identity": "m150-1m",
                "parameter_count": 1_037_696,
                "optimized_target_exposure": 256_000,
                "best_bpb": 5.4,
                "evaluation_identity": "m150-eval-v1",
                "data_identity": "m150-data-v1",
                "tokenizer_identity": "s0-byte-v1",
            },
        ],
    }
    return {"smaller_ladder_comparison": comparison}


def test_context_only_smaller_ladder_with_explicit_budget_caveat_is_valid() -> None:
    assert validate_smaller_ladder_context(_context()) == []


def test_context_only_comparison_cannot_claim_direct_quality_ordering() -> None:
    d06 = _context()
    d06["smaller_ladder_comparison"]["direct_quality_ordering_claimed"] = True
    assert (
        "bounded_pilot.d06.smaller_ladder.context_only_quality_claim_forbidden"
        in validate_smaller_ladder_context(d06)
    )


def test_budget_caveat_is_mandatory() -> None:
    d06 = _context()
    d06["smaller_ladder_comparison"]["budget_caveat"] = ""
    assert (
        "bounded_pilot.d06.smaller_ladder.budget_caveat_missing"
        in validate_smaller_ladder_context(d06)
    )


def test_smaller_ladder_rung_must_actually_be_smaller_than_20m() -> None:
    d06 = _context()
    d06["smaller_ladder_comparison"]["rungs"][0]["parameter_count"] = 20_613_440
    assert (
        "bounded_pilot.d06.smaller_ladder.rungs.0.parameter_count_invalid"
        in validate_smaller_ladder_context(d06)
    )


def test_matched_mode_rejects_false_identity_equivalence() -> None:
    d06 = _context(mode="MATCHED")
    comparison = d06["smaller_ladder_comparison"]
    comparison["learned20_reference"] = {
        "evaluation_identity": "learned20-eval",
        "data_identity": "learned20-data",
        "tokenizer_identity": "learned20-tokenizer",
    }
    blockers = validate_smaller_ladder_context(d06)
    assert "bounded_pilot.d06.smaller_ladder.rungs.0.evaluation_identity_not_matched" in blockers
    assert "bounded_pilot.d06.smaller_ladder.rungs.0.data_identity_not_matched" in blockers
    assert "bounded_pilot.d06.smaller_ladder.rungs.0.tokenizer_identity_not_matched" in blockers
