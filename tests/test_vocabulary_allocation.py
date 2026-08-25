from __future__ import annotations

from pathlib import Path

from twelve_six.model import load_stage_config
from twelve_six.vocabulary import (
    TokenizerTradeoffPoint,
    pareto_frontier,
    rebalance_d_ff_for_vocabulary,
    vocabulary_cost,
)

ROOT = Path(__file__).resolve().parents[1]


def test_current_s1_vocab_surface_and_untied_counterfactual() -> None:
    s1 = load_stage_config(ROOT / "configs/stages/s1_100k.json")
    tied = vocabulary_cost(s1.model)
    untied = vocabulary_cost(s1.model, tied_lm_head=False)
    assert tied.total_vocabulary_parameters == 24_576
    assert untied.total_vocabulary_parameters == 49_152


def test_s1_real_bpe_vocab_can_retarget_near_100k() -> None:
    s1 = load_stage_config(ROOT / "configs/stages/s1_100k.json")
    candidate = rebalance_d_ff_for_vocabulary(
        s1.model,
        target_parameters=100_000,
        vocab_size=472,
    )
    assert candidate.model.d_ff == 112
    assert candidate.parameter_count == 99_024
    assert candidate.vocabulary_parameters == 22_656


def test_s2_1024_anchor_reallocates_budget_to_ffn() -> None:
    s2 = load_stage_config(ROOT / "configs/stages/s2_1m.json")
    candidate = rebalance_d_ff_for_vocabulary(
        s2.model,
        target_parameters=1_000_000,
        vocab_size=1024,
    )
    assert candidate.model.d_ff == 392
    assert candidate.parameter_count == 996_480
    assert candidate.vocabulary_parameters == 131_072


def _point(*, algorithm: str, vocab: int, tokens: int, repeatable: bool) -> TokenizerTradeoffPoint:
    return TokenizerTradeoffPoint(
        algorithm=algorithm,
        requested_vocab_size=512,
        actual_vocab_size=vocab,
        held_out_tokens=tokens,
        byte_baseline_tokens=520,
        token_reduction_vs_bytes=1.0 - tokens / 520,
        repeatable_artifact_identity=repeatable,
        strict_round_trip=True,
        unknown_tokens=0,
        vocabulary_parameters=vocab * 48,
        vocabulary_share_of_stage_target=vocab * 48 / 100_000,
        output_projection_work_ratio_vs_bytes=(tokens * vocab) / (520 * 256),
        source_sha="0" * 64,
        evidence_sha256="1" * 64,
    )


def test_nonrepeatable_tokenizer_is_not_on_freeze_eligible_frontier() -> None:
    byte_like = _point(algorithm="small-bpe", vocab=400, tokens=320, repeatable=True)
    repeatable = _point(algorithm="bpe", vocab=472, tokens=286, repeatable=True)
    nonrepeatable = _point(algorithm="unigram", vocab=497, tokens=284, repeatable=False)
    frontier = pareto_frontier([byte_like, repeatable, nonrepeatable])
    assert frontier == [byte_like, repeatable]
