from types import SimpleNamespace

import pytest
import torch
from torch import nn

from twelve_six.memorization import (
    aggregate_scores,
    build_canary_suite,
    epoch_schedule,
    hashed_training_probe,
    score_canary,
    stop_diagnostic,
    training_canary_records,
)
from twelve_six.tokenization import ByteTokenizer


class UniformLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.spec = SimpleNamespace(max_seq_len=256)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        batch, time = input_ids.shape
        logits = torch.zeros(batch, time, 256, device=input_ids.device)
        return SimpleNamespace(logits=logits + self.anchor * 0)


def _synthetic_score(
    *,
    canary_id: str,
    exposure: int,
    nll: float,
    rank: int,
    exact: bool,
) -> dict:
    return {
        "canary_id": canary_id,
        "control": exposure == 0,
        "exposure_per_cycle": exposure,
        "observed_exposures": exposure,
        "continuation_sha256": "a" * 64,
        "nll_per_token": nll,
        "rank": rank,
        "candidate_count": 16,
        "exact_short_continuation": exact,
    }


def test_suite_is_deterministic_public_manifest_omits_canary_text():
    first = build_canary_suite(replicas=2, continuation_chars=6)
    second = build_canary_suite(replicas=2, continuation_chars=6)

    assert first.suite_sha256 == second.suite_sha256
    manifest = first.public()
    assert manifest["suite_sha256"] == first.suite_sha256
    assert all("prefix" not in item for item in manifest["canaries"])
    assert all("continuation" not in item for item in manifest["canaries"])
    assert all("text" not in item for item in manifest["canaries"])


def test_unseen_controls_never_enter_training_and_seen_frequency_is_exact():
    suite = build_canary_suite(
        exposures=(0, 1, 2, 4),
        replicas=2,
        continuation_chars=6,
    )
    expanded = training_canary_records(suite)
    counts: dict[str, int] = {}
    for record in expanded:
        counts[record["canary_id"]] = counts.get(record["canary_id"], 0) + 1

    for canary in suite.canaries:
        if canary.control:
            assert canary.canary_id not in counts
        else:
            assert counts[canary.canary_id] == canary.exposure_per_cycle

    schedule = epoch_schedule(
        [{"id": "base", "text": "project authored base sequence for deterministic mixing"}],
        suite,
        epoch=0,
        seed=7,
    )
    assert len(schedule) == 1 + len(expanded)


def test_canary_scorer_reports_rank_and_restores_model_mode():
    suite = build_canary_suite(exposures=(0, 1), replicas=2, continuation_chars=4)
    model = UniformLM()
    tokenizer = ByteTokenizer()
    model.train()

    score = score_canary(
        model,
        tokenizer,
        suite.canaries[0],
        observed_exposures=0,
        alternative_count=3,
    )

    assert model.training is True
    assert score["candidate_count"] == 4
    assert 1 <= score["rank"] <= 4
    assert "continuation" not in score


def test_non_canary_training_passage_report_is_hash_only():
    model = UniformLM()
    tokenizer = ByteTokenizer()
    rows = [
        {
            "id": "row-a",
            "text": "ordinary project-authored training passage alpha with enough bytes",
            "content_sha256": "1" * 64,
        },
        {
            "id": "row-b",
            "text": "ordinary project-authored training passage beta with enough bytes",
            "content_sha256": "2" * 64,
        },
    ]

    report = hashed_training_probe(
        model,
        tokenizer,
        rows,
        sample_count=2,
        seed=20260826,
        width=4,
    )

    assert report["text_emitted"] is False
    assert report["sample_count"] == 2
    assert all("text" not in item for item in report["items"])
    assert {item["content_sha256"] for item in report["items"]} == {"1" * 64, "2" * 64}


def test_stop_diagnostic_requires_improving_validation_and_two_control_relative_signals():
    scores = [
        _synthetic_score(canary_id="u0", exposure=0, nll=5.0, rank=10, exact=False),
        _synthetic_score(canary_id="u1", exposure=0, nll=5.1, rank=11, exact=False),
        _synthetic_score(canary_id="r0", exposure=16, nll=2.0, rank=1, exact=True),
        _synthetic_score(canary_id="r1", exposure=16, nll=2.1, rank=1, exact=True),
    ]
    curve = aggregate_scores(scores)

    decision = stop_diagnostic(curve, previous_bpb=7.0, current_bpb=6.8)
    assert decision["disproportionate_memorization"] is True
    assert decision["diagnostic_stop"] is True
    assert sum(decision["signals"].values()) >= 2

    no_validation_gain = stop_diagnostic(curve, previous_bpb=6.7, current_bpb=6.8)
    assert no_validation_gain["diagnostic_stop"] is False


def test_invalid_suite_without_unseen_controls_fails_closed():
    with pytest.raises(ValueError, match="unseen control"):
        build_canary_suite(exposures=(1, 2, 4))
