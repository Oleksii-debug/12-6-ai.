from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "evidence/milestone221/learned-base-ladder-v3.json"


def test_milestone221_terminal_admission_boundaries() -> None:
    report = json.loads(RECORD.read_text(encoding="utf-8"))
    assert set(report["rungs"]) == {"100k", "500k", "1m"}
    assert [x["scale"] for x in report["directly_comparable_same_recipe_ranking"]] == [
        "1m",
        "500k",
        "100k",
    ]
    assert all(
        x["optimized_tokens"] == 948_504
        for x in report["directly_comparable_same_recipe_ranking"]
    )

    three = report["different_token_budget_learned_evidence"]["3m"]
    assert three["workflow_conclusion"] == "success"
    assert three["admission_status"] == "NOT_ADMITTED_PENDING_VERIFY219"
    assert three["actual_checkpoints"][-1]["actual"] == 131_938
    assert three["best_checkpoint"]["aggregate_bpb"] == 2.2859499700392583
    assert three["first_party_logits_verification"]["status"] == (
        "NOT_RETAINED_IN_LEARN191_ARTIFACT"
    )
    assert "3m" not in {
        x["scale"] for x in report["directly_comparable_same_recipe_ranking"]
    }

    ten = report["different_token_budget_learned_evidence"]["10m"]
    assert ten["admission_status"] == "NOT_ADMITTED_NO_VERIFY218"
    assert ten["checkpoint211"]["full_10m_retraining_performed"] is False


def test_milestone221_memorization_is_separate_diagnostic() -> None:
    report = json.loads(RECORD.read_text(encoding="utf-8"))
    authority = report["terminal_memorization_authority"]
    assert authority["conclusion"] == "success"
    assert "separate" in authority["boundary"]
    for scale, rung in report["rungs"].items():
        diag = rung["memorization"]
        assert diag["authority"] == "RECOVER-178"
        assert diag["diagnostic_stop"] is True
        assert diag["disproportionate_memorization"] is True
        assert diag["privacy_claim"] == "NONE"
        assert diag["evaluation_non_mutating"] is True
        if scale == "100k":
            assert diag["exact_recovery_lift"] == 0.0
        else:
            assert diag["exact_recovery_lift"] == 1.0
