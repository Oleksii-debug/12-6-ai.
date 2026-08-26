from __future__ import annotations

import json

import pytest

from twelve_six.postbase_deliberation import Budget, Config, Response, Verification
from twelve_six.postbase_hypothesis_deliberation import (
    EvidenceCheck,
    HypothesisDeliberationController,
)


class FixtureModel:
    def generate(self, request):
        if request.stage == "propose":
            text = (
                "the evaluator applies addition before multiplication"
                if request.branch_id == "branch-1"
                else "the evaluator applies multiplication before addition"
            )
        elif request.stage == "critique":
            text = "check operator precedence with the deterministic arithmetic fixture"
        elif request.stage == "revise":
            text = "the evaluator applies multiplication before addition after revision"
        else:
            raise AssertionError(request.stage)
        return Response(text, len(text.split()), f"PRIVATE:{request.stage}:{request.branch_id}")


class FixtureVerifier:
    def evaluate(self, task, text, branch_id, iteration):
        del task, branch_id, iteration
        if text == "the evaluator applies addition before multiplication":
            return Verification(0.85, 1.0, "initially preferred but wrong")
        if text == "the evaluator applies multiplication before addition":
            return Verification(0.60, 1.0, "initially secondary")
        return Verification(0.40, 1.0, "revision remains below retained correct branch")


class ArithmeticEvidence:
    def checks(self, task, hypothesis_id, statement, iteration):
        del task, hypothesis_id, iteration
        observed = 2 + 3 * 4
        if "addition before multiplication" in statement:
            return (
                EvidenceCheck(
                    "operator precedence fixture",
                    20,
                    observed,
                    0.55,
                    True,
                    "local_fixture",
                ),
            )
        if "multiplication before addition" in statement:
            return (
                EvidenceCheck(
                    "operator precedence fixture",
                    14,
                    observed,
                    0.30,
                    True,
                    "local_fixture",
                ),
            )
        return ()


def controller():
    return HypothesisDeliberationController(
        FixtureModel(),
        FixtureVerifier(),
        ArithmeticEvidence(),
        config=Config(
            initial_branches=2,
            target_score=None,
            convergence_rounds=100,
        ),
    )


def budget():
    return Budget(
        model_calls=10,
        generated_tokens=200,
        tool_calls=0,
        candidate_branches=4,
    )


def test_wrong_initial_preference_rejected_and_best_retained():
    result = controller().run("determine the evaluator precedence rule", budget())
    state = result["trace"]["hypothesis_state"]

    assert result["final_text"] == "the evaluator applies multiplication before addition"
    assert result["score"] == pytest.approx(0.90)
    assert state["hypotheses"][0]["status"] == "rejected"
    assert state["hypotheses"][0]["score_history"][0]["score"] == pytest.approx(0.85)
    assert state["hypotheses"][0]["score_history"][-1]["score"] == 0.0
    assert state["contradictions"]
    assert result["trace"]["budget_consumed"]["rejections"] == 1
    assert result["trace"]["budget_consumed"]["revisions"] == 1
    assert len(state["hypotheses"]) == 3
    assert result["trace"]["retained_best_history"][0]["hypothesis_id"] == "H001"
    assert result["hypothesis_id"] == "H002"


def test_public_trace_excludes_private_scratch_and_critique_text():
    result = controller().run("privacy", budget())
    rendered = json.dumps(result, sort_keys=True)

    assert "PRIVATE:" not in rendered
    assert "private_scratch" not in rendered
    assert "check operator precedence with the deterministic arithmetic fixture" not in rendered
    assert result["trace"]["hypothesis_state"]["critiques"]


def test_budget_accounting_is_exact_for_fixture():
    result = controller().run("budget", budget())
    used = result["trace"]["budget_consumed"]

    assert used["model_calls"] == 6
    assert used["candidate_branches"] == 3
    assert used["evidence_tests"] == 3
    assert used["contradictions"] == 1
    assert used["rejections"] == 1
    assert used["revisions"] == 1
