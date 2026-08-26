from __future__ import annotations

import argparse
import json
from pathlib import Path

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
            raise ValueError(request.stage)
        return Response(text, len(text.split()), f"PRIVATE:{request.stage}:{request.branch_id}")


class FixtureVerifier:
    def evaluate(self, task, text, branch_id, iteration):
        del task, branch_id, iteration
        if text == "the evaluator applies addition before multiplication":
            return Verification(0.85, 1.0, "initially preferred but wrong")
        if text == "the evaluator applies multiplication before addition":
            return Verification(0.60, 1.0, "initially secondary")
        return Verification(0.40, 1.0, "revision lower than retained correct branch")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controller = HypothesisDeliberationController(
        FixtureModel(),
        FixtureVerifier(),
        ArithmeticEvidence(),
        config=Config(initial_branches=2, target_score=None, convergence_rounds=100),
    )
    result = controller.run(
        "determine the evaluator precedence rule",
        Budget(model_calls=10, generated_tokens=200, tool_calls=0, candidate_branches=4),
    )

    state = result["trace"]["hypothesis_state"]
    assert state["hypotheses"][0]["status"] == "rejected"
    assert state["hypotheses"][0]["score_history"][0]["score"] == 0.85
    assert state["hypotheses"][0]["score_history"][-1]["score"] == 0.0
    assert state["contradictions"]
    assert result["final_text"] == "the evaluator applies multiplication before addition"
    assert result["score"] == 0.9
    assert result["trace"]["budget_consumed"]["model_calls"] == 6
    assert result["trace"]["budget_consumed"]["candidate_branches"] == 3
    assert result["trace"]["budget_consumed"]["revisions"] == 1

    rendered = json.dumps(result, sort_keys=True)
    assert "PRIVATE:" not in rendered
    assert "private_scratch" not in rendered
    assert "check operator precedence with the deterministic arithmetic fixture" not in rendered

    report = {
        "schema": "12-6.next100081-local-free-probe.v1",
        "worker_id": "NEXT100-081-HYPOTHESIS-DELIBERATION-INTEGRATION",
        "execution_profile": "LOCAL_FREE",
        "external_teacher_api": False,
        "external_llm": False,
        "wrong_preferred_hypothesis_rejected": True,
        "retained_best_hypothesis_id": result["hypothesis_id"],
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
