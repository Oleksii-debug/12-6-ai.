from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from twelve_six.postbase_deliberation import (
    AdapterContractError,
    Budget,
    Config,
    DeliberationController,
    DeterministicMockAdapter,
    Request,
    Response,
    ToolCall,
    Verification,
)


@dataclass(frozen=True)
class ConstantVerifier:
    score: float = 0.5
    confidence: float = 1.0

    def evaluate(self, task, text, branch_id, iteration):
        del task, text, branch_id, iteration
        return Verification(self.score, self.confidence, "constant verifier")


def controller(verifier=None, branches=1):
    return DeliberationController(
        DeterministicMockAdapter(),
        verifier or ConstantVerifier(),
        config=Config(
            initial_branches=branches,
            target_score=None,
            convergence_delta=0.0,
            convergence_rounds=100,
        ),
    )


def test_larger_budget_more_search_not_assumed_better():
    small = controller().run("test", Budget(model_calls=3, generated_tokens=100, candidate_branches=2))
    large = controller().run("test", Budget(model_calls=7, generated_tokens=300, candidate_branches=4))
    assert large["trace"]["budget_consumed"]["model_calls"] > small["trace"]["budget_consumed"]["model_calls"]
    assert large["trace"]["budget_consumed"]["candidate_branches"] > small["trace"]["budget_consumed"]["candidate_branches"]
    assert large["score"] == small["score"] == 0.5


def test_early_verifier_termination():
    c = DeliberationController(
        DeterministicMockAdapter(),
        ConstantVerifier(1.0),
        config=Config(initial_branches=1, target_score=0.9),
    )
    result = c.run("early", Budget(model_calls=10, generated_tokens=100, candidate_branches=5))
    assert result["trace"]["stop_reason"] == "verifier_target"
    assert result["trace"]["budget_consumed"]["model_calls"] == 1


def test_private_scratch_not_public():
    result = controller().run("privacy", Budget(model_calls=3, generated_tokens=100, candidate_branches=2))
    rendered = json.dumps(result, sort_keys=True)
    assert "internal:" not in rendered
    assert "private_scratch_sha256" in rendered


class ToolAdapter:
    def generate(self, request: Request):
        if request.stage == "propose" and not request.tool_results:
            return Response("need tool", 2, "secret-scratch", (ToolCall("lookup", {"k": "a"}),))
        if request.stage == "propose":
            return Response("tool-informed candidate", 2, "secret-after-tool")
        return Response("unused", 1, "secret-unused")


class Tool:
    def execute(self, name, arguments):
        assert name == "lookup" and arguments == {"k": "a"}
        return "tool-secret-payload"


def test_tool_budget_and_hash_only_trace():
    c = DeliberationController(
        ToolAdapter(), ConstantVerifier(1.0), tools=Tool(),
        config=Config(initial_branches=1, target_score=0.9),
    )
    result = c.run(
        "tool", Budget(model_calls=4, generated_tokens=50, tool_calls=1, candidate_branches=1)
    )
    assert result["trace"]["budget_consumed"]["model_calls"] == 2
    assert result["trace"]["budget_consumed"]["tool_calls"] == 1
    rendered = json.dumps(result, sort_keys=True)
    assert "tool-secret-payload" not in rendered
    assert "secret-scratch" not in rendered


class Overspend:
    def generate(self, request: Request):
        assert request.max_generated_tokens is not None
        return Response("bad", request.max_generated_tokens + 1)


def test_generated_token_contract():
    c = DeliberationController(Overspend(), ConstantVerifier(), config=Config(initial_branches=1))
    with pytest.raises(AdapterContractError):
        c.run("bounded", Budget(model_calls=2, generated_tokens=5, candidate_branches=1))


def test_multiple_initial_branches_compared():
    result = controller(branches=3).run(
        "branches", Budget(model_calls=3, generated_tokens=100, candidate_branches=3)
    )
    assert len(result["trace"]["branches_attempted"]) == 3
    assert len(result["trace"]["comparisons"][0]["rejected_candidate_ids"]) == 2


class WorseRevision:
    def generate(self, request: Request):
        if request.stage == "propose":
            return Response("good", 1, "private-good")
        if request.stage == "critique":
            return Response("critique", 1, "private-critique")
        return Response("bad", 1, "private-bad")


class WordVerifier:
    def evaluate(self, task, text, branch_id, iteration):
        del task, branch_id, iteration
        return Verification(0.9 if text == "good" else 0.1, 1.0, "word verifier")


def test_worse_revision_does_not_replace_retained_best():
    c = DeliberationController(
        WorseRevision(), WordVerifier(),
        config=Config(
            initial_branches=1, target_score=None,
            convergence_delta=0.0, convergence_rounds=100,
        ),
    )
    result = c.run("retain", Budget(model_calls=3, generated_tokens=20, candidate_branches=2))
    assert result["final_text"] == "good"
    assert result["trace"]["selected_final_candidate"] == "candidate-1"


def test_budget_needs_execution_limiter():
    with pytest.raises(ValueError):
        Budget(tool_calls=2, candidate_branches=3)
