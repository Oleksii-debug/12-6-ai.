from __future__ import annotations

import json

import pytest

from twelve_six.postbase_hypothesis import HypothesisSearch


def test_initially_preferred_wrong_hypothesis_is_rejected_after_objective_evidence():
    search = HypothesisSearch()
    wrong = search.propose(
        "the evaluator applies addition before multiplication",
        assumptions=("the fixture uses ordinary integer arithmetic",),
        initial_score=0.85,
    )
    correct = search.propose(
        "the evaluator applies multiplication before addition",
        assumptions=("the fixture uses ordinary integer arithmetic",),
        initial_score=0.60,
    )
    assert search.best().id == wrong.id

    observed = 2 + 3 * 4
    failed = search.test(
        wrong.id,
        name="operator precedence code fixture",
        prediction=20,
        observed=observed,
        weight=0.55,
        hard=True,
    )
    assert failed.passed is False
    evidence = search.evidence(failed.evidence_id)
    assert evidence.kind == "contradiction"
    assert evidence.hard is True
    assert search.hypothesis(wrong.id).contradiction_ids

    rejected = search.reject(
        wrong.id,
        "hard deterministic fixture falsified the predicted result",
        evidence_ids=(failed.evidence_id,),
    )
    assert rejected.status == "rejected"
    assert rejected.score == 0.0

    passed = search.test(
        correct.id,
        name="operator precedence code fixture",
        prediction=14,
        observed=observed,
        weight=0.30,
        hard=True,
    )
    assert passed.passed is True
    assert search.best().id == correct.id
    assert search.best().score == pytest.approx(0.90)


def test_branch_preserves_parent_relation_and_can_change_assumptions():
    search = HypothesisSearch()
    root = search.propose("boundary failure", assumptions=("input is non-empty",), initial_score=0.5)
    branch = search.branch(
        root.id,
        "boundary failure is an off-by-one error",
        assumptions=("indexing is zero-based",),
        initial_score=0.7,
    )
    assert branch.parent_id == root.id
    assert branch.assumptions == ("indexing is zero-based",)
    assert root.status == "active"


def test_critique_records_text_and_score_history():
    search = HypothesisSearch()
    hypothesis = search.propose("cache invalidation is missing", initial_score=0.7)
    critique = search.critique(
        hypothesis.id,
        "the claim does not explain why only one key is stale",
        score_delta=-0.2,
    )
    updated = search.hypothesis(hypothesis.id)
    assert critique.id in updated.critique_ids
    assert updated.score == pytest.approx(0.5)
    assert len(updated.score_history) == 2
    assert updated.score_history[-1].reason == f"critique {critique.id}"


def test_test_records_support_contradiction_and_evidence_linkage():
    search = HypothesisSearch()
    hypothesis = search.propose("predicate is even", initial_score=0.5)
    record = search.test(
        hypothesis.id,
        name="parity fixture",
        prediction=True,
        observed=False,
        weight=0.4,
    )
    updated = search.hypothesis(hypothesis.id)
    assert record.id in updated.test_ids
    assert record.evidence_id in updated.evidence_ids
    assert updated.score == pytest.approx(0.1)
    assert updated.score_history[-1].evidence_id == record.evidence_id
    contradiction = search.contradiction(updated.contradiction_ids[0])
    assert contradiction.evidence_id == record.evidence_id


def test_revise_creates_child_and_marks_active_parent_revised():
    search = HypothesisSearch()
    parent = search.propose("all failures come from parsing", initial_score=0.4)
    revised = search.revise(
        parent.id,
        "failures come from parsing only when the token stream is malformed",
        assumptions=("tokenization already completed",),
        initial_score=0.65,
    )
    assert search.hypothesis(parent.id).status == "revised"
    assert revised.parent_id == parent.id
    assert revised.status == "active"
    assert revised.score == pytest.approx(0.65)
    assert search.best().id == revised.id


def test_rejected_hypothesis_can_be_revised_without_erasing_rejection_history():
    search = HypothesisSearch()
    parent = search.propose("always returns zero", initial_score=0.8)
    failed = search.test(
        parent.id,
        name="return fixture",
        prediction=0,
        observed=1,
        weight=0.6,
        hard=True,
    )
    search.reject(parent.id, "fixture contradiction", evidence_ids=(failed.evidence_id,))
    revised = search.revise(
        parent.id,
        "returns zero only for empty input",
        initial_score=0.55,
    )
    assert search.hypothesis(parent.id).status == "rejected"
    assert revised.parent_id == parent.id
    assert revised.status == "active"


def test_reject_refuses_foreign_evidence():
    search = HypothesisSearch()
    first = search.propose("first", initial_score=0.5)
    second = search.propose("second", initial_score=0.5)
    record = search.test(
        first.id,
        name="fixture",
        prediction=1,
        observed=2,
        weight=0.2,
    )
    with pytest.raises(ValueError, match="does not belong"):
        search.reject(second.id, "wrong evidence", evidence_ids=(record.evidence_id,))


def test_export_is_deterministic_and_contains_required_reasoning_state():
    search = HypothesisSearch()
    root = search.propose("root", assumptions=("a",), initial_score=0.6)
    child = search.branch(root.id, "child")
    search.critique(child.id, "needs an objective check")
    search.test(
        child.id,
        name="logic fixture",
        prediction=False,
        observed=True,
        weight=0.3,
        hard=True,
    )
    first = json.dumps(search.export(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(search.export(), sort_keys=True, separators=(",", ":"))
    assert first == second
    exported = search.export()
    assert exported["schema"] == "12-6.postbase-hypothesis-search.v1"
    assert exported["hypotheses"][1]["parent_id"] == root.id
    assert exported["hypotheses"][1]["score_history"]
    assert exported["evidence"]
    assert exported["contradictions"]
