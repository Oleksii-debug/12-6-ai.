from __future__ import annotations

from dataclasses import dataclass

import pytest

from twelve_six.postbase import (
    CandidateFact,
    ConsistencyChecker,
    CrossCandidateContradictionChecker,
    ExactAnswerFixture,
    ExactAnswerFixtureVerifier,
    HypothesisVerifierController,
    ReasonCode,
    StructuredFact,
    VerificationDimension,
    VerificationRequest,
    VerificationStatus,
    VerifierEnsemble,
    VerifierResult,
)
from twelve_six.postbase.verification import ClaimVerdict
from twelve_six.postbase_hypothesis import HypothesisSearch


def _exact_request(controller, hypothesis_id, expected, actual):
    claim = controller.claim_for(hypothesis_id)
    return VerificationRequest(
        claims=(claim,),
        exact_fixtures=(ExactAnswerFixture(claim.claim_id, expected, actual),),
    )


def test_high_heuristic_score_cannot_survive_hard_deterministic_fail() -> None:
    search = HypothesisSearch()
    wrong = search.propose("addition precedes multiplication", initial_score=0.95)
    correct = search.propose("multiplication precedes addition", initial_score=0.40)
    assert search.best().id == wrong.id

    controller = HypothesisVerifierController(
        search,
        VerifierEnsemble((ExactAnswerFixtureVerifier(),)),
    )
    result = controller.verify(wrong.id, _exact_request(controller, wrong.id, 14, 20))

    assert result.status is VerificationStatus.FAIL
    assert result.evidence.hypothesis_version_id == wrong.version_id
    assert result.evidence.deterministic_failure is True
    assert result.rejected is True
    assert search.hypothesis(wrong.id).status == "rejected"
    assert controller.preferred().id == correct.id


def test_pass_and_inconclusive_are_preserved_as_explicit_states() -> None:
    search = HypothesisSearch()
    passed = search.propose("fixture result is fourteen", initial_score=0.5)
    unknown = search.propose("fixture has an unspecified property", initial_score=0.4)
    controller = HypothesisVerifierController(
        search,
        VerifierEnsemble((ExactAnswerFixtureVerifier(),)),
    )

    pass_result = controller.verify(passed.id, _exact_request(controller, passed.id, 14, 14))
    unknown_claim = controller.claim_for(unknown.id)
    inconclusive_result = controller.verify(
        unknown.id,
        VerificationRequest(claims=(unknown_claim,)),
    )

    assert pass_result.status is VerificationStatus.PASS
    assert pass_result.evidence.deterministic_correctness_pass is True
    assert inconclusive_result.status is VerificationStatus.INCONCLUSIVE
    assert inconclusive_result.rejected is False


def test_contradictory_candidate_evidence_is_conflict_not_silent_pass() -> None:
    search = HypothesisSearch()
    hypothesis = search.propose("candidate observations agree", initial_score=0.8)
    controller = HypothesisVerifierController(
        search,
        VerifierEnsemble((CrossCandidateContradictionChecker(),)),
    )
    claim = controller.claim_for(hypothesis.id)
    request = VerificationRequest(
        claims=(claim,),
        candidate_facts=(
            CandidateFact("candidate-a", claim.claim_id, "observed", 14),
            CandidateFact("candidate-b", claim.claim_id, "observed", 20),
        ),
    )

    result = controller.verify(hypothesis.id, request)

    assert result.status is VerificationStatus.CONFLICT
    assert result.rejected is False
    assert search.hypothesis(hypothesis.id).status == "active"


def test_pass_plus_deterministic_contradiction_is_hard_rejection() -> None:
    search = HypothesisSearch()
    hypothesis = search.propose("all deterministic evidence is consistent", initial_score=0.99)
    controller = HypothesisVerifierController(
        search,
        VerifierEnsemble((ExactAnswerFixtureVerifier(), ConsistencyChecker())),
    )
    claim = controller.claim_for(hypothesis.id)
    request = VerificationRequest(
        claims=(claim,),
        exact_fixtures=(ExactAnswerFixture(claim.claim_id, 14, 14),),
        structured_facts=(
            StructuredFact(claim.claim_id, "value", 14),
            StructuredFact(claim.claim_id, "value", 20),
        ),
    )

    result = controller.verify(hypothesis.id, request)

    assert result.status is VerificationStatus.FAIL
    assert result.evidence.deterministic_failure is True
    assert ReasonCode.VERIFIER_DISAGREEMENT in result.ensemble_result.claim(
        claim.claim_id
    ).reason_codes
    assert search.hypothesis(hypothesis.id).status == "rejected"
    assert search.best() is None


def test_revised_hypothesis_gets_new_immutable_version_and_no_inherited_verdict() -> None:
    search = HypothesisSearch()
    original = search.propose("always returns zero", initial_score=0.9)
    controller = HypothesisVerifierController(
        search,
        VerifierEnsemble((ExactAnswerFixtureVerifier(),)),
    )
    old_request = _exact_request(controller, original.id, 1, 0)
    failed = controller.verify(original.id, old_request)
    old_version_id = original.version_id

    revised = search.revise(
        original.id,
        "returns zero only for empty input",
        initial_score=0.6,
    )

    assert failed.evidence.hypothesis_version_id == old_version_id
    assert search.hypothesis(original.id).version_id == old_version_id
    assert revised.version == original.version + 1
    assert revised.version_id != old_version_id
    assert revised.verifier_evidence_ids == ()
    with pytest.raises(ValueError, match="current hypothesis version"):
        controller.verify(revised.id, old_request)


def test_search_best_is_fail_closed_even_before_controller_rejection() -> None:
    search = HypothesisSearch()
    high = search.propose("high heuristic", initial_score=1.0)
    low = search.propose("lower heuristic", initial_score=0.1)
    search.record_verifier_evidence(
        high.id,
        claim_id=high.version_id,
        status="FAIL",
        reason_codes=("HARD_DETERMINISTIC_FAILURE",),
        deterministic_failure=True,
        deterministic_correctness_pass=False,
        verifier_ids=("local_fixture",),
    )
    assert search.hypothesis(high.id).status == "active"
    assert search.best().id == low.id


@dataclass(frozen=True)
class _ForbiddenModelJudge:
    verifier_id: str = "external_model_judge"
    deterministic: bool = False
    dimension: VerificationDimension = VerificationDimension.MODEL_JUDGMENT

    def verify(self, request: VerificationRequest) -> VerifierResult:
        return VerifierResult(
            self.verifier_id,
            self.deterministic,
            self.dimension,
            tuple(
                ClaimVerdict(claim.claim_id, VerificationStatus.PASS, ())
                for claim in request.claims
            ),
        )


def test_external_or_model_judge_verifiers_are_rejected_by_integration() -> None:
    with pytest.raises(ValueError, match="deterministic local verifiers only"):
        HypothesisVerifierController(
            HypothesisSearch(),
            VerifierEnsemble((_ForbiddenModelJudge(),)),
        )
