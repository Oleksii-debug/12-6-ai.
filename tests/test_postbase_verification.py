from __future__ import annotations

from dataclasses import dataclass

import pytest

from twelve_six.postbase import (
    CandidateFact,
    Claim,
    ClaimDisposition,
    ConsistencyChecker,
    CrossCandidateContradictionChecker,
    ExactAnswerFixture,
    ExactAnswerFixtureVerifier,
    FinalAnswerController,
    NumericCalculatorVerifier,
    NumericCheck,
    ProvenanceRecord,
    ReasonCode,
    SourceProvenanceChecker,
    StructuredFact,
    UnitTestCodeVerifier,
    UnitTestEvidence,
    VerificationDimension,
    VerificationRequest,
    VerificationStatus,
    VerifierEnsemble,
    VerifierResult,
)
from twelve_six.postbase.verification import ClaimVerdict, safe_calculate


@dataclass(frozen=True)
class _WeakVerifier:
    verifier_id: str
    status: VerificationStatus
    deterministic: bool = False
    dimension: VerificationDimension = VerificationDimension.MODEL_JUDGMENT

    def verify(self, request: VerificationRequest) -> VerifierResult:
        return VerifierResult(
            verifier_id=self.verifier_id,
            deterministic=False,
            dimension=self.dimension,
            verdicts=tuple(
                ClaimVerdict(claim.claim_id, self.status, ()) for claim in request.claims
            ),
        )


def _request(claim_id: str = "c1", text: str = "claim") -> VerificationRequest:
    return VerificationRequest(claims=(Claim(claim_id, text),))


def test_exact_fixture_pass_marks_claim_verified() -> None:
    request = VerificationRequest(
        claims=(Claim("answer", "answer is 42"),),
        exact_fixtures=(ExactAnswerFixture("answer", 42, 42),),
    )
    result = VerifierEnsemble((ExactAnswerFixtureVerifier(),)).verify(request)
    plan = FinalAnswerController().build_plan(request, result)
    assert result.status is VerificationStatus.PASS
    assert plan.verified_claim_ids == ("answer",)
    assert plan.claims[0].disposition is ClaimDisposition.VERIFIED


def test_weak_verifier_cannot_override_exact_deterministic_failure() -> None:
    request = VerificationRequest(
        claims=(Claim("answer", "answer is 42"),),
        exact_fixtures=(ExactAnswerFixture("answer", 42, 41),),
    )
    result = VerifierEnsemble(
        (ExactAnswerFixtureVerifier(), _WeakVerifier("weak_judge", VerificationStatus.PASS))
    ).verify(request)
    claim = result.claim("answer")
    plan = FinalAnswerController().build_plan(request, result)
    assert result.status is VerificationStatus.FAIL
    assert claim.deterministic_failure is True
    assert ReasonCode.HARD_DETERMINISTIC_FAILURE in claim.reason_codes
    assert ReasonCode.VERIFIER_DISAGREEMENT in claim.reason_codes
    assert plan.claims[0].disposition is ClaimDisposition.REJECTED


def test_unit_test_verifier_catches_error_even_when_weak_judge_passes() -> None:
    request = VerificationRequest(
        claims=(Claim("code", "implementation is correct"),),
        unit_tests=(UnitTestEvidence("code", "pytest", exit_code=1, passed=9, failed=1),),
    )
    result = VerifierEnsemble(
        (UnitTestCodeVerifier(), _WeakVerifier("looks_good", VerificationStatus.PASS))
    ).verify(request)
    assert result.status is VerificationStatus.FAIL
    assert ReasonCode.UNIT_TESTS_FAIL in result.claim("code").reason_codes


def test_conflicting_weak_verifiers_are_conflict_not_verified() -> None:
    request = _request()
    result = VerifierEnsemble(
        (
            _WeakVerifier("judge_a", VerificationStatus.PASS),
            _WeakVerifier("judge_b", VerificationStatus.FAIL),
        )
    ).verify(request)
    plan = FinalAnswerController().build_plan(request, result)
    assert result.status is VerificationStatus.CONFLICT
    assert plan.claims[0].disposition is ClaimDisposition.CONFLICTED
    assert not plan.verified_claim_ids


def test_heuristic_pass_alone_remains_proposed() -> None:
    request = _request()
    result = VerifierEnsemble((_WeakVerifier("judge", VerificationStatus.PASS),)).verify(request)
    plan = FinalAnswerController().build_plan(request, result)
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert ReasonCode.HEURISTIC_ONLY_SUPPORT in result.claim("c1").reason_codes
    assert plan.proposed_claim_ids == ("c1",)


def test_numeric_calculator_is_deterministic_and_rejects_calls() -> None:
    request = VerificationRequest(
        claims=(Claim("math", "calculation is correct"),),
        numeric_checks=(NumericCheck("math", "(7 * 6) + 0.5", "42.5"),),
    )
    result = VerifierEnsemble((NumericCalculatorVerifier(),)).verify(request)
    assert result.status is VerificationStatus.PASS
    assert safe_calculate("2 ** 10") == 1024
    with pytest.raises(TypeError):
        safe_calculate("__import__('os').system('true')")


def test_rejected_numeric_expression_is_hard_failure() -> None:
    request = VerificationRequest(
        claims=(Claim("math", "calculation is correct"),),
        numeric_checks=(NumericCheck("math", "sum([1, 2])", 3),),
    )
    result = VerifierEnsemble((NumericCalculatorVerifier(),)).verify(request)
    assert result.status is VerificationStatus.FAIL
    assert ReasonCode.NUMERIC_EXPRESSION_REJECTED in result.claim("math").reason_codes


def test_consistency_checker_detects_structured_contradiction() -> None:
    request = VerificationRequest(
        claims=(Claim("version", "version is internally consistent"),),
        structured_facts=(
            StructuredFact("version", "version", "v1"),
            StructuredFact("version", "version", "v2"),
        ),
    )
    result = VerifierEnsemble((ConsistencyChecker(),)).verify(request)
    assert result.status is VerificationStatus.FAIL
    assert ReasonCode.INTERNAL_CONTRADICTION in result.claim("version").reason_codes


def test_provenance_pass_does_not_promote_truth_without_correctness_evidence() -> None:
    request = VerificationRequest(
        claims=(Claim("source", "source-backed statement"),),
        provenance=(
            ProvenanceRecord(
                "source",
                "fixture-a",
                "tests/fixture.json:1",
                "a" * 64,
            ),
        ),
    )
    result = VerifierEnsemble((SourceProvenanceChecker(require_content_hash=True),)).verify(request)
    plan = FinalAnswerController().build_plan(request, result)
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert ReasonCode.PROVENANCE_COMPLETE in result.claim("source").reason_codes
    assert plan.claims[0].disposition is ClaimDisposition.PROPOSED


def test_missing_provenance_fails_provenance_gate() -> None:
    request = _request("source", "must have provenance")
    result = VerifierEnsemble((SourceProvenanceChecker(),)).verify(request)
    assert result.status is VerificationStatus.FAIL
    assert ReasonCode.PROVENANCE_MISSING in result.claim("source").reason_codes


def test_cross_candidate_contradiction_is_explicit_conflict() -> None:
    request = VerificationRequest(
        claims=(Claim("candidate", "candidate outputs agree"),),
        candidate_facts=(
            CandidateFact("a", "candidate", "answer", 17),
            CandidateFact("b", "candidate", "answer", 19),
        ),
    )
    result = VerifierEnsemble((CrossCandidateContradictionChecker(),)).verify(request)
    assert result.status is VerificationStatus.CONFLICT
    assert ReasonCode.CANDIDATE_CONTRADICTION in result.claim("candidate").reason_codes


def test_agreement_and_provenance_still_do_not_claim_scientific_truth() -> None:
    request = VerificationRequest(
        claims=(Claim("hypothesis", "hypothesis is true"),),
        provenance=(ProvenanceRecord("hypothesis", "paper", "paper:section-3"),),
        candidate_facts=(
            CandidateFact("a", "hypothesis", "score", "supports"),
            CandidateFact("b", "hypothesis", "score", "supports"),
        ),
    )
    result = VerifierEnsemble(
        (SourceProvenanceChecker(), CrossCandidateContradictionChecker())
    ).verify(request)
    plan = FinalAnswerController().build_plan(request, result)
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert plan.claims[0].disposition is ClaimDisposition.PROPOSED


def test_exact_equality_does_not_conflate_bool_and_int() -> None:
    request = VerificationRequest(
        claims=(Claim("typed", "typed exact equality"),),
        exact_fixtures=(ExactAnswerFixture("typed", True, 1),),
    )
    result = VerifierEnsemble((ExactAnswerFixtureVerifier(),)).verify(request)
    assert result.status is VerificationStatus.FAIL


def test_exact_equality_is_recursive_for_container_values() -> None:
    request = VerificationRequest(
        claims=(Claim("nested", "nested exact equality"),),
        exact_fixtures=(ExactAnswerFixture("nested", {"x": [True]}, {"x": [1]}),),
    )
    result = VerifierEnsemble((ExactAnswerFixtureVerifier(),)).verify(request)
    assert result.status is VerificationStatus.FAIL


def test_unit_test_evidence_cannot_claim_zero_test_success() -> None:
    request = VerificationRequest(
        claims=(Claim("code", "tests prove code"),),
        unit_tests=(UnitTestEvidence("code", "pytest", exit_code=0, passed=0),),
    )
    result = VerifierEnsemble((UnitTestCodeVerifier(),)).verify(request)
    assert result.status is VerificationStatus.FAIL
    assert ReasonCode.UNIT_TEST_EVIDENCE_INVALID in result.claim("code").reason_codes


def test_exact_equality_is_type_strict_for_dict_keys() -> None:
    request = VerificationRequest(
        claims=(Claim("keys", "dict key exact equality"),),
        exact_fixtures=(ExactAnswerFixture("keys", {True: "x"}, {1: "x"}),),
    )
    result = VerifierEnsemble((ExactAnswerFixtureVerifier(),)).verify(request)
    assert result.status is VerificationStatus.FAIL
