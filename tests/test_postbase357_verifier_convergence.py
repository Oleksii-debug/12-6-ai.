from __future__ import annotations

from dataclasses import dataclass

from twelve_six.postbase import (
    CandidateFact,
    Claim,
    ConsistencyChecker,
    CrossCandidateContradictionChecker,
    ExactAnswerFixture,
    ExactAnswerFixtureVerifier,
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
from twelve_six.postbase.verification import ClaimVerdict


@dataclass(frozen=True, slots=True)
class _HeuristicConfidence:
    verifier_id: str = "postbase357_heuristic"
    deterministic: bool = False
    dimension: VerificationDimension = VerificationDimension.MODEL_JUDGMENT

    def verify(self, request: VerificationRequest) -> VerifierResult:
        return VerifierResult(
            self.verifier_id,
            False,
            self.dimension,
            tuple(
                ClaimVerdict(claim.claim_id, VerificationStatus.PASS, ())
                for claim in request.claims
            ),
        )


def _ensemble() -> VerifierEnsemble:
    return VerifierEnsemble(
        (
            ExactAnswerFixtureVerifier(),
            UnitTestCodeVerifier(),
            NumericCalculatorVerifier(),
            SourceProvenanceChecker(require_content_hash=True),
            ConsistencyChecker(),
            CrossCandidateContradictionChecker(),
            _HeuristicConfidence(),
        )
    )


def _base_request() -> VerificationRequest:
    return VerificationRequest(
        claims=(Claim("c", "integrated fixture"),),
        exact_fixtures=(ExactAnswerFixture("c", 42, 42),),
        unit_tests=(UnitTestEvidence("c", "pytest fixture", exit_code=0, passed=2),),
        numeric_checks=(NumericCheck("c", "6 * 7", 42),),
        provenance=(ProvenanceRecord("c", "fixture", "fixture://c", "a" * 64),),
        structured_facts=(
            StructuredFact("c", "version", "v1"),
            StructuredFact("c", "version", "v1"),
        ),
        candidate_facts=(
            CandidateFact("a", "c", "answer", 42),
            CandidateFact("b", "c", "answer", 42),
        ),
    )


def test_all_required_verifiers_converge_to_pass() -> None:
    result = _ensemble().verify(_base_request())
    claim = result.claim("c")
    assert result.status is VerificationStatus.PASS
    assert claim.deterministic_correctness_pass is True
    assert claim.deterministic_failure is False
    assert ReasonCode.EXACT_MATCH in claim.reason_codes
    assert ReasonCode.UNIT_TESTS_PASS in claim.reason_codes
    assert ReasonCode.NUMERIC_MATCH in claim.reason_codes
    assert ReasonCode.PROVENANCE_COMPLETE in claim.reason_codes
    assert ReasonCode.CONSISTENT_FACTS in claim.reason_codes
    assert ReasonCode.CANDIDATES_AGREE in claim.reason_codes


def test_exact_failure_dominates_heuristic_confidence() -> None:
    request = _base_request()
    request = VerificationRequest(
        claims=request.claims,
        exact_fixtures=(ExactAnswerFixture("c", 42, 41),),
        unit_tests=request.unit_tests,
        numeric_checks=request.numeric_checks,
        provenance=request.provenance,
        structured_facts=request.structured_facts,
        candidate_facts=request.candidate_facts,
    )
    claim = _ensemble().verify(request).claim("c")
    assert claim.status is VerificationStatus.FAIL
    assert claim.deterministic_failure is True
    assert ReasonCode.HARD_DETERMINISTIC_FAILURE in claim.reason_codes
    assert ReasonCode.VERIFIER_DISAGREEMENT in claim.reason_codes


def test_unit_test_failure_dominates_heuristic_confidence() -> None:
    request = _base_request()
    request = VerificationRequest(
        claims=request.claims,
        exact_fixtures=request.exact_fixtures,
        unit_tests=(UnitTestEvidence("c", "pytest fixture", exit_code=1, passed=1, failed=1),),
        numeric_checks=request.numeric_checks,
        provenance=request.provenance,
        structured_facts=request.structured_facts,
        candidate_facts=request.candidate_facts,
    )
    claim = _ensemble().verify(request).claim("c")
    assert claim.status is VerificationStatus.FAIL
    assert ReasonCode.UNIT_TESTS_FAIL in claim.reason_codes
    assert ReasonCode.HARD_DETERMINISTIC_FAILURE in claim.reason_codes


def test_numeric_failure_dominates_heuristic_confidence() -> None:
    request = _base_request()
    request = VerificationRequest(
        claims=request.claims,
        exact_fixtures=request.exact_fixtures,
        unit_tests=request.unit_tests,
        numeric_checks=(NumericCheck("c", "6 * 7", 41),),
        provenance=request.provenance,
        structured_facts=request.structured_facts,
        candidate_facts=request.candidate_facts,
    )
    claim = _ensemble().verify(request).claim("c")
    assert claim.status is VerificationStatus.FAIL
    assert ReasonCode.NUMERIC_MISMATCH in claim.reason_codes
    assert ReasonCode.HARD_DETERMINISTIC_FAILURE in claim.reason_codes


def test_provenance_failure_dominates_heuristic_confidence() -> None:
    request = _base_request()
    request = VerificationRequest(
        claims=request.claims,
        exact_fixtures=request.exact_fixtures,
        unit_tests=request.unit_tests,
        numeric_checks=request.numeric_checks,
        provenance=(ProvenanceRecord("c", "fixture", "fixture://c", "A" * 64),),
        structured_facts=request.structured_facts,
        candidate_facts=request.candidate_facts,
    )
    claim = _ensemble().verify(request).claim("c")
    assert claim.status is VerificationStatus.FAIL
    assert ReasonCode.PROVENANCE_HASH_INVALID in claim.reason_codes
    assert ReasonCode.HARD_DETERMINISTIC_FAILURE in claim.reason_codes


def test_consistency_failure_dominates_heuristic_confidence() -> None:
    request = _base_request()
    request = VerificationRequest(
        claims=request.claims,
        exact_fixtures=request.exact_fixtures,
        unit_tests=request.unit_tests,
        numeric_checks=request.numeric_checks,
        provenance=request.provenance,
        structured_facts=(
            StructuredFact("c", "version", "v1"),
            StructuredFact("c", "version", "v2"),
        ),
        candidate_facts=request.candidate_facts,
    )
    claim = _ensemble().verify(request).claim("c")
    assert claim.status is VerificationStatus.FAIL
    assert ReasonCode.INTERNAL_CONTRADICTION in claim.reason_codes
    assert ReasonCode.HARD_DETERMINISTIC_FAILURE in claim.reason_codes


def test_cross_candidate_contradiction_returns_conflict() -> None:
    request = _base_request()
    request = VerificationRequest(
        claims=request.claims,
        exact_fixtures=request.exact_fixtures,
        unit_tests=request.unit_tests,
        numeric_checks=request.numeric_checks,
        provenance=request.provenance,
        structured_facts=request.structured_facts,
        candidate_facts=(
            CandidateFact("a", "c", "answer", 42),
            CandidateFact("b", "c", "answer", 43),
        ),
    )
    claim = _ensemble().verify(request).claim("c")
    assert claim.status is VerificationStatus.CONFLICT
    assert claim.deterministic_failure is False
    assert ReasonCode.CANDIDATE_CONTRADICTION in claim.reason_codes
    assert ReasonCode.VERIFIER_DISAGREEMENT in claim.reason_codes


def test_non_correctness_support_remains_inconclusive() -> None:
    request = VerificationRequest(
        claims=(Claim("c", "support without correctness proof"),),
        provenance=(ProvenanceRecord("c", "fixture", "fixture://c", "a" * 64),),
        structured_facts=(StructuredFact("c", "version", "v1"),),
        candidate_facts=(
            CandidateFact("a", "c", "answer", 42),
            CandidateFact("b", "c", "answer", 42),
        ),
    )
    claim = _ensemble().verify(request).claim("c")
    assert claim.status is VerificationStatus.INCONCLUSIVE
    assert claim.deterministic_correctness_pass is False
    assert ReasonCode.NO_DETERMINISTIC_CORRECTNESS_SUPPORT in claim.reason_codes
    assert ReasonCode.HEURISTIC_ONLY_SUPPORT in claim.reason_codes
