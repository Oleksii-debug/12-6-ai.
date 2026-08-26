from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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

CANDIDATE_SHA = "8744c8ab4c21299ed5fd12e937ab2dadb92d574e"
CANDIDATE_SOURCE_BLOB = "e3c0504c1a6b3768c8aaea1aaaa3b3eb637eaab7"
EXECUTION_PROFILE = "LOCAL_FREE"


@dataclass(frozen=True, slots=True)
class HeuristicPassVerifier:
    verifier_id: str = "postbase357_heuristic_confidence"
    deterministic: bool = False
    dimension: VerificationDimension = VerificationDimension.MODEL_JUDGMENT

    def verify(self, request: VerificationRequest) -> VerifierResult:
        return VerifierResult(
            verifier_id=self.verifier_id,
            deterministic=False,
            dimension=self.dimension,
            verdicts=tuple(
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
            HeuristicPassVerifier(),
        )
    )


def _request(
    *,
    exact_actual: int = 42,
    test_exit: int = 0,
    test_passed: int = 3,
    test_failed: int = 0,
    numeric_expected: int = 42,
    provenance_hash: str = "a" * 64,
    facts: tuple[StructuredFact, ...] | None = None,
    candidates: tuple[CandidateFact, ...] | None = None,
    include_correctness: bool = True,
) -> VerificationRequest:
    claim = Claim("claim", "fixture claim")
    if facts is None:
        facts = (
            StructuredFact("claim", "version", "v1"),
            StructuredFact("claim", "version", "v1"),
        )
    if candidates is None:
        candidates = (
            CandidateFact("candidate-a", "claim", "answer", 42),
            CandidateFact("candidate-b", "claim", "answer", 42),
        )
    return VerificationRequest(
        claims=(claim,),
        exact_fixtures=(
            (ExactAnswerFixture("claim", 42, exact_actual),) if include_correctness else ()
        ),
        unit_tests=(
            (
                UnitTestEvidence(
                    "claim",
                    "python -m pytest fixture",
                    exit_code=test_exit,
                    passed=test_passed,
                    failed=test_failed,
                ),
            )
            if include_correctness
            else ()
        ),
        numeric_checks=(
            (NumericCheck("claim", "6 * 7", numeric_expected),)
            if include_correctness
            else ()
        ),
        provenance=(
            ProvenanceRecord(
                "claim",
                "fixture-source",
                "fixture://postbase357/claim",
                provenance_hash,
            ),
        ),
        structured_facts=facts,
        candidate_facts=candidates,
    )


def _run_case(name: str, request: VerificationRequest) -> dict[str, object]:
    result = _ensemble().verify(request)
    claim = result.claim("claim")
    return {
        "name": name,
        "overall_status": result.status.value,
        "claim_status": claim.status.value,
        "deterministic_correctness_pass": claim.deterministic_correctness_pass,
        "deterministic_failure": claim.deterministic_failure,
        "reason_codes": [code.value for code in claim.reason_codes],
    }


def build_report() -> dict[str, object]:
    cases = [
        _run_case("all_six_verifiers_pass", _request()),
        _run_case("exact_failure_overrides_heuristic", _request(exact_actual=41)),
        _run_case(
            "unit_test_failure_overrides_heuristic",
            _request(test_exit=1, test_passed=2, test_failed=1),
        ),
        _run_case("numeric_failure_overrides_heuristic", _request(numeric_expected=41)),
        _run_case(
            "provenance_failure_overrides_heuristic",
            _request(provenance_hash="A" * 64),
        ),
        _run_case(
            "consistency_failure_overrides_heuristic",
            _request(
                facts=(
                    StructuredFact("claim", "version", "v1"),
                    StructuredFact("claim", "version", "v2"),
                )
            ),
        ),
        _run_case(
            "cross_candidate_contradiction_is_conflict",
            _request(
                candidates=(
                    CandidateFact("candidate-a", "claim", "answer", 42),
                    CandidateFact("candidate-b", "claim", "answer", 43),
                )
            ),
        ),
        _run_case("non_correctness_support_is_inconclusive", _request(include_correctness=False)),
    ]

    expected = {
        "all_six_verifiers_pass": VerificationStatus.PASS.value,
        "exact_failure_overrides_heuristic": VerificationStatus.FAIL.value,
        "unit_test_failure_overrides_heuristic": VerificationStatus.FAIL.value,
        "numeric_failure_overrides_heuristic": VerificationStatus.FAIL.value,
        "provenance_failure_overrides_heuristic": VerificationStatus.FAIL.value,
        "consistency_failure_overrides_heuristic": VerificationStatus.FAIL.value,
        "cross_candidate_contradiction_is_conflict": VerificationStatus.CONFLICT.value,
        "non_correctness_support_is_inconclusive": VerificationStatus.INCONCLUSIVE.value,
    }
    observed = {str(case["name"]): str(case["overall_status"]) for case in cases}
    if observed != expected:
        raise AssertionError(f"convergence status mismatch: {observed!r}")

    by_name = {str(case["name"]): case for case in cases}
    for name in (
        "exact_failure_overrides_heuristic",
        "unit_test_failure_overrides_heuristic",
        "numeric_failure_overrides_heuristic",
        "provenance_failure_overrides_heuristic",
        "consistency_failure_overrides_heuristic",
    ):
        case = by_name[name]
        reasons = set(case["reason_codes"])
        if not case["deterministic_failure"]:
            raise AssertionError(f"{name} did not record deterministic failure")
        if ReasonCode.HARD_DETERMINISTIC_FAILURE.value not in reasons:
            raise AssertionError(f"{name} missing hard-failure reason")
        if ReasonCode.VERIFIER_DISAGREEMENT.value not in reasons:
            raise AssertionError(f"{name} missing disagreement reason")

    conflict_reasons = set(by_name["cross_candidate_contradiction_is_conflict"]["reason_codes"])
    if ReasonCode.CANDIDATE_CONTRADICTION.value not in conflict_reasons:
        raise AssertionError("candidate contradiction reason missing")

    inconclusive_reasons = set(by_name["non_correctness_support_is_inconclusive"]["reason_codes"])
    if ReasonCode.NO_DETERMINISTIC_CORRECTNESS_SUPPORT.value not in inconclusive_reasons:
        raise AssertionError("inconclusive case missing no-correctness-support reason")

    return {
        "schema": "12-6.postbase357.verifier-ensemble-convergence.v1",
        "worker_id": "POSTBASE-357-VERIFIER-ENSEMBLE-CONVERGENCE",
        "execution_profile": EXECUTION_PROFILE,
        "candidate": {
            "sha": CANDIDATE_SHA,
            "verification_source_blob_sha1": CANDIDATE_SOURCE_BLOB,
            "production_source_modified_by_postbase357": False,
        },
        "verifiers": [
            "exact_answer_fixture",
            "unit_test_code",
            "numeric_calculator",
            "source_provenance_checker",
            "consistency_checker",
            "cross_candidate_contradiction_checker",
        ],
        "external_llm_called": False,
        "network_verifier_called": False,
        "cases": cases,
        "verdict": "PASS_COMPONENT_CONVERGENCE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
