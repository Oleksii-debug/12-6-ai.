from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.postbase import (
    CandidateFact,
    ConsistencyChecker,
    CrossCandidateContradictionChecker,
    ExactAnswerFixture,
    ExactAnswerFixtureVerifier,
    HypothesisVerifierController,
    StructuredFact,
    VerificationRequest,
    VerifierEnsemble,
)
from twelve_six.postbase_hypothesis import HypothesisSearch


def _write(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run() -> dict[str, object]:
    search = HypothesisSearch()
    wrong = search.propose("addition precedes multiplication", initial_score=0.95)
    correct = search.propose("multiplication precedes addition", initial_score=0.40)
    initial_preferred = search.best()

    exact = HypothesisVerifierController(
        search,
        VerifierEnsemble((ExactAnswerFixtureVerifier(),)),
    )
    wrong_claim = exact.claim_for(wrong.id)
    wrong_result = exact.verify(
        wrong.id,
        VerificationRequest(
            claims=(wrong_claim,),
            exact_fixtures=(ExactAnswerFixture(wrong_claim.claim_id, 14, 20),),
        ),
    )
    correct_claim = exact.claim_for(correct.id)
    correct_result = exact.verify(
        correct.id,
        VerificationRequest(
            claims=(correct_claim,),
            exact_fixtures=(ExactAnswerFixture(correct_claim.claim_id, 14, 14),),
        ),
    )

    conflict = search.propose("candidate observations agree", initial_score=0.20)
    conflict_controller = HypothesisVerifierController(
        search,
        VerifierEnsemble((CrossCandidateContradictionChecker(),)),
    )
    conflict_claim = conflict_controller.claim_for(conflict.id)
    conflict_result = conflict_controller.verify(
        conflict.id,
        VerificationRequest(
            claims=(conflict_claim,),
            candidate_facts=(
                CandidateFact("a", conflict_claim.claim_id, "value", 14),
                CandidateFact("b", conflict_claim.claim_id, "value", 20),
            ),
        ),
    )

    contradictory = search.propose("deterministic evidence agrees", initial_score=0.99)
    contradiction_controller = HypothesisVerifierController(
        search,
        VerifierEnsemble((ExactAnswerFixtureVerifier(), ConsistencyChecker())),
    )
    contradiction_claim = contradiction_controller.claim_for(contradictory.id)
    contradiction_result = contradiction_controller.verify(
        contradictory.id,
        VerificationRequest(
            claims=(contradiction_claim,),
            exact_fixtures=(ExactAnswerFixture(contradiction_claim.claim_id, 14, 14),),
            structured_facts=(
                StructuredFact(contradiction_claim.claim_id, "value", 14),
                StructuredFact(contradiction_claim.claim_id, "value", 20),
            ),
        ),
    )

    revised = search.revise(
        wrong.id,
        "multiplication precedes addition for ordinary arithmetic",
        initial_score=0.30,
    )
    final_preferred = search.best()
    proof_passed = all(
        (
            initial_preferred is not None and initial_preferred.id == wrong.id,
            wrong_result.status.value == "FAIL",
            wrong_result.rejected,
            search.hypothesis(wrong.id).status == "rejected",
            correct_result.status.value == "PASS",
            conflict_result.status.value == "CONFLICT",
            not conflict_result.rejected,
            contradiction_result.status.value == "FAIL",
            contradiction_result.rejected,
            revised.version_id != wrong.version_id,
            revised.verifier_evidence_ids == (),
            final_preferred is not None and final_preferred.id == correct.id,
        )
    )
    report: dict[str, object] = {
        "schema": "12-6.hypothesis-verifier-integration.probe.v1",
        "worker_id": "NEXT100-082-HYPOTHESIS-VERIFIER-INTEGRATION",
        "execution_profile": "LOCAL_FREE",
        "external_model_judge_called": False,
        "network_verifier_called": False,
        "initial_preferred_hypothesis_id": initial_preferred.id if initial_preferred else None,
        "wrong_version_id": wrong.version_id,
        "wrong_status": wrong_result.status.value,
        "wrong_rejected": wrong_result.rejected,
        "correct_status": correct_result.status.value,
        "conflict_status": conflict_result.status.value,
        "contradictory_status": contradiction_result.status.value,
        "contradictory_rejected": contradiction_result.rejected,
        "revised_version_id": revised.version_id,
        "revised_inherited_verifier_evidence": bool(revised.verifier_evidence_ids),
        "final_preferred_hypothesis_id": final_preferred.id if final_preferred else None,
        "proof_passed": proof_passed,
    }
    if not proof_passed:
        raise RuntimeError("NEXT100-082 deterministic probe failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write(run(), args.output)


if __name__ == "__main__":
    main()
