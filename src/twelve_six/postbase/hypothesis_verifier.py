from __future__ import annotations

from dataclasses import dataclass

from ..postbase_hypothesis import HypothesisSearch, VerifierEvidence
from .verification import (
    Claim,
    EnsembleResult,
    VerificationDimension,
    VerificationRequest,
    VerificationStatus,
    VerifierEnsemble,
)


@dataclass(frozen=True, slots=True)
class HypothesisVerificationResult:
    hypothesis_id: str
    hypothesis_version_id: str
    status: VerificationStatus
    evidence: VerifierEvidence
    ensemble_result: EnsembleResult
    rejected: bool


class HypothesisVerifierController:
    """Bind deterministic verifier-ensemble evidence to exact hypothesis versions."""

    worker_id = "NEXT100-082-HYPOTHESIS-VERIFIER-INTEGRATION"
    schema = "12-6.hypothesis-verifier-integration.v1"

    def __init__(self, search: HypothesisSearch, ensemble: VerifierEnsemble) -> None:
        disallowed = tuple(
            verifier.verifier_id
            for verifier in ensemble.verifiers
            if not verifier.deterministic
            or verifier.dimension is VerificationDimension.MODEL_JUDGMENT
        )
        if disallowed:
            joined = ", ".join(disallowed)
            raise ValueError(
                "hypothesis verification integration accepts deterministic local verifiers only: "
                f"{joined}"
            )
        self.search = search
        self.ensemble = ensemble

    def claim_for(self, hypothesis_id: str) -> Claim:
        hypothesis = self.search.hypothesis(hypothesis_id)
        return Claim(hypothesis.version_id, hypothesis.statement)

    def verify(
        self,
        hypothesis_id: str,
        request: VerificationRequest,
    ) -> HypothesisVerificationResult:
        hypothesis = self.search.hypothesis(hypothesis_id)
        if hypothesis.status != "active":
            raise ValueError(f"hypothesis {hypothesis_id} is not active")
        claim_ids = tuple(claim.claim_id for claim in request.claims)
        if claim_ids != (hypothesis.version_id,):
            raise ValueError(
                "verification request must contain exactly the current hypothesis version claim id"
            )

        result = self.ensemble.verify(request)
        claim_result = result.claim(hypothesis.version_id)
        evidence = self.search.record_verifier_evidence(
            hypothesis.id,
            claim_id=hypothesis.version_id,
            status=claim_result.status.value,
            reason_codes=tuple(code.value for code in claim_result.reason_codes),
            deterministic_failure=claim_result.deterministic_failure,
            deterministic_correctness_pass=claim_result.deterministic_correctness_pass,
            verifier_ids=tuple(verifier.verifier_id for verifier in self.ensemble.verifiers),
        )

        rejected = False
        if claim_result.deterministic_failure:
            self.search.reject(
                hypothesis.id,
                "hard deterministic verifier failure",
                verifier_evidence_ids=(evidence.id,),
            )
            rejected = True

        return HypothesisVerificationResult(
            hypothesis_id=hypothesis.id,
            hypothesis_version_id=hypothesis.version_id,
            status=claim_result.status,
            evidence=evidence,
            ensemble_result=result,
            rejected=rejected,
        )

    def preferred(self):
        return self.search.best()
