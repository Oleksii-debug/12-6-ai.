"""Deterministic verifier harness for isolated post-training experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Candidate, VerifierTask
from .verifiers import VerificationResult, VerifierRegistry


@dataclass(frozen=True, slots=True)
class VerificationCase:
    task: VerifierTask
    candidate: Candidate
    verifier_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.verifier_names:
            raise ValueError("verification case requires at least one verifier")
        if any(not name.strip() for name in self.verifier_names):
            raise ValueError("verifier names must be non-empty")


@dataclass(frozen=True, slots=True)
class CaseVerification:
    task_id: str
    candidate_id: str
    results: tuple[VerificationResult, ...]

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)


@dataclass(frozen=True, slots=True)
class VerifierHarnessReport:
    cases: tuple[CaseVerification, ...]
    total_verifications: int
    passed_verifications: int
    mean_score: float

    @property
    def all_cases_passed(self) -> bool:
        return bool(self.cases) and all(case.all_passed for case in self.cases)


def run_verifier_harness(
    registry: VerifierRegistry,
    cases: tuple[VerificationCase, ...],
) -> VerifierHarnessReport:
    """Run selected deterministic verifiers without training or reward updates.

    Unknown verifier names fail closed through ``VerifierRegistry.get``. The harness
    preserves case/verifier order and computes descriptive metrics only; it does not
    transform scores into gradients, preference labels, or stage-promotion claims.
    """

    completed: list[CaseVerification] = []
    total = 0
    passed = 0
    score_sum = 0.0

    for case in cases:
        results: list[VerificationResult] = []
        for verifier_name in case.verifier_names:
            verifier = registry.get(verifier_name)
            result = verifier.verify(case.task, case.candidate)
            results.append(result)
            total += 1
            passed += int(result.passed)
            score_sum += result.score
        completed.append(
            CaseVerification(
                task_id=case.task.task_id,
                candidate_id=case.candidate.candidate_id,
                results=tuple(results),
            )
        )

    mean_score = score_sum / total if total else 0.0
    return VerifierHarnessReport(
        cases=tuple(completed),
        total_verifications=total,
        passed_verifications=passed,
        mean_score=mean_score,
    )
