"""Framework-neutral verifier interfaces and deterministic baseline verifiers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

from .contracts import Candidate, VerifierTask


@dataclass(frozen=True, slots=True)
class VerificationResult:
    verifier: str
    passed: bool
    score: float
    reason: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.verifier.strip():
            raise ValueError("verifier must be non-empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")


@runtime_checkable
class Verifier(Protocol):
    """Small adapter boundary for math/code/logic or learned verifiers."""

    name: str

    def verify(self, task: VerifierTask, candidate: Candidate) -> VerificationResult:
        ...


@dataclass(frozen=True, slots=True)
class ExactTextVerifier:
    name: str = "exact_text"
    strip: bool = True
    case_sensitive: bool = True

    def verify(self, task: VerifierTask, candidate: Candidate) -> VerificationResult:
        if task.reference is None:
            return VerificationResult(
                verifier=self.name,
                passed=False,
                score=0.0,
                reason="task has no reference",
            )
        actual = candidate.text.strip() if self.strip else candidate.text
        expected = task.reference.strip() if self.strip else task.reference
        if not self.case_sensitive:
            actual = actual.casefold()
            expected = expected.casefold()
        passed = actual == expected
        return VerificationResult(
            verifier=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="exact match" if passed else "text mismatch",
        )


@dataclass(frozen=True, slots=True)
class NumericToleranceVerifier:
    """Verify a decimal-valued answer without using eval or executing code."""

    tolerance: Decimal = Decimal(0)
    name: str = "numeric_tolerance"

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    def verify(self, task: VerifierTask, candidate: Candidate) -> VerificationResult:
        if task.reference is None:
            return VerificationResult(self.name, False, 0.0, "task has no reference")
        try:
            actual = Decimal(candidate.text.strip())
            expected = Decimal(task.reference.strip())
        except InvalidOperation:
            return VerificationResult(self.name, False, 0.0, "non-decimal answer")
        error = abs(actual - expected)
        passed = error <= self.tolerance
        return VerificationResult(
            verifier=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="within tolerance" if passed else "outside tolerance",
            metadata={"absolute_error": str(error), "tolerance": str(self.tolerance)},
        )


class VerifierRegistry:
    """Explicit registry; duplicate names fail closed."""

    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, verifier: Verifier) -> None:
        if verifier.name in self._verifiers:
            raise ValueError(f"verifier already registered: {verifier.name}")
        self._verifiers[verifier.name] = verifier

    def get(self, name: str) -> Verifier:
        try:
            return self._verifiers[name]
        except KeyError as exc:
            raise KeyError(f"unknown verifier: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._verifiers))
