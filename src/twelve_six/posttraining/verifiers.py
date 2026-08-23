"""Verifier API and deterministic registry for reasoning/post-training experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VerificationContext:
    task_id: str
    candidate_id: str
    prompt: str
    candidate: str
    reference_answer: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.candidate_id.strip():
            raise ValueError("task_id and candidate_id must be non-empty")
        if not self.prompt or not self.candidate:
            raise ValueError("prompt and candidate must be non-empty")


@dataclass(frozen=True, slots=True)
class VerifierResult:
    verifier_id: str
    verifier_version: str
    passed: bool
    score: float
    reason: str
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.verifier_id.strip() or not self.verifier_version.strip():
            raise ValueError("verifier identity must be non-empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        if not self.reason:
            raise ValueError("reason must be non-empty")


class Verifier(Protocol):
    verifier_id: str
    verifier_version: str

    def verify(self, context: VerificationContext) -> VerifierResult:
        ...


class VerifierRegistry:
    """Fail-closed registry keyed by stable verifier id and version."""

    def __init__(self) -> None:
        self._verifiers: dict[tuple[str, str], Verifier] = {}

    def register(self, verifier: Verifier) -> None:
        key = (verifier.verifier_id, verifier.verifier_version)
        if not all(part.strip() for part in key):
            raise ValueError("verifier identity must be non-empty")
        if key in self._verifiers:
            raise ValueError(f"duplicate verifier registration: {key[0]}@{key[1]}")
        self._verifiers[key] = verifier

    def get(self, verifier_id: str, verifier_version: str) -> Verifier:
        key = (verifier_id, verifier_version)
        try:
            return self._verifiers[key]
        except KeyError as exc:
            raise KeyError(f"unknown verifier: {verifier_id}@{verifier_version}") from exc

    def identities(self) -> tuple[str, ...]:
        return tuple(f"{key[0]}@{key[1]}" for key in sorted(self._verifiers))


class ExactMatchVerifier:
    """Reference verifier used only for harness/contract smoke tests."""

    verifier_id = "exact_match"
    verifier_version = "1"

    def __init__(self, *, strip: bool = True, casefold: bool = False) -> None:
        self._strip = strip
        self._casefold = casefold

    def _normalize(self, value: str) -> str:
        if self._strip:
            value = value.strip()
        if self._casefold:
            value = value.casefold()
        return value

    def verify(self, context: VerificationContext) -> VerifierResult:
        if context.reference_answer is None:
            return VerifierResult(
                verifier_id=self.verifier_id,
                verifier_version=self.verifier_version,
                passed=False,
                score=0.0,
                reason="reference_answer_missing",
            )
        passed = self._normalize(context.candidate) == self._normalize(context.reference_answer)
        return VerifierResult(
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="exact_match" if passed else "exact_mismatch",
        )
