from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class VerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONFLICT = "CONFLICT"


class VerificationDimension(StrEnum):
    CORRECTNESS = "CORRECTNESS"
    CONSISTENCY = "CONSISTENCY"
    PROVENANCE = "PROVENANCE"
    CROSS_CANDIDATE = "CROSS_CANDIDATE"
    MODEL_JUDGMENT = "MODEL_JUDGMENT"


class ReasonCode(StrEnum):
    EXACT_MATCH = "EXACT_MATCH"
    EXACT_MISMATCH = "EXACT_MISMATCH"
    UNIT_TESTS_PASS = "UNIT_TESTS_PASS"
    UNIT_TESTS_FAIL = "UNIT_TESTS_FAIL"
    UNIT_TEST_EVIDENCE_INVALID = "UNIT_TEST_EVIDENCE_INVALID"
    NUMERIC_MATCH = "NUMERIC_MATCH"
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
    NUMERIC_EXPRESSION_REJECTED = "NUMERIC_EXPRESSION_REJECTED"
    CONSISTENT_FACTS = "CONSISTENT_FACTS"
    INTERNAL_CONTRADICTION = "INTERNAL_CONTRADICTION"
    PROVENANCE_COMPLETE = "PROVENANCE_COMPLETE"
    PROVENANCE_MISSING = "PROVENANCE_MISSING"
    PROVENANCE_HASH_INVALID = "PROVENANCE_HASH_INVALID"
    CANDIDATES_AGREE = "CANDIDATES_AGREE"
    CANDIDATE_CONTRADICTION = "CANDIDATE_CONTRADICTION"
    NO_APPLICABLE_EVIDENCE = "NO_APPLICABLE_EVIDENCE"
    HARD_DETERMINISTIC_FAILURE = "HARD_DETERMINISTIC_FAILURE"
    VERIFIER_DISAGREEMENT = "VERIFIER_DISAGREEMENT"
    NO_DETERMINISTIC_CORRECTNESS_SUPPORT = "NO_DETERMINISTIC_CORRECTNESS_SUPPORT"
    HEURISTIC_ONLY_SUPPORT = "HEURISTIC_ONLY_SUPPORT"


class ClaimDisposition(StrEnum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    CONFLICTED = "CONFLICTED"
    PROPOSED = "PROPOSED"


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.text.strip():
            raise ValueError("claim id and text must be non-empty")


@dataclass(frozen=True, slots=True)
class ExactAnswerFixture:
    claim_id: str
    expected: Any
    actual: Any


@dataclass(frozen=True, slots=True)
class UnitTestEvidence:
    claim_id: str
    command: str
    exit_code: int
    passed: int
    failed: int = 0
    errors: int = 0


@dataclass(frozen=True, slots=True)
class NumericCheck:
    claim_id: str
    expression: str
    expected: int | float | str | Decimal
    abs_tolerance: int | float | str | Decimal = 0
    rel_tolerance: int | float | str | Decimal = 0


@dataclass(frozen=True, slots=True)
class StructuredFact:
    claim_id: str
    key: str
    value: Any


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    claim_id: str
    source_id: str
    locator: str
    content_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateFact:
    candidate_id: str
    claim_id: str
    key: str
    value: Any


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    claims: tuple[Claim, ...]
    exact_fixtures: tuple[ExactAnswerFixture, ...] = ()
    unit_tests: tuple[UnitTestEvidence, ...] = ()
    numeric_checks: tuple[NumericCheck, ...] = ()
    structured_facts: tuple[StructuredFact, ...] = ()
    provenance: tuple[ProvenanceRecord, ...] = ()
    candidate_facts: tuple[CandidateFact, ...] = ()

    def __post_init__(self) -> None:
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim ids must be unique")


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    claim_id: str
    status: VerificationStatus
    reason_codes: tuple[ReasonCode, ...]


@dataclass(frozen=True, slots=True)
class VerifierResult:
    verifier_id: str
    deterministic: bool
    dimension: VerificationDimension
    verdicts: tuple[ClaimVerdict, ...]


@runtime_checkable
class Verifier(Protocol):
    verifier_id: str
    deterministic: bool
    dimension: VerificationDimension

    def verify(self, request: VerificationRequest) -> VerifierResult: ...


@runtime_checkable
class ModelJudge(Protocol):
    """Future heuristic model-judge seam; this module provides no implementation."""

    judge_id: str

    def judge(self, request: VerificationRequest) -> VerifierResult: ...


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return len(left) == len(right) and all(
            key in right and _exact_equal(value, right[key]) for key, value in left.items()
        )
    return bool(left == right)


def _group(items: tuple[Any, ...]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for item in items:
        grouped.setdefault(item.claim_id, []).append(item)
    return grouped


def _none(claim_id: str) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id,
        VerificationStatus.INCONCLUSIVE,
        (ReasonCode.NO_APPLICABLE_EVIDENCE,),
    )


class ExactAnswerFixtureVerifier:
    verifier_id = "exact_answer_fixture"
    deterministic = True
    dimension = VerificationDimension.CORRECTNESS

    def verify(self, request: VerificationRequest) -> VerifierResult:
        grouped = _group(request.exact_fixtures)
        verdicts = []
        for claim in request.claims:
            fixtures = grouped.get(claim.claim_id, [])
            if not fixtures:
                verdicts.append(_none(claim.claim_id))
            elif all(_exact_equal(x.expected, x.actual) for x in fixtures):
                verdicts.append(
                    ClaimVerdict(
                        claim.claim_id,
                        VerificationStatus.PASS,
                        (ReasonCode.EXACT_MATCH,),
                    )
                )
            else:
                verdicts.append(
                    ClaimVerdict(
                        claim.claim_id,
                        VerificationStatus.FAIL,
                        (ReasonCode.EXACT_MISMATCH,),
                    )
                )
        return VerifierResult(self.verifier_id, True, self.dimension, tuple(verdicts))


class UnitTestCodeVerifier:
    verifier_id = "unit_test_code"
    deterministic = True
    dimension = VerificationDimension.CORRECTNESS

    def verify(self, request: VerificationRequest) -> VerifierResult:
        grouped = _group(request.unit_tests)
        verdicts = []
        for claim in request.claims:
            runs = grouped.get(claim.claim_id, [])
            if not runs:
                verdicts.append(_none(claim.claim_id))
                continue
            invalid = any(
                not x.command.strip()
                or min(x.exit_code, x.passed, x.failed, x.errors) < 0
                or x.passed + x.failed + x.errors == 0
                or (x.exit_code == 0 and (x.failed or x.errors))
                for x in runs
            )
            failing = any(x.exit_code != 0 or x.failed or x.errors for x in runs)
            if invalid:
                code = ReasonCode.UNIT_TEST_EVIDENCE_INVALID
                status = VerificationStatus.FAIL
            elif failing:
                code = ReasonCode.UNIT_TESTS_FAIL
                status = VerificationStatus.FAIL
            else:
                code = ReasonCode.UNIT_TESTS_PASS
                status = VerificationStatus.PASS
            verdicts.append(ClaimVerdict(claim.claim_id, status, (code,)))
        return VerifierResult(self.verifier_id, True, self.dimension, tuple(verdicts))


def _decimal(value: int | float | str | Decimal) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric verification value")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid decimal value") from exc
    if not result.is_finite():
        raise ValueError("numeric values must be finite")
    return result


def _numeric(node: ast.AST, depth: int = 0) -> Decimal:
    if depth > 32:
        raise ValueError("numeric expression too deep")
    if isinstance(node, ast.Expression):
        return _numeric(node.body, depth + 1)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _decimal(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _numeric(node.operand, depth + 1)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp):
        raise ValueError("unsupported numeric expression")
    left, right = _numeric(node.left, depth + 1), _numeric(node.right, depth + 1)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.FloorDiv):
        return left // right
    if isinstance(node.op, ast.Mod):
        return left % right
    if isinstance(node.op, ast.Pow):
        if right != right.to_integral_value() or abs(right) > 1000:
            raise ValueError("unsupported exponent")
        return left ** int(right)
    raise ValueError("unsupported numeric operator")


def safe_calculate(expression: str) -> Decimal:
    if not 0 < len(expression.strip()) <= 512:
        raise ValueError("numeric expression must be 1..512 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid numeric expression") from exc
    with localcontext() as context:
        context.prec = 50
        return _numeric(tree)


class NumericCalculatorVerifier:
    verifier_id = "numeric_calculator"
    deterministic = True
    dimension = VerificationDimension.CORRECTNESS

    def verify(self, request: VerificationRequest) -> VerifierResult:
        grouped = _group(request.numeric_checks)
        verdicts = []
        for claim in request.claims:
            checks = grouped.get(claim.claim_id, [])
            if not checks:
                verdicts.append(_none(claim.claim_id))
                continue
            mismatch = False
            rejected = False
            for check in checks:
                try:
                    actual = safe_calculate(check.expression)
                    expected = _decimal(check.expected)
                    absolute = _decimal(check.abs_tolerance)
                    relative = _decimal(check.rel_tolerance)
                    if absolute < 0 or relative < 0:
                        raise ValueError("negative tolerance")
                    mismatch |= abs(actual - expected) > max(absolute, relative * abs(expected))
                except (ValueError, ArithmeticError, InvalidOperation):
                    rejected = True
            if rejected:
                code, status = ReasonCode.NUMERIC_EXPRESSION_REJECTED, VerificationStatus.FAIL
            elif mismatch:
                code, status = ReasonCode.NUMERIC_MISMATCH, VerificationStatus.FAIL
            else:
                code, status = ReasonCode.NUMERIC_MATCH, VerificationStatus.PASS
            verdicts.append(ClaimVerdict(claim.claim_id, status, (code,)))
        return VerifierResult(self.verifier_id, True, self.dimension, tuple(verdicts))


class ConsistencyChecker:
    verifier_id = "consistency_checker"
    deterministic = True
    dimension = VerificationDimension.CONSISTENCY

    def verify(self, request: VerificationRequest) -> VerifierResult:
        grouped = _group(request.structured_facts)
        verdicts = []
        for claim in request.claims:
            facts = grouped.get(claim.claim_id, [])
            if not facts:
                verdicts.append(_none(claim.claim_id))
                continue
            by_key: dict[str, list[Any]] = {}
            for fact in facts:
                by_key.setdefault(fact.key, []).append(fact.value)
            conflict = any(
                any(not _exact_equal(values[0], value) for value in values[1:])
                for values in by_key.values()
            )
            code = ReasonCode.INTERNAL_CONTRADICTION if conflict else ReasonCode.CONSISTENT_FACTS
            status = VerificationStatus.FAIL if conflict else VerificationStatus.PASS
            verdicts.append(ClaimVerdict(claim.claim_id, status, (code,)))
        return VerifierResult(self.verifier_id, True, self.dimension, tuple(verdicts))


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceProvenanceChecker:
    verifier_id = "source_provenance_checker"
    deterministic = True
    dimension = VerificationDimension.PROVENANCE

    def __init__(self, *, require_content_hash: bool = False) -> None:
        self.require_content_hash = require_content_hash

    def verify(self, request: VerificationRequest) -> VerifierResult:
        grouped = _group(request.provenance)
        verdicts = []
        for claim in request.claims:
            records = grouped.get(claim.claim_id, [])
            missing = not records or any(
                not x.source_id.strip()
                or not x.locator.strip()
                or (self.require_content_hash and x.content_sha256 is None)
                for x in records
            )
            bad_hash = any(
                x.content_sha256 is not None and not _SHA256.fullmatch(x.content_sha256)
                for x in records
            )
            if missing:
                code, status = ReasonCode.PROVENANCE_MISSING, VerificationStatus.FAIL
            elif bad_hash:
                code, status = ReasonCode.PROVENANCE_HASH_INVALID, VerificationStatus.FAIL
            else:
                code, status = ReasonCode.PROVENANCE_COMPLETE, VerificationStatus.PASS
            verdicts.append(ClaimVerdict(claim.claim_id, status, (code,)))
        return VerifierResult(self.verifier_id, True, self.dimension, tuple(verdicts))


class CrossCandidateContradictionChecker:
    verifier_id = "cross_candidate_contradiction_checker"
    deterministic = True
    dimension = VerificationDimension.CROSS_CANDIDATE

    def verify(self, request: VerificationRequest) -> VerifierResult:
        grouped = _group(request.candidate_facts)
        verdicts = []
        for claim in request.claims:
            facts = grouped.get(claim.claim_id, [])
            if not facts:
                verdicts.append(_none(claim.claim_id))
                continue
            by_key: dict[str, list[Any]] = {}
            for fact in facts:
                by_key.setdefault(fact.key, []).append(fact.value)
            conflict = any(
                any(not _exact_equal(values[0], value) for value in values[1:])
                for values in by_key.values()
            )
            code = ReasonCode.CANDIDATE_CONTRADICTION if conflict else ReasonCode.CANDIDATES_AGREE
            status = VerificationStatus.CONFLICT if conflict else VerificationStatus.PASS
            verdicts.append(ClaimVerdict(claim.claim_id, status, (code,)))
        return VerifierResult(self.verifier_id, True, self.dimension, tuple(verdicts))


@dataclass(frozen=True, slots=True)
class AggregatedClaimResult:
    claim_id: str
    status: VerificationStatus
    reason_codes: tuple[ReasonCode, ...]
    deterministic_correctness_pass: bool
    deterministic_failure: bool


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    status: VerificationStatus
    claims: tuple[AggregatedClaimResult, ...]

    def claim(self, claim_id: str) -> AggregatedClaimResult:
        return next(item for item in self.claims if item.claim_id == claim_id)


class VerifierEnsemble:
    def __init__(self, verifiers: tuple[Verifier, ...]) -> None:
        if not verifiers or len({x.verifier_id for x in verifiers}) != len(verifiers):
            raise ValueError("ensemble needs non-empty unique verifier ids")
        self.verifiers = verifiers

    def verify(self, request: VerificationRequest) -> EnsembleResult:
        results = tuple(verifier.verify(request) for verifier in self.verifiers)
        for verifier, result in zip(self.verifiers, results, strict=True):
            if (
                result.verifier_id != verifier.verifier_id
                or result.deterministic != verifier.deterministic
                or result.dimension is not verifier.dimension
            ):
                raise ValueError("verifier result metadata mismatch")
        aggregates = []
        for claim in request.claims:
            evidence = [
                (result, verdict)
                for result in results
                for verdict in result.verdicts
                if verdict.claim_id == claim.claim_id
            ]
            deterministic_fail = any(
                result.deterministic and verdict.status is VerificationStatus.FAIL
                for result, verdict in evidence
            )
            correctness_pass = any(
                result.deterministic
                and result.dimension is VerificationDimension.CORRECTNESS
                and verdict.status is VerificationStatus.PASS
                for result, verdict in evidence
            )
            passed = any(verdict.status is VerificationStatus.PASS for _, verdict in evidence)
            failed = any(verdict.status is VerificationStatus.FAIL for _, verdict in evidence)
            conflict = any(verdict.status is VerificationStatus.CONFLICT for _, verdict in evidence)
            heuristic_pass = any(
                not result.deterministic and verdict.status is VerificationStatus.PASS
                for result, verdict in evidence
            )
            reasons = list(dict.fromkeys(code for _, v in evidence for code in v.reason_codes))
            if deterministic_fail:
                status = VerificationStatus.FAIL
                reasons.append(ReasonCode.HARD_DETERMINISTIC_FAILURE)
                if passed or conflict:
                    reasons.append(ReasonCode.VERIFIER_DISAGREEMENT)
            elif conflict or (passed and failed):
                status = VerificationStatus.CONFLICT
                reasons.append(ReasonCode.VERIFIER_DISAGREEMENT)
            elif correctness_pass:
                status = VerificationStatus.PASS
            else:
                status = VerificationStatus.INCONCLUSIVE
                reasons.append(ReasonCode.NO_DETERMINISTIC_CORRECTNESS_SUPPORT)
                if heuristic_pass:
                    reasons.append(ReasonCode.HEURISTIC_ONLY_SUPPORT)
            aggregates.append(
                AggregatedClaimResult(
                    claim.claim_id,
                    status,
                    tuple(dict.fromkeys(reasons)),
                    correctness_pass,
                    deterministic_fail,
                )
            )
        statuses = [x.status for x in aggregates]
        if VerificationStatus.FAIL in statuses:
            overall = VerificationStatus.FAIL
        elif VerificationStatus.CONFLICT in statuses:
            overall = VerificationStatus.CONFLICT
        elif statuses and all(x is VerificationStatus.PASS for x in statuses):
            overall = VerificationStatus.PASS
        else:
            overall = VerificationStatus.INCONCLUSIVE
        return EnsembleResult(overall, tuple(aggregates))


@dataclass(frozen=True, slots=True)
class ControlledClaim:
    claim: Claim
    disposition: ClaimDisposition
    verification_status: VerificationStatus
    reason_codes: tuple[ReasonCode, ...]


@dataclass(frozen=True, slots=True)
class FinalAnswerPlan:
    claims: tuple[ControlledClaim, ...]

    @property
    def verified_claim_ids(self) -> tuple[str, ...]:
        return tuple(
            item.claim.claim_id
            for item in self.claims
            if item.disposition is ClaimDisposition.VERIFIED
        )

    @property
    def proposed_claim_ids(self) -> tuple[str, ...]:
        return tuple(
            item.claim.claim_id
            for item in self.claims
            if item.disposition is ClaimDisposition.PROPOSED
        )


class FinalAnswerController:
    def build_plan(self, request: VerificationRequest, result: EnsembleResult) -> FinalAnswerPlan:
        controlled = []
        for claim in request.claims:
            item = result.claim(claim.claim_id)
            if item.status is VerificationStatus.PASS and item.deterministic_correctness_pass:
                disposition = ClaimDisposition.VERIFIED
            elif item.status is VerificationStatus.FAIL:
                disposition = ClaimDisposition.REJECTED
            elif item.status is VerificationStatus.CONFLICT:
                disposition = ClaimDisposition.CONFLICTED
            else:
                disposition = ClaimDisposition.PROPOSED
            controlled.append(ControlledClaim(claim, disposition, item.status, item.reason_codes))
        return FinalAnswerPlan(tuple(controlled))
