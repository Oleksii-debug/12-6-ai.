"""Deterministic candidate-response harness for immutable EVAL-354 fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .communication_eval import (
    CLAIM_SCOPE,
    CommunicationCase,
    CommunicationSuite,
    Message,
    evaluate_suite,
    load_responses,
    load_suite,
)

RESULT_SCHEMA = "12-6.post-base.communication-eval-harness-result.v1"
DETERMINISTIC_MOCK = "deterministic_mock"
LEARNED_BASE_ADAPTER_PLUMBING = "learned_base_adapter_plumbing"
POST_SFT_MODEL = "post_sft_model"
ALLOWED_CANDIDATE_KINDS = frozenset(
    {DETERMINISTIC_MOCK, LEARNED_BASE_ADAPTER_PLUMBING, POST_SFT_MODEL}
)
EVALUATION_USE = "evaluation_only"
MOCK_EVIDENCE_SCOPE = "mock_harness_mechanics"
BASE_PLUMBING_SCOPE = "base_adapter_plumbing_only"
POST_SFT_FIXTURE_SCOPE = "post_sft_fixture_behavior_only"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


@dataclass(frozen=True, slots=True)
class CandidateRequest:
    """Candidate-visible request with scorer answers deliberately removed."""

    case_id: str
    messages: tuple[Message, ...]
    tool_state: tuple[tuple[str, str], ...]

    @classmethod
    def from_case(cls, case: CommunicationCase) -> "CandidateRequest":
        return cls(case.case_id, case.messages, case.tool_state)


@dataclass(frozen=True, slots=True)
class CandidateDescriptor:
    candidate_id: str
    candidate_kind: str
    generator_identity_sha256: str
    deterministic_generation: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if self.candidate_kind not in ALLOWED_CANDIDATE_KINDS:
            raise ValueError(f"unsupported candidate_kind: {self.candidate_kind!r}")
        if not _valid_sha256(self.generator_identity_sha256):
            raise ValueError("generator_identity_sha256 must be lowercase SHA-256 hex")
        if self.deterministic_generation is not True:
            raise ValueError("EVAL-354 candidate generation must be deterministic")


@runtime_checkable
class CandidateResponder(Protocol):
    descriptor: CandidateDescriptor
    case_id_allowlist: frozenset[str] | None

    def generate(self, request: CandidateRequest) -> str:
        ...


class DeterministicMappingResponder:
    """Project-owned deterministic responder for harness-mechanics fixtures."""

    def __init__(self, *, candidate_id: str, responses: Mapping[str, str]) -> None:
        copied = dict(responses)
        if any(not isinstance(key, str) or not key for key in copied):
            raise ValueError("mock response case IDs must be non-empty strings")
        if any(not isinstance(value, str) for value in copied.values()):
            raise TypeError("all mock responses must be strings")
        material = [
            {"case_id": case_id, "response": copied[case_id]} for case_id in sorted(copied)
        ]
        self._responses = copied
        self.case_id_allowlist = frozenset(copied)
        self.descriptor = CandidateDescriptor(
            candidate_id=candidate_id,
            candidate_kind=DETERMINISTIC_MOCK,
            generator_identity_sha256=_sha256(_canonical_json(material)),
        )

    def generate(self, request: CandidateRequest) -> str:
        return self._responses[request.case_id]


class CallableResponder:
    """Thin local adapter for learned-Base plumbing or future post-SFT candidates."""

    def __init__(
        self,
        *,
        descriptor: CandidateDescriptor,
        generate: Callable[[CandidateRequest], str],
    ) -> None:
        self.descriptor = descriptor
        self.case_id_allowlist: frozenset[str] | None = None
        self._generate = generate

    def generate(self, request: CandidateRequest) -> str:
        return self._generate(request)


@dataclass(frozen=True, slots=True)
class HarnessCaseResult:
    case_id: str
    category: str
    response_sha256: str
    response_utf8_bytes: int
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "response_sha256": self.response_sha256,
            "response_utf8_bytes": self.response_utf8_bytes,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HarnessResult:
    suite_id: str
    suite_identity_sha256: str
    candidate: CandidateDescriptor
    candidate_evidence_scope: str
    response_set_sha256: str
    total: int
    passed_count: int
    failed_count: int
    passed: bool
    category_results: tuple[tuple[str, int, int], ...]
    case_results: tuple[HarnessCaseResult, ...]
    result_identity_sha256: str

    def payload_without_identity(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "suite_id": self.suite_id,
            "suite_identity_sha256": self.suite_identity_sha256,
            "candidate": {
                "candidate_id": self.candidate.candidate_id,
                "candidate_kind": self.candidate.candidate_kind,
                "generator_identity_sha256": self.candidate.generator_identity_sha256,
                "deterministic_generation": True,
            },
            "candidate_evidence_scope": self.candidate_evidence_scope,
            "evaluation_use": EVALUATION_USE,
            "suite_training_eligible": False,
            "candidate_outputs_training_eligible": False,
            "final_test_training_reuse_authorized": False,
            "raw_responses_embedded": False,
            "reference_responses_exposed_to_candidate": False,
            "claim_scope": CLAIM_SCOPE,
            "broad_quality_claim_authorized": False,
            "base_raw_lm_diagnostic": False,
            "response_set_sha256": self.response_set_sha256,
            "repeated_generation_verified": True,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "passed": self.passed,
            "category_results": [
                {"category": category, "passed_count": passed, "failed_count": failed}
                for category, passed, failed in self.category_results
            ],
            "case_results": [result.as_dict() for result in self.case_results],
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload_without_identity()
        payload["result_identity_sha256"] = self.result_identity_sha256
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.as_dict()) + b"\n"


def _evidence_scope(kind: str) -> str:
    if kind == DETERMINISTIC_MOCK:
        return MOCK_EVIDENCE_SCOPE
    if kind == LEARNED_BASE_ADAPTER_PLUMBING:
        return BASE_PLUMBING_SCOPE
    if kind == POST_SFT_MODEL:
        return POST_SFT_FIXTURE_SCOPE
    raise AssertionError(kind)


def run_candidate_harness(
    suite: CommunicationSuite,
    responder: CandidateResponder,
) -> HarnessResult:
    """Generate each case twice, fail on divergence, then score exactly once."""

    if not isinstance(responder, CandidateResponder):
        raise TypeError("responder does not satisfy CandidateResponder protocol")
    expected_ids = frozenset(case.case_id for case in suite.cases)
    if responder.case_id_allowlist is not None and responder.case_id_allowlist != expected_ids:
        missing = sorted(expected_ids - responder.case_id_allowlist)
        extra = sorted(responder.case_id_allowlist - expected_ids)
        raise ValueError(f"responder case IDs mismatch; missing={missing}, extra={extra}")

    responses: dict[str, str] = {}
    for case in suite.cases:
        request = CandidateRequest.from_case(case)
        first = responder.generate(request)
        second = responder.generate(request)
        if not isinstance(first, str) or not isinstance(second, str):
            raise TypeError(f"candidate response must be a string for {case.case_id}")
        if first != second:
            raise ValueError(f"nondeterministic candidate response for {case.case_id}")
        responses[case.case_id] = first

    scored = evaluate_suite(suite, responses)
    scored_by_id = {result.case_id: result for result in scored.case_results}
    case_results = tuple(
        HarnessCaseResult(
            case_id=case.case_id,
            category=case.category,
            response_sha256=_sha256(responses[case.case_id].encode()),
            response_utf8_bytes=len(responses[case.case_id].encode()),
            passed=scored_by_id[case.case_id].passed,
            reason=scored_by_id[case.case_id].reason,
        )
        for case in suite.cases
    )
    category_rows = []
    for category in sorted({case.category for case in suite.cases}):
        rows = [result for result in case_results if result.category == category]
        passed = sum(result.passed for result in rows)
        category_rows.append((category, passed, len(rows) - passed))
    response_hashes = [
        {"case_id": result.case_id, "response_sha256": result.response_sha256}
        for result in case_results
    ]
    provisional = HarnessResult(
        suite_id=suite.suite_id,
        suite_identity_sha256=suite.identity_sha256,
        candidate=responder.descriptor,
        candidate_evidence_scope=_evidence_scope(responder.descriptor.candidate_kind),
        response_set_sha256=_sha256(_canonical_json(response_hashes)),
        total=scored.total,
        passed_count=scored.passed_count,
        failed_count=scored.failed_count,
        passed=scored.passed,
        category_results=tuple(category_rows),
        case_results=case_results,
        result_identity_sha256="0" * 64,
    )
    identity = _sha256(_canonical_json(provisional.payload_without_identity()))
    return replace(provisional, result_identity_sha256=identity)


def validate_harness_result(payload: Mapping[str, Any]) -> None:
    """Fail closed on schema drift, training-boundary weakening, or tampering."""

    keys = {
        "schema",
        "suite_id",
        "suite_identity_sha256",
        "candidate",
        "candidate_evidence_scope",
        "evaluation_use",
        "suite_training_eligible",
        "candidate_outputs_training_eligible",
        "final_test_training_reuse_authorized",
        "raw_responses_embedded",
        "reference_responses_exposed_to_candidate",
        "claim_scope",
        "broad_quality_claim_authorized",
        "base_raw_lm_diagnostic",
        "response_set_sha256",
        "repeated_generation_verified",
        "total",
        "passed_count",
        "failed_count",
        "passed",
        "category_results",
        "case_results",
        "result_identity_sha256",
    }
    _exact_keys(payload, keys, "harness result")
    if payload["schema"] != RESULT_SCHEMA or payload["evaluation_use"] != EVALUATION_USE:
        raise ValueError("unsupported or non-evaluation harness result")
    false_fields = (
        "suite_training_eligible",
        "candidate_outputs_training_eligible",
        "final_test_training_reuse_authorized",
        "raw_responses_embedded",
        "reference_responses_exposed_to_candidate",
        "broad_quality_claim_authorized",
        "base_raw_lm_diagnostic",
    )
    if any(payload[field] is not False for field in false_fields):
        raise ValueError("harness result separation/claim boundary was weakened")
    if payload["repeated_generation_verified"] is not True:
        raise ValueError("repeatability evidence is mandatory")
    if payload["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("claim_scope mismatch")

    candidate = payload["candidate"]
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be an object")
    _exact_keys(
        candidate,
        {
            "candidate_id",
            "candidate_kind",
            "generator_identity_sha256",
            "deterministic_generation",
        },
        "candidate",
    )
    CandidateDescriptor(**candidate)
    if payload["candidate_evidence_scope"] != _evidence_scope(candidate["candidate_kind"]):
        raise ValueError("candidate evidence scope mismatch")

    cases = payload["case_results"]
    if not isinstance(cases, list) or len(cases) != payload["total"] or not cases:
        raise ValueError("invalid case_results length")
    case_keys = {
        "case_id",
        "category",
        "response_sha256",
        "response_utf8_bytes",
        "passed",
        "reason",
    }
    response_hashes = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TypeError(f"case result {index} must be an object")
        _exact_keys(case, case_keys, f"case result {index}")
        if not _valid_sha256(case["response_sha256"]):
            raise ValueError(f"invalid response hash at case result {index}")
        response_hashes.append(
            {"case_id": case["case_id"], "response_sha256": case["response_sha256"]}
        )
    if _sha256(_canonical_json(response_hashes)) != payload["response_set_sha256"]:
        raise ValueError("response_set_sha256 mismatch")
    passed_count = sum(case["passed"] is True for case in cases)
    if passed_count != payload["passed_count"]:
        raise ValueError("case pass count mismatch")
    if payload["passed_count"] + payload["failed_count"] != payload["total"]:
        raise ValueError("aggregate count mismatch")
    if payload["passed"] != (payload["failed_count"] == 0):
        raise ValueError("aggregate passed flag mismatch")

    material = dict(payload)
    claimed = material.pop("result_identity_sha256")
    if not _valid_sha256(claimed) or _sha256(_canonical_json(material)) != claimed:
        raise ValueError("harness result identity mismatch")


def write_harness_result(path: Path, result: HarnessResult) -> None:
    validate_harness_result(result.as_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.canonical_bytes())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mock-responses", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    suite = load_suite(args.manifest)
    responder = DeterministicMappingResponder(
        candidate_id=args.candidate_id,
        responses=load_responses(args.mock_responses),
    )
    result = run_candidate_harness(suite, responder)
    write_harness_result(args.output, result)
    print(result.canonical_bytes().decode(), end="")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
