"""Immutable post-Base communication-behavior evaluation suite mechanics.

This module is deliberately separate from canonical Base raw-LM diagnostics. It does
not call a model, update weights, consume Base evaluation evidence, or make broad
capability claims. It only validates a frozen project-authored fixture suite and
scores externally supplied assistant responses with deterministic rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUITE_SCHEMA = "12-6.post-base.communication-eval-suite.v1"
CASE_SCHEMA = "12-6.post-base.communication-eval-case.v1"
EVIDENCE_NAMESPACE = "evidence/post_base/eval354"
BASE_EVIDENCE_NAMESPACE = "evidence/base"
ALLOWED_CATEGORIES = frozenset(
    {
        "instruction_adherence",
        "dialogue_consistency",
        "unavailable_tool_results",
        "formatting",
        "context_handling",
    }
)
ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
ALLOWED_EXPECTATIONS = frozenset({"exact_text", "json_exact", "contains"})
PROJECT_AUTHORED = "PROJECT_AUTHORED"
CLAIM_SCOPE = "fixture_behavior_only"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    return records


def _require_exact_keys(record: dict[str, Any], keys: set[str], *, label: str) -> None:
    actual = set(record)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"{label} keys mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Expectation:
    kind: str
    value: Any
    forbidden_substrings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommunicationCase:
    case_id: str
    category: str
    messages: tuple[Message, ...]
    tool_state: tuple[tuple[str, str], ...]
    expectation: Expectation
    reference_response: str


@dataclass(frozen=True, slots=True)
class CommunicationSuite:
    suite_id: str
    version: int
    identity_sha256: str
    cases_sha256: str
    cases: tuple[CommunicationCase, ...]
    category_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    category: str
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SuiteResult:
    suite_id: str
    suite_identity_sha256: str
    total: int
    passed_count: int
    failed_count: int
    passed: bool
    case_results: tuple[CaseResult, ...]
    claim_scope: str = CLAIM_SCOPE
    broad_intelligence_claim: bool = False
    base_raw_lm_diagnostic: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_identity_sha256": self.suite_identity_sha256,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "passed": self.passed,
            "claim_scope": self.claim_scope,
            "broad_intelligence_claim": self.broad_intelligence_claim,
            "base_raw_lm_diagnostic": self.base_raw_lm_diagnostic,
            "case_results": [
                {
                    "case_id": item.case_id,
                    "category": item.category,
                    "passed": item.passed,
                    "reason": item.reason,
                }
                for item in self.case_results
            ],
        }


def _parse_expectation(raw: dict[str, Any], *, case_id: str) -> Expectation:
    _require_exact_keys(
        raw,
        {"kind", "value", "forbidden_substrings"},
        label=f"expectation for {case_id}",
    )
    kind = raw["kind"]
    if kind not in ALLOWED_EXPECTATIONS:
        raise ValueError(f"unsupported expectation kind for {case_id}: {kind!r}")
    forbidden = raw["forbidden_substrings"]
    if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
        raise ValueError(f"forbidden_substrings must be a string list for {case_id}")
    if kind in {"exact_text", "contains"} and not isinstance(raw["value"], str | list):
        raise ValueError(f"invalid expectation value for {case_id}")
    if kind == "exact_text" and not isinstance(raw["value"], str):
        raise ValueError(f"exact_text value must be a string for {case_id}")
    if kind == "contains":
        value = raw["value"]
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) for item in value)
        ):
            raise ValueError(f"contains value must be a non-empty string list for {case_id}")
    return Expectation(kind=kind, value=raw["value"], forbidden_substrings=tuple(forbidden))


def _parse_case(raw: dict[str, Any]) -> CommunicationCase:
    _require_exact_keys(
        raw,
        {
            "schema",
            "case_id",
            "category",
            "provenance",
            "messages",
            "tool_state",
            "expectation",
            "reference_response",
        },
        label="communication case",
    )
    if raw["schema"] != CASE_SCHEMA:
        raise ValueError("unsupported communication case schema")
    case_id = raw["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    category = raw["category"]
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"unsupported category for {case_id}: {category!r}")
    if raw["provenance"] != PROJECT_AUTHORED:
        raise ValueError(f"non-project-authored fixture rejected: {case_id}")

    raw_messages = raw["messages"]
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"messages must be a non-empty list for {case_id}")
    messages: list[Message] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be an object for {case_id}")
        _require_exact_keys(message, {"role", "content"}, label=f"message {index} for {case_id}")
        role = message["role"]
        content = message["content"]
        if role not in ALLOWED_ROLES or not isinstance(content, str) or not content:
            raise ValueError(f"invalid message {index} for {case_id}")
        messages.append(Message(role=role, content=content))

    raw_tool_state = raw["tool_state"]
    if not isinstance(raw_tool_state, dict):
        raise TypeError(f"tool_state must be an object for {case_id}")
    tool_state: list[tuple[str, str]] = []
    for tool_name, state in sorted(raw_tool_state.items()):
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"invalid tool name for {case_id}")
        if state not in {"available", "unavailable"}:
            raise ValueError(f"invalid tool state for {case_id}: {tool_name}")
        tool_state.append((tool_name, state))

    expectation_raw = raw["expectation"]
    if not isinstance(expectation_raw, dict):
        raise TypeError(f"expectation must be an object for {case_id}")
    expectation = _parse_expectation(expectation_raw, case_id=case_id)
    reference_response = raw["reference_response"]
    if not isinstance(reference_response, str):
        raise TypeError(f"reference_response must be a string for {case_id}")

    return CommunicationCase(
        case_id=case_id,
        category=category,
        messages=tuple(messages),
        tool_state=tuple(tool_state),
        expectation=expectation,
        reference_response=reference_response,
    )


def _manifest_identity(manifest: dict[str, Any]) -> str:
    material = dict(manifest)
    material.pop("suite_identity_sha256", None)
    return _sha256_bytes(_canonical_json(material))


def load_suite(manifest_path: Path) -> CommunicationSuite:
    """Load and cryptographically verify the immutable EVAL-354 v1 suite."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("suite manifest must be an object")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "suite_id",
            "version",
            "purpose",
            "provenance",
            "separation",
            "claims",
            "cases_file",
            "cases_sha256",
            "case_count",
            "category_counts",
            "suite_identity_sha256",
        },
        label="suite manifest",
    )
    if manifest["schema"] != SUITE_SCHEMA:
        raise ValueError("unsupported communication evaluation suite schema")
    if manifest["purpose"] != "post_base_communication_behavior_evaluation":
        raise ValueError("suite purpose must remain post-Base communication behavior evaluation")
    provenance = manifest["provenance"]
    if provenance != {"origin": PROJECT_AUTHORED, "foreign_model_output": False}:
        raise ValueError(
            "suite provenance must remain project-authored with no foreign model output"
        )
    separation = manifest["separation"]
    if separation != {
        "base_raw_lm_diagnostics": False,
        "base_evidence_namespace": BASE_EVIDENCE_NAMESPACE,
        "post_base_evidence_namespace": EVIDENCE_NAMESPACE,
        "training_eligible": False,
    }:
        raise ValueError(
            "communication evaluation must remain separate from Base diagnostics/training"
        )
    claims = manifest["claims"]
    if claims != {
        "scope": CLAIM_SCOPE,
        "broad_intelligence_claim_authorized": False,
    }:
        raise ValueError("suite claims boundary was weakened")
    if not isinstance(manifest["version"], int) or manifest["version"] != 1:
        raise ValueError("EVAL-354 v1 manifest version must equal 1")
    if not isinstance(manifest["suite_id"], str) or not manifest["suite_id"]:
        raise ValueError("suite_id must be a non-empty string")
    expected_identity = _manifest_identity(manifest)
    if manifest["suite_identity_sha256"] != expected_identity:
        raise ValueError("suite manifest identity mismatch")

    cases_file = manifest["cases_file"]
    if not isinstance(cases_file, str) or Path(cases_file).name != cases_file:
        raise ValueError("cases_file must be a same-directory filename")
    cases_path = manifest_path.parent / cases_file
    cases_payload = cases_path.read_bytes()
    if _sha256_bytes(cases_payload) != manifest["cases_sha256"]:
        raise ValueError("communication fixture bytes do not match immutable manifest hash")

    raw_cases = _read_jsonl(cases_path)
    if len(raw_cases) != manifest["case_count"]:
        raise ValueError("communication fixture case_count mismatch")
    cases = tuple(_parse_case(item) for item in raw_cases)
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("communication fixture case_id values must be unique")

    observed_counts = {category: 0 for category in sorted(ALLOWED_CATEGORIES)}
    for case in cases:
        observed_counts[case.category] += 1
    if manifest["category_counts"] != observed_counts:
        raise ValueError("communication fixture category_counts mismatch")
    if any(count <= 0 for count in observed_counts.values()):
        raise ValueError("every required communication behavior category needs at least one case")

    return CommunicationSuite(
        suite_id=manifest["suite_id"],
        version=manifest["version"],
        identity_sha256=manifest["suite_identity_sha256"],
        cases_sha256=manifest["cases_sha256"],
        cases=cases,
        category_counts=tuple(sorted(observed_counts.items())),
    )


def evaluate_case(case: CommunicationCase, response: str) -> CaseResult:
    if not isinstance(response, str):
        return CaseResult(case.case_id, case.category, False, "response_not_string")
    expectation = case.expectation
    for forbidden in expectation.forbidden_substrings:
        if forbidden in response:
            return CaseResult(case.case_id, case.category, False, "forbidden_substring")

    if expectation.kind == "exact_text":
        passed = response == expectation.value
        reason = "exact_match" if passed else "exact_text_mismatch"
    elif expectation.kind == "json_exact":
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return CaseResult(case.case_id, case.category, False, "invalid_json")
        passed = parsed == expectation.value
        reason = "json_exact_match" if passed else "json_value_mismatch"
    elif expectation.kind == "contains":
        required = expectation.value
        passed = all(item in response for item in required)
        reason = "required_substrings_present" if passed else "missing_required_substring"
    else:  # pragma: no cover - loader rejects unsupported kinds
        raise AssertionError(f"unsupported expectation kind: {expectation.kind}")
    return CaseResult(case.case_id, case.category, passed, reason)


def evaluate_suite(suite: CommunicationSuite, responses: dict[str, str]) -> SuiteResult:
    """Score one complete externally supplied response set; missing/extra IDs fail closed."""
    expected_ids = {case.case_id for case in suite.cases}
    supplied_ids = set(responses)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids - supplied_ids)
        extra = sorted(supplied_ids - expected_ids)
        raise ValueError(f"response case IDs mismatch; missing={missing}, extra={extra}")
    results = tuple(evaluate_case(case, responses[case.case_id]) for case in suite.cases)
    passed_count = sum(item.passed for item in results)
    return SuiteResult(
        suite_id=suite.suite_id,
        suite_identity_sha256=suite.identity_sha256,
        total=len(results),
        passed_count=passed_count,
        failed_count=len(results) - passed_count,
        passed=passed_count == len(results),
        case_results=results,
    )


def load_responses(path: Path) -> dict[str, str]:
    records = _read_jsonl(path)
    responses: dict[str, str] = {}
    for index, record in enumerate(records):
        _require_exact_keys(record, {"case_id", "response"}, label=f"response record {index}")
        case_id = record["case_id"]
        response = record["response"]
        if not isinstance(case_id, str) or not isinstance(response, str):
            raise TypeError(f"invalid response record {index}")
        if case_id in responses:
            raise ValueError(f"duplicate response case_id: {case_id}")
        responses[case_id] = response
    return responses


def reference_responses(suite: CommunicationSuite) -> dict[str, str]:
    """Return project-authored scorer-control responses; never model evidence."""
    return {case.case_id: case.reference_response for case in suite.cases}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    suite = load_suite(args.manifest)
    result = evaluate_suite(suite, load_responses(args.responses))
    payload = json.dumps(result.as_dict(), sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
