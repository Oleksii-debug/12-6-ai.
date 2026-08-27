from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class LIDBakeoffError(ValueError):
    """Raised when LID bakeoff evidence violates the fail-closed contract."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LIDBakeoffError(message)


def _load_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LIDBakeoffError(f"{label} must be UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LIDBakeoffError(f"{label} must be UTF-8 JSONL") from exc
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LIDBakeoffError(f"{label} line {line_no} is invalid JSON") from exc
        _require(isinstance(value, dict), f"{label} line {line_no} must be an object")
        records.append(value)
    _require(records, f"{label} must contain at least one record")
    return records


def _component_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    components = registry.get("components")
    _require(isinstance(components, list), "registry components must be a list")
    result: dict[str, dict[str, Any]] = {}
    for component in components:
        _require(isinstance(component, dict), "registry component must be an object")
        component_id = component.get("id")
        _require(isinstance(component_id, str) and component_id, "registry component id missing")
        _require(component_id not in result, f"duplicate registry component {component_id}")
        result[component_id] = component
    return result


def validate_fixture(contract: dict[str, Any], fixture_bytes: bytes) -> list[dict[str, Any]]:
    fixture = load_jsonl_bytes(fixture_bytes, "fixture")
    fixture_cfg = contract["fixture"]
    _require(
        sha256_hex(fixture_bytes) == fixture_cfg["sha256"],
        "fixture SHA-256 does not match contract",
    )
    required_categories = set(contract["required_categories"])
    allowed_labels = set(contract["allowed_labels"])
    minimum = fixture_cfg["minimum_cases_per_category"]
    _require(isinstance(minimum, int) and minimum > 0, "minimum_cases_per_category invalid")

    ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    for record in fixture:
        case_id = record.get("case_id")
        category = record.get("category")
        expected = record.get("expected_label")
        text = record.get("text")
        _require(isinstance(case_id, str) and case_id, "fixture case_id missing")
        _require(case_id not in ids, f"duplicate fixture case_id {case_id}")
        ids.add(case_id)
        _require(category in required_categories, f"invalid fixture category for {case_id}")
        _require(expected in allowed_labels, f"invalid expected label for {case_id}")
        _require(isinstance(text, str) and text, f"fixture text missing for {case_id}")
        _require(record.get("purpose") == "lid_calibration_only", f"bad purpose for {case_id}")
        _require(record.get("project_authored") is True, f"case {case_id} must be project-authored")
        _require(record.get("training_allowed") is False, f"case {case_id} cannot allow training")
        _require(record.get("tokenizer_fit_allowed") is False, f"case {case_id} cannot allow tokenizer fit")
        _require(record.get("final_test") is False, f"case {case_id} cannot be a final test")
        _require(record.get("benchmark_material") is False, f"case {case_id} cannot be benchmark material")
        category_counts[category] += 1

    _require(set(category_counts) == required_categories, "fixture category coverage is incomplete")
    for category in required_categories:
        _require(
            category_counts[category] >= minimum,
            f"fixture category {category} has fewer than {minimum} cases",
        )
    return fixture


def validate_contract(
    contract: dict[str, Any], registry_bytes: bytes, fixture_bytes: bytes
) -> list[dict[str, Any]]:
    _require(contract.get("schema_version") == 1, "unsupported LID contract schema")
    _require(contract.get("contract_id") == "D03-LID-BAKEOFF-V1", "unexpected contract id")
    _require(contract.get("status") == "PREPARED_NOT_EXECUTED", "checked-in contract must be nonterminal")
    _require(contract.get("automatic_adoption_allowed") is False, "automatic adoption must remain disabled")

    authority = contract.get("authority")
    _require(isinstance(authority, dict), "authority block missing")
    expected_blob = authority.get("open_source_registry_blob_sha1")
    _require(isinstance(expected_blob, str) and len(expected_blob) == 40, "registry blob SHA-1 missing")
    _require(git_blob_sha1(registry_bytes) == expected_blob, "open-source registry Git blob identity drift")
    registry = _load_json_bytes(registry_bytes, "open-source registry")
    _require(registry.get("registry_id") == "OPEN-SOURCE-REUSE-REGISTRY-V2", "wrong registry id")
    _require(registry.get("schema_version") == 2, "wrong registry schema")

    components = _component_map(registry)
    candidates = contract.get("comparison_candidates")
    _require(isinstance(candidates, list) and candidates, "comparison_candidates missing")
    candidate_ids: list[str] = []
    for candidate in candidates:
        _require(isinstance(candidate, dict), "candidate entry must be object")
        candidate_id = candidate.get("id")
        _require(isinstance(candidate_id, str) and candidate_id, "candidate id missing")
        _require(candidate_id not in candidate_ids, f"duplicate candidate {candidate_id}")
        candidate_ids.append(candidate_id)
        upstream = components.get(candidate_id)
        _require(upstream is not None, f"candidate {candidate_id} missing from registry")
        for field in ("kind", "upstream", "license", "decision"):
            _require(
                candidate.get(field) == upstream.get(field),
                f"candidate {candidate_id} {field} drift from registry",
            )

    _require(
        candidate_ids == ["FASTTEXT_LID176", "OPENLID_V3", "GLOTLID", "LINGUA"],
        "comparison candidate order or membership drift",
    )
    excluded = contract.get("excluded_candidates")
    _require(isinstance(excluded, list) and len(excluded) == 1, "excluded candidate contract invalid")
    nllb = excluded[0]
    _require(nllb.get("id") == "NLLB_LID218E", "NLLB exclusion missing")
    upstream_nllb = components.get("NLLB_LID218E")
    _require(upstream_nllb is not None, "NLLB registry component missing")
    _require(
        nllb.get("decision")
        == upstream_nllb.get("decision")
        == "DO_NOT_USE_AS_HIDDEN_UNRESTRICTED_DEPENDENCY",
        "NLLB unrestricted-dependency exclusion drift",
    )
    _require(
        nllb.get("unrestricted_adoption_allowed") is False,
        "NLLB unrestricted adoption must be false",
    )

    boundaries = contract.get("truth_boundaries")
    _require(isinstance(boundaries, dict), "truth_boundaries missing")
    for forbidden in (
        "external_lid_model_executed",
        "corpus_mutated",
        "tokenizer_fit_executed",
        "model_training_executed",
        "final_test_accessed",
        "paid_compute_used",
    ):
        _require(boundaries.get(forbidden) is False, f"checked-in boundary {forbidden} must be false")

    return validate_fixture(contract, fixture_bytes)


def _validate_runtime_identity(candidate_id: str, runtime: Any) -> None:
    _require(isinstance(runtime, dict), f"runtime identity missing for {candidate_id}")
    for field in ("upstream_ref", "artifact_identity", "adapter_identity", "command_identity"):
        value = runtime.get(field)
        _require(isinstance(value, str) and value.strip(), f"{candidate_id} runtime {field} missing")


def _score_predictions(
    candidate_id: str,
    expected_by_id: dict[str, dict[str, Any]],
    predictions: Any,
    allowed_labels: set[str],
) -> dict[str, Any]:
    _require(isinstance(predictions, list), f"predictions missing for {candidate_id}")
    seen: set[str] = set()
    correct = 0
    category_total: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for prediction in predictions:
        _require(isinstance(prediction, dict), f"prediction for {candidate_id} must be object")
        case_id = prediction.get("case_id")
        predicted = prediction.get("predicted_label")
        raw_label = prediction.get("raw_label")
        _require(case_id in expected_by_id, f"unknown case {case_id} for {candidate_id}")
        _require(case_id not in seen, f"duplicate prediction {case_id} for {candidate_id}")
        seen.add(case_id)
        _require(predicted in allowed_labels, f"invalid predicted label for {candidate_id}:{case_id}")
        _require(isinstance(raw_label, str) and raw_label, f"raw label missing for {candidate_id}:{case_id}")
        confidence = prediction.get("confidence")
        if confidence is not None:
            _require(
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and math.isfinite(confidence)
                and 0.0 <= confidence <= 1.0,
                f"invalid confidence for {candidate_id}:{case_id}",
            )

        expected_record = expected_by_id[case_id]
        expected = expected_record["expected_label"]
        category = expected_record["category"]
        category_total[category] += 1
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
            category_correct[category] += 1

    missing = sorted(set(expected_by_id) - seen)
    _require(not missing, f"missing predictions for {candidate_id}: {', '.join(missing)}")
    total = len(expected_by_id)
    return {
        "candidate_id": candidate_id,
        "cases": total,
        "correct": correct,
        "accuracy": correct / total,
        "per_category_accuracy": {
            category: category_correct[category] / category_total[category]
            for category in sorted(category_total)
        },
        "confusion": {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in sorted(confusion.items())
        },
    }


def score_evidence(
    contract: dict[str, Any],
    registry_bytes: bytes,
    fixture_bytes: bytes,
    evidence: dict[str, Any],
    *,
    allow_test_evidence: bool = False,
) -> dict[str, Any]:
    fixture = validate_contract(contract, registry_bytes, fixture_bytes)
    expected_by_id = {record["case_id"]: record for record in fixture}
    candidate_ids = [candidate["id"] for candidate in contract["comparison_candidates"]]
    allowed_labels = set(contract["allowed_labels"])

    _require(evidence.get("schema_version") == 1, "unsupported evidence schema")
    _require(evidence.get("contract_id") == contract["contract_id"], "evidence contract mismatch")
    _require(evidence.get("fixture_sha256") == sha256_hex(fixture_bytes), "evidence fixture mismatch")
    _require(
        evidence.get("registry_blob_sha1") == git_blob_sha1(registry_bytes),
        "evidence registry identity mismatch",
    )
    evidence_kind = evidence.get("evidence_kind")
    allowed_kind = "TEST_FIXTURE_ONLY" if allow_test_evidence else "EXTERNAL_CANDIDATE_EXECUTION"
    _require(evidence_kind == allowed_kind, f"evidence_kind must be {allowed_kind}")

    executions = evidence.get("executions")
    _require(isinstance(executions, list), "executions must be a list")
    execution_by_id: dict[str, dict[str, Any]] = {}
    for execution in executions:
        _require(isinstance(execution, dict), "execution entry must be object")
        candidate_id = execution.get("candidate_id")
        _require(candidate_id in candidate_ids, f"unexpected execution candidate {candidate_id}")
        _require(candidate_id not in execution_by_id, f"duplicate execution {candidate_id}")
        execution_by_id[candidate_id] = execution
    _require(set(execution_by_id) == set(candidate_ids), "candidate execution coverage incomplete")

    scores: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        execution = execution_by_id[candidate_id]
        _require(execution.get("executed") is True, f"{candidate_id} was not executed")
        _validate_runtime_identity(candidate_id, execution.get("runtime_identity"))
        license_review = execution.get("license_review")
        _require(isinstance(license_review, dict), f"license review missing for {candidate_id}")
        _require(
            license_review.get("status") == "REVIEWED_FOR_BAKEOFF",
            f"license review incomplete for {candidate_id}",
        )
        _require(
            isinstance(license_review.get("reference"), str) and license_review["reference"].strip(),
            f"license review reference missing for {candidate_id}",
        )
        _require(
            execution.get("automatic_adoption_requested") is False,
            f"{candidate_id} evidence cannot request automatic adoption",
        )
        scores.append(
            _score_predictions(candidate_id, expected_by_id, execution.get("predictions"), allowed_labels)
        )

    return {
        "schema_version": 1,
        "report_id": "D03-LID-BAKEOFF-V1-REPORT",
        "comparison_status": "COMPARABLE_EVIDENCE_READY",
        "contract_id": contract["contract_id"],
        "registry_blob_sha1": git_blob_sha1(registry_bytes),
        "fixture_sha256": sha256_hex(fixture_bytes),
        "scores": scores,
        "automatic_adoption_allowed": False,
        "selected_candidate": None,
        "scientific_verdict": "NOT_ADOPTED_REQUIRES_D03_REVIEW",
    }


def preflight_report(
    contract: dict[str, Any], registry_bytes: bytes, fixture_bytes: bytes
) -> dict[str, Any]:
    fixture = validate_contract(contract, registry_bytes, fixture_bytes)
    return {
        "schema_version": 1,
        "report_id": "D03-LID-BAKEOFF-V1-PREFLIGHT",
        "status": "PREPARED_NOT_EXECUTED",
        "registry_blob_sha1": git_blob_sha1(registry_bytes),
        "fixture_sha256": sha256_hex(fixture_bytes),
        "case_count": len(fixture),
        "candidate_ids": [candidate["id"] for candidate in contract["comparison_candidates"]],
        "external_lid_model_executed": False,
        "automatic_adoption_allowed": False,
    }


def _read(path: str) -> bytes:
    return Path(path).read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/score D03 LID bakeoff evidence")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--evidence")
    parser.add_argument("--output")
    parser.add_argument("--allow-test-evidence", action="store_true")
    args = parser.parse_args(argv)

    contract = _load_json_bytes(_read(args.contract), "contract")
    registry_bytes = _read(args.registry)
    fixture_bytes = _read(args.fixture)
    if args.evidence:
        evidence = _load_json_bytes(_read(args.evidence), "evidence")
        report = score_evidence(
            contract,
            registry_bytes,
            fixture_bytes,
            evidence,
            allow_test_evidence=args.allow_test_evidence,
        )
    else:
        report = preflight_report(contract, registry_bytes, fixture_bytes)

    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
