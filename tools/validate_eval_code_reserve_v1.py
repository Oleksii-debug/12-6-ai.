#!/usr/bin/env python3
"""Validate the EVAL-647 sealed reservation contract and source evidence."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/evaluation/eval_code_reserve_v1.json"
DEFAULT_EVIDENCE = ROOT / "evidence/eval647/code_selection_source_materialization_v1.json"
EXPECTED = [
    {
        "source_family": "github:jd/tenacity",
        "repository": "jd/tenacity",
        "release": "9.2.0",
        "revision": "a2af454834c6bb5a1e39d67334031cdaf0f475b5",
        "tree_sha1": "8ac992632c2c1c2d38741d9fad90a12759d32cb4",
        "path": "tenacity/wait.py",
        "git_blob_sha1": "18fb6ea7b610f71f17cff7ea25de63177856dfbe",
        "expected_raw_bytes": 10438,
        "raw_sha256": "ed8fecab2e515676af051d00098b0f043c6d30ca56480d85d00902a49ae5d0c0",
        "license_spdx": "Apache-2.0",
    },
    {
        "source_family": "github:more-itertools/more-itertools",
        "repository": "more-itertools/more-itertools",
        "release": "v11.1.0",
        "revision": "64be96ceb2a6e836f76f069f4a96d2394d59fd0c",
        "tree_sha1": "f7409b66b75d5649b9fc6414114f8035362f9fcf",
        "path": "more_itertools/recipes.py",
        "git_blob_sha1": "b984d86f2341b9fb74801d9b173f5e0fd00632f3",
        "expected_raw_bytes": 45752,
        "raw_sha256": "6aff1f84b0a70b96c102e3b92a70539255f1489765fc54140b1c1478f95b4828",
        "license_spdx": "MIT",
    },
]
EXPECTED_LICENSES = {
    "jd/tenacity": {
        "license_path": "LICENSE",
        "license_git_blob_sha1": "7a4a3ea2424c09fbe48d455aed1eaa94d9124835",
        "license_raw_sha256": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
    },
    "more-itertools/more-itertools": {
        "license_path": "LICENSE",
        "license_git_blob_sha1": "0a523bece3e50519653c4d7a38399baa487fefa1",
        "license_raw_sha256": "09f1c8c9e941af3e584d59641ea9b87d83c0cb0fd007eb5ef391a7e2643c1a46",
    },
}
EXPECTED_EVIDENCE_IDENTITY = "3401db10bad35fd1c6fac2839413fc6afffac58fa5f2135d2202b944bc2fda82"
REQUIRED_PENDING_GATES = (
    "PROJECT_HISTORY_TOKENIZER_EXPOSURE_ZERO_PROVEN",
    "PROJECT_HISTORY_TRAINING_EXPOSURE_ZERO_PROVEN",
    "GLOBAL_CORPUS_EXACT_AND_NEAR_OVERLAP_ZERO_PROVEN",
    "FUTURE_TRAINING_EXCLUSION_CONSUMED_BY_CORPUS_PIPELINE",
    "PURPOSE_SPECIFIC_EVALUATION_AUTHORITY_TERMINAL",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_integer(value: object, expected: int, message: str) -> None:
    _require(type(value) is int and value == expected, message)

def _require_exact_fields(value: object, expected: set[str], label: str) -> None:
    _require(
        type(value) is dict and set(value) == expected,
        f"{label} fields are not closed-world",
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_constant:{value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("json_number_not_finite")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ValueError("nonzero_json_number_underflowed_to_zero")
    return parsed


def _require_finite_json_value(
    value: object, *, label: str, depth: int = 0,
    _budget: list[int] | None = None, _is_key: bool = False,
) -> None:
    # Apply the same parsed-value node budget to programmatic inputs as to
    # file-backed JSON. Count repeated DAG references, not only unique objects;
    # otherwise a tiny shared graph can require exponential traversal.
    _require(depth <= 64, f"{label} JSON nesting limit exceeded")
    if _budget is None:
        _budget = [0]
    if not _is_key:
        _budget[0] += 1
        _require(_budget[0] <= MAX_JSON_NODES, f"{label} exceeds node limit")
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains non-finite float")
        return
    if type(value) is dict:
        for key, item in value.items():
            _require(type(key) is str, f"{label} has a non-string JSON key")
            _require_finite_json_value(
                key, label=f"{label}.key", depth=depth + 1,
                _budget=_budget, _is_key=True,
            )
            _require_finite_json_value(
                item, label=f"{label}.{key}", depth=depth + 1, _budget=_budget,
            )
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_finite_json_value(
                item, label=f"{label}[{index}]", depth=depth + 1, _budget=_budget,
            )
        return
    if type(value) is str:
        _require(len(value) <= MAX_INPUT_BYTES, f"{label} exceeds byte limit")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError(f"{label} contains invalid UTF-8") from exc
        return
    _require(
        value is None or type(value) is int or type(value) is bool,
        f"{label} is not a JSON scalar",
    )


MAX_INPUT_BYTES = 1_048_576
MAX_JSON_NODES = 10_000


def _load_mapping(path: Path) -> dict[str, Any]:
    # These two reservation authorities are small. Refuse resource-heavy
    # untrusted input before JSON decoding or any scientific validation.
    with path.open("rb") as source:
        raw = source.read(MAX_INPUT_BYTES + 1)
    _require(len(raw) <= MAX_INPUT_BYTES, "EVAL647 authority exceeds byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds decoder limit") from exc
    _require(type(value) is dict, f"{path} must contain a JSON object")

    pending: list[Any] = [value]
    nodes = 0
    while pending:
        current = pending.pop()
        nodes += 1
        _require(nodes <= MAX_JSON_NODES, "EVAL647 authority exceeds node limit")
        if type(current) is dict:
            pending.extend(current.values())
        elif type(current) is list:
            pending.extend(current)
    _require_finite_json_value(value, label=str(path))
    return value


def validate_document(doc: dict[str, Any]) -> dict[str, Any]:
    _require(type(doc) is dict, "reservation document must be a JSON object")
    _require_finite_json_value(doc, label="reservation document")
    _require(
        len(_canonical_bytes(doc)) <= MAX_INPUT_BYTES,
        "reservation document exceeds byte limit",
    )
    _require_exact_fields(doc, {
        "schema_version", "worker_id", "issue", "execution_class",
        "purpose", "predecessor", "reservation", "objects",
        "materialization_evidence", "completed_successor_gates",
        "remaining_successor_gates", "terminal_status", "truth_boundary",
    }, "reservation contract")
    _require(doc.get("schema_version") == "12-6.eval-code-reserve-v1.contract.v1", "schema drift")
    _require(doc.get("worker_id") == "EVAL-647-CODE-SELECTION-RESERVE-V1", "worker drift")
    _require_exact_integer(doc.get("issue"), 647, "issue binding drift")
    _require(doc.get("execution_class") == "LOCAL_FREE", "execution class drift")
    _require(doc.get("purpose") == "selection_validation_only", "purpose drift")
    predecessor = doc.get("predecessor")
    _require(type(predecessor) is dict, "predecessor must be a JSON object")
    _require_exact_fields(predecessor, {"worker_id", "head_sha"}, "predecessor")
    _require(predecessor.get("worker_id") == "NEXT100-057-CODE-EVAL-SET-V2", "predecessor worker drift")
    _require(predecessor.get("head_sha") == "6713fe972b875b8a516122bda347264fb4099b2b", "predecessor head drift")

    reservation = doc.get("reservation")
    _require(type(reservation) is dict, "reservation must be a JSON object")
    _require_exact_fields(reservation, {
        "effective_at_utc", "minimum_independent_families", "final_test",
        "final_test_payload_access_allowed", "final_test_outcome_access_allowed",
        "training_allowed", "tokenizer_fit_allowed",
        "permanent_future_training_exclusion", "historical_training_exposure_required",
        "historical_tokenizer_fit_exposure_required", "training_overlap_required",
        "raw_payload_persisted_in_repository",
    }, "reservation")
    _require(reservation.get("effective_at_utc") == "2026-08-26T19:46:57Z", "reservation timestamp drift")
    _require_exact_integer(reservation.get("minimum_independent_families"), 2, "family minimum drift")
    _require(reservation.get("final_test") is False, "final-test boundary widened")
    _require(reservation.get("final_test_payload_access_allowed") is False, "final-test payload boundary widened")
    _require(reservation.get("final_test_outcome_access_allowed") is False, "final-test outcome boundary widened")
    _require(reservation.get("training_allowed") is False, "training accidentally allowed")
    _require(reservation.get("tokenizer_fit_allowed") is False, "tokenizer fitting accidentally allowed")
    _require(reservation.get("permanent_future_training_exclusion") is True, "future exclusion missing")
    _require_exact_integer(reservation.get("historical_training_exposure_required"), 0, "historical training boundary drift")
    _require_exact_integer(reservation.get("historical_tokenizer_fit_exposure_required"), 0, "historical tokenizer boundary drift")
    _require_exact_integer(reservation.get("training_overlap_required"), 0, "overlap boundary drift")
    _require(reservation.get("raw_payload_persisted_in_repository") is False, "raw eval payload must not be persisted")

    objects = doc.get("objects")
    _require(isinstance(objects, list) and len(objects) == 2, "exact two-object reservation required")
    for observed, expected in zip(objects, EXPECTED, strict=True):
        _require(type(observed) is dict, "reserved object must be an object")
        _require_exact_fields(observed, set(expected) | {
            "evaluation_use", "training_allowed", "tokenizer_fit_allowed",
            "permanent_future_training_exclusion",
        }, "reserved object")
        for key, value in expected.items():
            _require(
                type(observed.get(key)) is type(value) and observed.get(key) == value,
                f"identity drift for {expected['repository']}:{key}",
            )
        _require(observed.get("evaluation_use") == "selection_validation", "evaluation purpose drift")
        _require(observed.get("training_allowed") is False, "object training accidentally allowed")
        _require(observed.get("tokenizer_fit_allowed") is False, "object tokenizer fitting accidentally allowed")
        _require(observed.get("permanent_future_training_exclusion") is True, "object future exclusion missing")

    families = {row["source_family"] for row in objects}
    _require(len(families) == 2, "two independent upstream families required")
    _require(doc.get("completed_successor_gates") == ["IMMUTABLE_RAW_BYTES_MATERIALIZED_AND_SHA256_SEALED"], "completed gate drift")
    _require(
        type(doc.get("remaining_successor_gates")) is list
        and doc["remaining_successor_gates"] == list(REQUIRED_PENDING_GATES),
        "remaining successor gates drift",
    )
    _require(doc.get("terminal_status") == "EXACT_RAW_OBJECTS_SEALED_PENDING_PROJECT_OVERLAP_AUDIT", "terminal status drift")
    evidence_ref = doc.get("materialization_evidence")
    _require(type(evidence_ref) is dict, "materialization evidence reference must be an object")
    _require_exact_fields(evidence_ref, {"path", "identity_sha256"}, "evidence reference")
    _require(evidence_ref.get("path") == "evidence/eval647/code_selection_source_materialization_v1.json", "evidence path drift")
    _require(evidence_ref.get("identity_sha256") == EXPECTED_EVIDENCE_IDENTITY, "evidence identity drift")
    truth = doc.get("truth_boundary")
    _require(type(truth) is dict, "reservation truth boundary must be an object")
    _require_exact_fields(truth, {
        "selection_validation_records_authorized", "final_test_touched",
        "model_training_authorized", "optimizer_updates_authorized",
        "tokenizer_fit_authorized", "training_executed",
        "learned_weights_created", "paid_compute_used",
    }, "reservation truth boundary")
    _require_exact_integer(truth.get("selection_validation_records_authorized"), 0, "selection records prematurely authorized")
    _require(truth.get("final_test_touched") is False, "final test touched")
    _require(truth.get("model_training_authorized") is False, "model training prematurely authorized")
    _require_exact_integer(truth.get("optimizer_updates_authorized"), 0, "optimizer updates prematurely authorized")
    _require(truth.get("tokenizer_fit_authorized") is False, "tokenizer prematurely authorized")
    _require(truth.get("training_executed") is False, "training execution fabricated")
    _require(truth.get("learned_weights_created") is False, "learned weights fabricated")
    _require(truth.get("paid_compute_used") is False, "paid compute truth drift")
    return {
        "status": doc["terminal_status"],
        "reserved_objects": len(objects),
        "independent_families": len(families),
        "selection_validation_records_authorized": 0,
    }


def validate_materialization_evidence(doc: dict[str, Any], evidence: dict[str, Any]) -> None:
    _require(type(doc) is dict, "reservation document must be a JSON object")
    _require(type(evidence) is dict, "materialization evidence must be a JSON object")
    # A direct caller must not obtain valid evidence against an unvalidated
    # contract that widens training, tokenizer, or final-test authority.
    validate_document(doc)
    _require_finite_json_value(evidence, label="materialization evidence")
    _require(
        len(_canonical_bytes(evidence)) <= MAX_INPUT_BYTES,
        "materialization evidence exceeds byte limit",
    )
    _require_exact_fields(evidence, {
        "completed_gate", "discovery_head_sha", "evidence_identity_sha256",
        "execution_profile", "object_set_identity_sha256", "objects",
        "raw_payload_persisted_in_repository", "remaining_gates",
        "repeat_execution_byte_identical", "repeat_materializations",
        "reservation_authority_issue", "schema_version",
        "selection_validation_records_authorized", "status",
        "swarm_control_issue", "truth_boundary", "worker_id",
        "workflow_conclusion", "workflow_job_id", "workflow_run_id",
    }, "materialization evidence")
    _require(evidence.get("schema_version") == "12-6.eval-code-reserve-v1.source-materialization-terminal.v1", "evidence schema drift")
    claimed = evidence.get("evidence_identity_sha256")
    _require(claimed == EXPECTED_EVIDENCE_IDENTITY, "evidence claimed identity drift")
    body = deepcopy(evidence)
    body.pop("evidence_identity_sha256", None)
    _require(hashlib.sha256(_canonical_bytes(body)).hexdigest() == claimed, "evidence self-hash drift")
    _require(evidence.get("execution_profile") == "LOCAL_FREE", "evidence execution profile drift")
    _require(evidence.get("workflow_conclusion") == "success", "source execution was not successful")
    _require_exact_integer(evidence.get("repeat_materializations"), 2, "repeat materialization count drift")
    _require(evidence.get("repeat_execution_byte_identical") is True, "source materialization was not deterministic")
    _require(evidence.get("raw_payload_persisted_in_repository") is False, "source payload persisted in evidence")
    _require_exact_integer(evidence.get("selection_validation_records_authorized"), 0, "evidence prematurely authorizes selection records")
    _require(evidence.get("status") == doc.get("terminal_status"), "evidence/contract status drift")
    _require(
        evidence.get("completed_gate")
        == "IMMUTABLE_RAW_BYTES_MATERIALIZED_AND_SHA256_SEALED"
        and doc.get("completed_successor_gates") == [evidence["completed_gate"]],
        "evidence completed-gate mismatch",
    )
    _require_exact_integer(
        evidence.get("reservation_authority_issue"), 647,
        "evidence reservation authority issue drift",
    )
    _require(
        type(evidence.get("remaining_gates")) is list
        and evidence["remaining_gates"] == list(REQUIRED_PENDING_GATES)
        and evidence["remaining_gates"] == doc.get("remaining_successor_gates"),
        "evidence remaining gates drift",
    )
    expected_by_repo = {row["repository"]: row for row in EXPECTED}
    observed = evidence.get("objects")
    _require(isinstance(observed, list) and len(observed) == 2, "evidence object count drift")
    seen_repositories: set[str] = set()
    for row in observed:
        _require(type(row) is dict, "evidence reserved object must be an object")
        _require_exact_fields(row, {
            "evaluation_use", "git_blob_sha1", "license_git_blob_sha1",
            "license_path", "license_raw_sha256", "license_spdx", "path",
            "permanent_future_training_exclusion", "raw_bytes", "raw_sha256",
            "repository", "revision", "source_family",
            "tokenizer_fit_allowed", "training_allowed",
        }, "evidence reserved object")
        _require(
            type(row.get("repository")) is str,
            "evidence repository must be a string",
        )
        expected = expected_by_repo.get(row["repository"])
        _require(expected is not None, "unexpected evidence repository")
        _require(row["repository"] not in seen_repositories, "duplicate evidence repository")
        seen_repositories.add(row["repository"])
        for key in ("source_family", "repository", "revision", "path", "git_blob_sha1", "raw_sha256", "license_spdx"):
            _require(row.get(key) == expected.get(key), f"evidence identity drift: {key}")
        license_expected = EXPECTED_LICENSES[row["repository"]]
        for key, value in license_expected.items():
            _require(row.get(key) == value, f"evidence license identity drift: {key}")
        _require_exact_integer(row.get("raw_bytes"), expected["expected_raw_bytes"], "evidence raw byte-count drift")
        _require(
            row.get("evaluation_use") == "selection_validation",
            "evidence purpose drift",
        )
        _require(row.get("training_allowed") is False, "evidence training boundary widened")
        _require(row.get("tokenizer_fit_allowed") is False, "evidence tokenizer boundary widened")
        _require(row.get("permanent_future_training_exclusion") is True, "evidence future exclusion missing")
    _require(seen_repositories == set(expected_by_repo), "evidence family membership drift")
    reservation = doc.get("reservation")
    _require(
        type(reservation) is dict
        and reservation.get("effective_at_utc") == "2026-08-26T19:46:57Z",
        "evidence reservation timestamp drift",
    )
    identity_payload = {
        "reservation_effective_at_utc": reservation["effective_at_utc"],
        "objects": observed,
    }
    _require(
        evidence.get("object_set_identity_sha256") == hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest(),
        "evidence object-set identity drift",
    )
    truth = evidence.get("truth_boundary")
    _require(type(truth) is dict, "evidence truth boundary must be an object")
    _require_exact_fields(truth, {
        "external_llm_or_api_used_for_data_or_intelligence",
        "final_test_outcomes_read", "final_test_payload_accessed",
        "foreign_pretrained_weights_used", "learned_weights_created",
        "model_training_authorized", "optimizer_updates_authorized",
        "paid_compute_used", "tokenizer_fit_authorized", "training_executed",
    }, "evidence truth boundary")
    for key in ("final_test_outcomes_read", "final_test_payload_accessed", "model_training_authorized", "tokenizer_fit_authorized", "training_executed", "learned_weights_created", "paid_compute_used", "foreign_pretrained_weights_used", "external_llm_or_api_used_for_data_or_intelligence"):
        _require(truth.get(key) is False, f"evidence truth boundary widened: {key}")
    _require_exact_integer(truth.get("optimizer_updates_authorized"), 0, "evidence optimizer authority widened")


def validate(path: Path = DEFAULT_MANIFEST, evidence_path: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    doc = _load_mapping(path)
    result = validate_document(doc)
    evidence = _load_mapping(evidence_path)
    validate_materialization_evidence(doc, evidence)
    return result


def main() -> int:
    try:
        result = validate(DEFAULT_MANIFEST, DEFAULT_EVIDENCE)
    except (OSError, ValueError, RecursionError) as exc:
        # Data/contract rejection is not a Python traceback or evidence of
        # successful evaluation. Preserve a one-line machine-readable failure.
        print(json.dumps({
            "status": "BLOCKED_INVALID_EVAL647_AUTHORITY",
            "error": str(exc),
            "selection_validation_records_authorized": 0,
            "model_training_authorized": False,
            "final_test_outcomes_read": False,
        }, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
