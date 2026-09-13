#!/usr/bin/env python3
"""Validate the EVAL-647 sealed reservation contract and source evidence."""
from __future__ import annotations

import hashlib
import json
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
        "raw_sha256": "ed8fec5e66288a92f67a378611111865da025c26edee07ab1b8c88642e0dd0c0",
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
        "raw_sha256": "6aff1fac06b82008f90440892536195fc1a3fd0d933c59ee45cc0f6e007b4828",
        "license_spdx": "MIT",
    },
]
EXPECTED_EVIDENCE_IDENTITY = "ecde57cc0865e71f85dc4bce88bb17b0661fdf9d29e353a57b35ea4e04a7b6cc"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def validate_document(doc: dict[str, Any]) -> dict[str, Any]:
    _require(doc.get("schema_version") == "12-6.eval-code-reserve-v1.contract.v1", "schema drift")
    _require(doc.get("worker_id") == "EVAL-647-CODE-SELECTION-RESERVE-V1", "worker drift")
    _require(doc.get("issue") == 647, "issue binding drift")
    _require(doc.get("execution_class") == "LOCAL_FREE", "execution class drift")
    _require(doc.get("purpose") == "selection_validation_only", "purpose drift")
    predecessor = doc.get("predecessor", {})
    _require(predecessor.get("worker_id") == "NEXT100-057-CODE-EVAL-SET-V2", "predecessor worker drift")
    _require(predecessor.get("head_sha") == "6713fe972b875b8a516122bda347264fb4099b2b", "predecessor head drift")

    reservation = doc.get("reservation", {})
    _require(reservation.get("effective_at_utc") == "2026-08-26T19:46:57Z", "reservation timestamp drift")
    _require(reservation.get("minimum_independent_families") == 2, "family minimum drift")
    _require(reservation.get("final_test") is False, "final-test boundary widened")
    _require(reservation.get("final_test_payload_access_allowed") is False, "final-test payload boundary widened")
    _require(reservation.get("final_test_outcome_access_allowed") is False, "final-test outcome boundary widened")
    _require(reservation.get("training_allowed") is False, "training accidentally allowed")
    _require(reservation.get("tokenizer_fit_allowed") is False, "tokenizer fitting accidentally allowed")
    _require(reservation.get("permanent_future_training_exclusion") is True, "future exclusion missing")
    _require(reservation.get("historical_training_exposure_required") == 0, "historical training boundary drift")
    _require(reservation.get("historical_tokenizer_fit_exposure_required") == 0, "historical tokenizer boundary drift")
    _require(reservation.get("training_overlap_required") == 0, "overlap boundary drift")
    _require(reservation.get("raw_payload_persisted_in_repository") is False, "raw eval payload must not be persisted")

    objects = doc.get("objects")
    _require(isinstance(objects, list) and len(objects) == 2, "exact two-object reservation required")
    for observed, expected in zip(objects, EXPECTED, strict=True):
        for key, value in expected.items():
            _require(observed.get(key) == value, f"identity drift for {expected['repository']}:{key}")
        _require(observed.get("evaluation_use") == "selection_validation", "evaluation purpose drift")
        _require(observed.get("training_allowed") is False, "object training accidentally allowed")
        _require(observed.get("tokenizer_fit_allowed") is False, "object tokenizer fitting accidentally allowed")
        _require(observed.get("permanent_future_training_exclusion") is True, "object future exclusion missing")

    families = {row["source_family"] for row in objects}
    _require(len(families) == 2, "two independent upstream families required")
    _require(doc.get("completed_successor_gates") == ["IMMUTABLE_RAW_BYTES_MATERIALIZED_AND_SHA256_SEALED"], "completed gate drift")
    _require(doc.get("terminal_status") == "EXACT_RAW_OBJECTS_SEALED_PENDING_PROJECT_OVERLAP_AUDIT", "terminal status drift")
    evidence_ref = doc.get("materialization_evidence", {})
    _require(evidence_ref.get("path") == "evidence/eval647/code_selection_source_materialization_v1.json", "evidence path drift")
    _require(evidence_ref.get("identity_sha256") == EXPECTED_EVIDENCE_IDENTITY, "evidence identity drift")
    truth = doc.get("truth_boundary", {})
    _require(truth.get("selection_validation_records_authorized") == 0, "selection records prematurely authorized")
    _require(truth.get("final_test_touched") is False, "final test touched")
    _require(truth.get("model_training_authorized") is False, "model training prematurely authorized")
    _require(truth.get("optimizer_updates_authorized") == 0, "optimizer updates prematurely authorized")
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
    _require(evidence.get("schema_version") == "12-6.eval-code-reserve-v1.source-materialization-terminal.v1", "evidence schema drift")
    claimed = evidence.get("evidence_identity_sha256")
    _require(claimed == EXPECTED_EVIDENCE_IDENTITY, "evidence claimed identity drift")
    body = deepcopy(evidence)
    body.pop("evidence_identity_sha256", None)
    _require(hashlib.sha256(_canonical_bytes(body)).hexdigest() == claimed, "evidence self-hash drift")
    _require(evidence.get("execution_profile") == "LOCAL_FREE", "evidence execution profile drift")
    _require(evidence.get("workflow_conclusion") == "success", "source execution was not successful")
    _require(evidence.get("repeat_materializations") == 2, "repeat materialization count drift")
    _require(evidence.get("repeat_execution_byte_identical") is True, "source materialization was not deterministic")
    _require(evidence.get("raw_payload_persisted_in_repository") is False, "source payload persisted in evidence")
    _require(evidence.get("selection_validation_records_authorized") == 0, "evidence prematurely authorizes selection records")
    _require(evidence.get("status") == doc.get("terminal_status"), "evidence/contract status drift")
    expected_by_repo = {row["repository"]: row for row in EXPECTED}
    observed = evidence.get("objects")
    _require(isinstance(observed, list) and len(observed) == 2, "evidence object count drift")
    for row in observed:
        expected = expected_by_repo.get(row.get("repository"))
        _require(expected is not None, "unexpected evidence repository")
        for key in ("source_family", "repository", "revision", "path", "git_blob_sha1", "raw_sha256", "license_spdx"):
            _require(row.get(key) == expected.get(key), f"evidence identity drift: {key}")
        _require(row.get("raw_bytes") == expected["expected_raw_bytes"], "evidence raw byte-count drift")
        _require(row.get("training_allowed") is False, "evidence training boundary widened")
        _require(row.get("tokenizer_fit_allowed") is False, "evidence tokenizer boundary widened")
        _require(row.get("permanent_future_training_exclusion") is True, "evidence future exclusion missing")
    truth = evidence.get("truth_boundary", {})
    for key in ("final_test_outcomes_read", "final_test_payload_accessed", "model_training_authorized", "tokenizer_fit_authorized", "training_executed", "learned_weights_created", "paid_compute_used", "foreign_pretrained_weights_used", "external_llm_or_api_used_for_data_or_intelligence"):
        _require(truth.get(key) is False, f"evidence truth boundary widened: {key}")
    _require(truth.get("optimizer_updates_authorized") == 0, "evidence optimizer authority widened")


def validate(path: Path = DEFAULT_MANIFEST, evidence_path: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    result = validate_document(doc)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    validate_materialization_evidence(doc, evidence)
    return result


def main() -> int:
    print(json.dumps(validate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
