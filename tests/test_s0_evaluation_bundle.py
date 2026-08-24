from __future__ import annotations

import hashlib
import json

import pytest

from twelve_six.evaluation_evidence import (
    S0EvaluationEvidenceError,
    build_s0_evaluation_bundle,
    validate_s0_evaluation_bundle,
)
from twelve_six.training.s0_evidence_contract import (
    LOCK_INDEX_FILE_SHA256,
    LOCK_INDEX_PATH,
    LOCK_INDEX_SEMANTIC_SHA256,
    LOCK_PROFILE_FILE_SHA256,
    LOCK_PROFILE_ID,
    LOCK_PROFILE_MANIFEST_SHA256,
    PYTHON_VERSION,
)

CANDIDATE_SHA = "a" * 40
REPORT_HASHES = {
    "candidate_evidence.json": "1" * 64,
    "stage_gate_report.json": "2" * 64,
    "promotion_eligibility.json": "3" * 64,
}


def _canonical_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _locked_environment(source_sha: str = CANDIDATE_SHA) -> dict:
    evidence = {
        "schema_version": "12-6.locked-environment-evidence.v1",
        "source_sha": source_sha,
        "profile_id": LOCK_PROFILE_ID,
        "python": {"version": PYTHON_VERSION},
        "lock_index": {
            "path": LOCK_INDEX_PATH,
            "file_sha256": LOCK_INDEX_FILE_SHA256,
            "index_sha256": LOCK_INDEX_SEMANTIC_SHA256,
        },
        "lock_profile": {
            "manifest_sha256": LOCK_PROFILE_MANIFEST_SHA256,
            "file_sha256": LOCK_PROFILE_FILE_SHA256,
        },
        "verification": {
            "committed_lock_validation": "PASS",
            "editable_install_import_cli": "PASS",
            "wheel_install_import_cli": "PASS",
            "repo_checks": "PASS",
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    return evidence


def _candidate_evidence() -> dict:
    return {
        "candidate": {"sha": CANDIDATE_SHA},
        "tokenizer": {
            "config_sha256": "4" * 64,
            "vocab_sha256": "5" * 64,
        },
        "dataset": {"dataset_id": "s0-tiny-controlled-v1"},
        "checkpoint": {
            "environment_lock_sha256": LOCK_INDEX_FILE_SHA256,
            "final_checkpoint_id": "6" * 64,
        },
        "provenance": {
            "repository": "Oleksii-debug/12-6-ai.",
            "checkout_head_sha": CANDIDATE_SHA,
            "environment_lock_sha256": LOCK_INDEX_FILE_SHA256,
            "dataset_manifest_sha256": "7" * 64,
            "packing_sha256": "8" * 64,
        },
    }


def _stage_gate_report() -> dict:
    return {
        "candidate": {"sha": CANDIDATE_SHA},
        "promotion_authority": {
            "status": "NOT_TESTED",
            "blockers": ["missing exact candidate-bound audits"],
        },
        "summary": {
            "overall_status": "PASS",
            "evaluation_complete": True,
            "promotion_eligible": False,
            "promotion_authority_status": "NOT_TESTED",
            "counts": {"PASS": 15, "FAIL": 0, "NOT_TESTED": 0},
        },
    }


def _promotion_report() -> dict:
    return {
        "candidate_sha": CANDIDATE_SHA,
        "evaluation_complete": True,
        "quality_overall_status": "PASS",
        "promotion_eligible": False,
        "promotion_authority_status": "NOT_TESTED",
        "promotion_blockers": ["missing exact candidate-bound audits"],
    }


def _build(**overrides) -> dict:
    values = {
        "candidate_sha": CANDIDATE_SHA,
        "candidate_evidence": _candidate_evidence(),
        "stage_gate_report": _stage_gate_report(),
        "promotion_report": _promotion_report(),
        "locked_environment_evidence": _locked_environment(),
        "report_hashes": REPORT_HASHES,
        "require_quality_pass": True,
    }
    values.update(overrides)
    return build_s0_evaluation_bundle(**values)


def test_bundle_binds_exact_candidate_runtime_and_all_reports() -> None:
    bundle = _build()
    validate_s0_evaluation_bundle(bundle)
    assert bundle["candidate_sha"] == CANDIDATE_SHA
    assert bundle["locked_environment"]["lock_index_file_sha256"] == (
        LOCK_INDEX_FILE_SHA256
    )
    assert bundle["summary"]["quality_gate_counts"] == {
        "PASS": 15,
        "FAIL": 0,
        "NOT_TESTED": 0,
    }
    assert bundle["summary"]["promotion_eligible"] is False


def test_stale_locked_environment_source_sha_is_rejected() -> None:
    with pytest.raises(ValueError, match="source SHA mismatch"):
        _build(locked_environment_evidence=_locked_environment("b" * 40))


def test_locked_environment_tamper_is_rejected() -> None:
    environment = _locked_environment()
    environment["lock_index"]["index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="semantic SHA mismatch"):
        _build(locked_environment_evidence=environment)


def test_locked_environment_without_repo_checks_is_rejected() -> None:
    environment = _locked_environment()
    environment["verification"]["repo_checks"] = "NOT_RUN"
    environment["evidence_sha256"] = _canonical_hash(
        {key: value for key, value in environment.items() if key != "evidence_sha256"}
    )
    with pytest.raises(ValueError, match="repo_checks is not PASS"):
        _build(locked_environment_evidence=environment)


def test_mixed_candidate_promotion_report_is_rejected() -> None:
    promotion = _promotion_report()
    promotion["candidate_sha"] = "b" * 40
    with pytest.raises(S0EvaluationEvidenceError, match="promotion report SHA mismatch"):
        _build(promotion_report=promotion)


def test_report_hash_set_cannot_be_weakened() -> None:
    hashes = dict(REPORT_HASHES)
    hashes.pop("promotion_eligibility.json")
    with pytest.raises(S0EvaluationEvidenceError, match="hash set mismatch"):
        _build(report_hashes=hashes)


def test_bundle_self_hash_detects_tampering() -> None:
    bundle = _build()
    bundle["summary"]["promotion_eligible"] = True
    with pytest.raises(S0EvaluationEvidenceError, match="self-hash mismatch"):
        validate_s0_evaluation_bundle(bundle)
