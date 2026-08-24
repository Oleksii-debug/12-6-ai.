"""Fail-closed binding for exact-candidate S0 evaluation evidence bundles."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from twelve_six.training.s0_evidence_contract import (
    LOCK_INDEX_FILE_SHA256,
    validate_locked_environment_evidence,
)

SCHEMA_VERSION = "12-6.s0-evaluation-bundle.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_NAMES = (
    "candidate_evidence.json",
    "stage_gate_report.json",
    "promotion_eligibility.json",
)


class S0EvaluationEvidenceError(ValueError):
    """Raised when D04 evaluation evidence cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0EvaluationEvidenceError(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} block missing")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_report_hashes(report_hashes: Mapping[str, Any]) -> dict[str, str]:
    _require(set(report_hashes) == set(_REPORT_NAMES), "evaluation report hash set mismatch")
    result: dict[str, str] = {}
    for name in _REPORT_NAMES:
        value = report_hashes.get(name)
        _require(
            isinstance(value, str) and _SHA256.fullmatch(value) is not None,
            f"{name} SHA-256 is invalid",
        )
        result[name] = value
    return result


def build_s0_evaluation_bundle(
    *,
    candidate_sha: str,
    candidate_evidence: Mapping[str, Any],
    stage_gate_report: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    locked_environment_evidence: Mapping[str, Any],
    report_hashes: Mapping[str, Any],
    require_quality_pass: bool = False,
) -> dict[str, Any]:
    """Validate and bind D04 reports to one exact candidate and D08 runtime."""

    _require(
        _GIT_SHA.fullmatch(candidate_sha) is not None,
        "candidate SHA must be full lowercase Git SHA",
    )
    runtime_binding = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=candidate_sha,
    )
    hashes = _validate_report_hashes(report_hashes)

    candidate = _mapping(candidate_evidence.get("candidate"), "candidate evidence candidate")
    _require(candidate.get("sha") == candidate_sha, "candidate evidence SHA mismatch")

    provenance = _mapping(candidate_evidence.get("provenance"), "candidate evidence provenance")
    _require(provenance.get("repository") == REPOSITORY, "candidate repository mismatch")
    _require(provenance.get("checkout_head_sha") == candidate_sha, "candidate checkout SHA mismatch")
    _require(
        provenance.get("environment_lock_sha256") == LOCK_INDEX_FILE_SHA256,
        "candidate provenance lock SHA mismatch",
    )

    checkpoint = _mapping(candidate_evidence.get("checkpoint"), "candidate evidence checkpoint")
    _require(
        checkpoint.get("environment_lock_sha256") == LOCK_INDEX_FILE_SHA256,
        "checkpoint lock SHA mismatch",
    )

    gate_candidate = _mapping(stage_gate_report.get("candidate"), "stage gate candidate")
    _require(gate_candidate.get("sha") == candidate_sha, "stage gate candidate SHA mismatch")
    summary = _mapping(stage_gate_report.get("summary"), "stage gate summary")
    authority = _mapping(stage_gate_report.get("promotion_authority"), "promotion authority")
    blockers = authority.get("blockers", [])
    _require(isinstance(blockers, list), "promotion authority blockers must be a list")

    counts = _mapping(summary.get("counts"), "stage gate counts")
    _require(set(counts) == {"PASS", "FAIL", "NOT_TESTED"}, "stage gate status set mismatch")
    for status, value in counts.items():
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"stage gate {status} count is invalid",
        )

    _require(promotion_report.get("candidate_sha") == candidate_sha, "promotion report SHA mismatch")
    cross_fields = {
        "evaluation_complete": summary.get("evaluation_complete"),
        "quality_overall_status": summary.get("overall_status"),
        "promotion_eligible": summary.get("promotion_eligible"),
        "promotion_authority_status": summary.get("promotion_authority_status"),
    }
    for field, expected in cross_fields.items():
        _require(promotion_report.get(field) == expected, f"promotion report {field} mismatch")
    _require(
        promotion_report.get("promotion_blockers") == blockers,
        "promotion blocker set diverges from stage gate authority",
    )

    evaluation_complete = summary.get("evaluation_complete")
    overall_status = summary.get("overall_status")
    promotion_eligible = summary.get("promotion_eligible")
    authority_status = summary.get("promotion_authority_status")
    _require(type(evaluation_complete) is bool, "evaluation_complete must be boolean")
    _require(type(promotion_eligible) is bool, "promotion_eligible must be boolean")
    _require(
        overall_status in {"PASS", "FAIL", "NOT_TESTED"},
        "invalid quality overall status",
    )
    _require(
        authority_status in {"PASS", "FAIL", "NOT_TESTED"},
        "invalid authority status",
    )
    _require(
        not promotion_eligible or (evaluation_complete and authority_status == "PASS"),
        "promotion eligibility is inconsistent with quality/authority",
    )

    if require_quality_pass:
        _require(evaluation_complete is True, "quality evaluation is incomplete")
        _require(overall_status == "PASS", "quality evaluation did not PASS")
        _require(
            dict(counts) == {"PASS": 15, "FAIL": 0, "NOT_TESTED": 0},
            "S0 quality gate counts are not exactly 15 PASS",
        )

    dataset = _mapping(candidate_evidence.get("dataset"), "candidate evidence dataset")
    tokenizer = _mapping(candidate_evidence.get("tokenizer"), "candidate evidence tokenizer")
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": "LOCAL_FREE_EXACT_CANDIDATE_EVALUATION_EVIDENCE_NOT_PROMOTION",
        "repository": REPOSITORY,
        "candidate_sha": candidate_sha,
        "locked_environment": runtime_binding,
        "reports": {name: {"sha256": hashes[name]} for name in _REPORT_NAMES},
        "identity": {
            "dataset_manifest_sha256": provenance.get("dataset_manifest_sha256"),
            "tokenizer_config_sha256": tokenizer.get("config_sha256"),
            "tokenizer_vocab_sha256": tokenizer.get("vocab_sha256"),
            "packing_sha256": provenance.get("packing_sha256"),
            "checkpoint_id": checkpoint.get("final_checkpoint_id"),
            "dataset_id": dataset.get("dataset_id"),
        },
        "summary": {
            "quality_overall_status": overall_status,
            "evaluation_complete": evaluation_complete,
            "quality_gate_counts": dict(counts),
            "promotion_authority_status": authority_status,
            "promotion_eligible": promotion_eligible,
        },
        "truth_boundary": (
            "bundle PASS proves exact-candidate D04 quality evidence ran under the bound D08 "
            "locked runtime; it does not create audit or promotion authority"
        ),
    }
    bundle["bundle_sha256"] = _canonical_hash(bundle)
    return bundle


def validate_s0_evaluation_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate the self-hash and stable outer identity of a materialized bundle."""

    _require(bundle.get("schema_version") == SCHEMA_VERSION, "wrong evaluation bundle schema")
    _require(bundle.get("repository") == REPOSITORY, "evaluation bundle repository mismatch")
    candidate_sha = bundle.get("candidate_sha")
    _require(
        isinstance(candidate_sha, str) and _GIT_SHA.fullmatch(candidate_sha) is not None,
        "evaluation bundle candidate SHA is invalid",
    )
    claimed_hash = bundle.get("bundle_sha256")
    _require(
        isinstance(claimed_hash, str) and _SHA256.fullmatch(claimed_hash) is not None,
        "evaluation bundle self-hash missing",
    )
    unhashed = dict(bundle)
    unhashed.pop("bundle_sha256", None)
    _require(_canonical_hash(unhashed) == claimed_hash, "evaluation bundle self-hash mismatch")
