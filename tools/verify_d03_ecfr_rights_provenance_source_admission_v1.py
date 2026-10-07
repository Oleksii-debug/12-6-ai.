#!/usr/bin/env python3
"""Fail-closed eCFR rights/provenance source-admission authority.

This package qualifies only a conservative source-admission policy over the
already-merged point-in-time eCFR request. It never grants corpus, tokenizer,
training, evaluation, loss-position, or final-test authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-ecfr-rights-provenance-source-admission.v1"
POLICY_PATH = Path("configs/data/d03_ecfr_rights_provenance_source_admission_v1.json")
EXPECTED_REQUEST_PATH = Path("configs/data/d03_ecfr_point_in_time_request_v1.json")
EXPECTED_MATERIALIZER_PATH = Path("tools/materialize_d03_ecfr_point_in_time.py")
REQUEST_PATH = EXPECTED_REQUEST_PATH
MATERIALIZER_PATH = EXPECTED_MATERIALIZER_PATH

EXPECTED_REQUEST_BLOB_SHA1 = "92fb471260020dffc3b9d53ca98581cf700b0916"
EXPECTED_MATERIALIZER_BLOB_SHA1 = "226fa5a089985451b49239e5e02b1fd059095fa7"
EXPECTED_REQUEST_IDENTITY = "d9955ff11713a358164513d912d39ba107f0073c75bed15be64afe2cc8c3a2e4"
EXPECTED_URL = "https://www.ecfr.gov/api/versioner/v1/full/2026-09-03/title-1.xml"
EXPECTED_SOURCE_ID = "en.us.ecfr.regulations"
EXPECTED_FAMILY_ID = "us.federal-regulations.ecfr"

ROOT_KEYS = {
    "schema_version",
    "execution_profile",
    "project_authority",
    "candidate_binding",
    "primary_public_evidence",
    "admission_policy",
    "truth_boundary",
}
AUTHORITY_KEYS = {
    "swarm_control_issue",
    "ownership_issue",
    "parent_issue",
    "source_request_pr",
    "terminal_execution_evidence_issue",
    "request_path",
    "request_git_blob_sha1",
    "materializer_path",
    "materializer_git_blob_sha1",
}
BINDING_KEYS = {
    "request_identity_sha256",
    "date",
    "title",
    "title_name",
    "url",
    "source_id",
    "family_id",
    "stratum",
}
EVIDENCE_KEYS = {"authority", "title", "url", "accessed_utc", "supported_fact"}
POLICY_KEYS = {
    "decision_class",
    "hosting_or_public_availability_sufficient_alone",
    "blanket_section_105_for_hosted_objects",
    "requires_exact_point_in_time_request",
    "requires_embedded_federal_regulatory_text",
    "external_ibr_content_source_admitted",
    "contractor_or_private_authorship_source_admitted",
    "transferred_copyright_content_source_admitted",
    "image_or_media_source_admitted",
    "ambiguous_provenance_source_admitted",
    "record_level_provenance_classification_required",
    "legal_conclusion_claimed",
    "payload_source_admission_executed",
    "admitted_payload_records",
    "admitted_payload_bytes",
}
TRUTH_KEYS = {
    "source_policy_review_complete",
    "payload_source_admission_executed",
    "canonical_capacity_credited",
    "family_credit_added",
    "training_authorized_bytes",
    "authorized_unique_loss_positions",
    "tokenizer_fit_authorized",
    "training_eligible",
    "evaluation_eligible",
    "optimizer_updates",
    "model_training_executed",
    "final_test_accessed",
    "paid_compute_used",
    "foreign_pretrained_weights",
    "external_llm_api_data_or_intelligence",
}
UNIT_KEYS = {
    "request_identity_sha256",
    "source_id",
    "family_id",
    "embedded_in_ecfr_xml",
    "federal_regulatory_text",
    "external_incorporated_by_reference",
    "contractor_or_private_authorship",
    "transferred_copyright_notice_or_status",
    "image_or_media_payload",
    "ambiguous_provenance",
}

EXPECTED_AUTHORITY = {
    "swarm_control_issue": 723,
    "ownership_issue": 1234,
    "parent_issue": 672,
    "source_request_pr": 707,
    "terminal_execution_evidence_issue": 1103,
    "request_path": str(EXPECTED_REQUEST_PATH),
    "request_git_blob_sha1": EXPECTED_REQUEST_BLOB_SHA1,
    "materializer_path": str(EXPECTED_MATERIALIZER_PATH),
    "materializer_git_blob_sha1": EXPECTED_MATERIALIZER_BLOB_SHA1,
}
EXPECTED_BINDING = {
    "request_identity_sha256": EXPECTED_REQUEST_IDENTITY,
    "date": "2026-09-03",
    "title": 1,
    "title_name": "General Provisions",
    "url": EXPECTED_URL,
    "source_id": EXPECTED_SOURCE_ID,
    "family_id": EXPECTED_FAMILY_ID,
    "stratum": "en",
}
EXPECTED_EVIDENCE = [
    {
        "authority": "Office of the Law Revision Counsel, U.S. House of Representatives",
        "title": "17 U.S.C. § 105 — Subject matter of copyright: United States Government works",
        "url": "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title17-section105&num=0&edition=prelim",
        "accessed_utc": "2026-09-11",
        "supported_fact": (
            "Section 105 states that copyright protection under Title 17 is not available "
            "for a work of the United States Government while allowing the Government to "
            "receive and hold transferred copyrights."
        ),
    },
    {
        "authority": "National Archives and Records Administration",
        "title": "Code of Federal Regulations",
        "url": "https://www.archives.gov/federal-register/cfr",
        "accessed_utc": "2026-09-11",
        "supported_fact": (
            "NARA describes the CFR as the codification of general and permanent rules "
            "published in the Federal Register by executive departments and agencies."
        ),
    },
    {
        "authority": "National Archives and Records Administration",
        "title": "Incorporation by Reference in the CFR",
        "url": "https://www.archives.gov/federal-register/cfr/ibr-locations",
        "accessed_utc": "2026-09-11",
        "supported_fact": (
            "NARA explains that incorporation by reference gives legal effect to material "
            "published elsewhere and its IBR locations page does not provide the text of "
            "the incorporated standards themselves."
        ),
    },
]
EXPECTED_POLICY = {
    "decision_class": "CONDITIONAL_US_FEDERAL_REGULATORY_TEXT_SOURCE_ADMISSION",
    "hosting_or_public_availability_sufficient_alone": False,
    "blanket_section_105_for_hosted_objects": False,
    "requires_exact_point_in_time_request": True,
    "requires_embedded_federal_regulatory_text": True,
    "external_ibr_content_source_admitted": False,
    "contractor_or_private_authorship_source_admitted": False,
    "transferred_copyright_content_source_admitted": False,
    "image_or_media_source_admitted": False,
    "ambiguous_provenance_source_admitted": False,
    "record_level_provenance_classification_required": True,
    "legal_conclusion_claimed": False,
    "payload_source_admission_executed": False,
    "admitted_payload_records": 0,
    "admitted_payload_bytes": 0,
}
EXPECTED_TRUTH = {
    "source_policy_review_complete": True,
    "payload_source_admission_executed": False,
    "canonical_capacity_credited": 0,
    "family_credit_added": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "tokenizer_fit_authorized": False,
    "training_eligible": False,
    "evaluation_eligible": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "final_test_accessed": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "external_llm_api_data_or_intelligence": False,
}
NUMERIC_ZERO_POLICY_KEYS = {"admitted_payload_records", "admitted_payload_bytes"}
NUMERIC_ZERO_TRUTH_KEYS = {
    "canonical_capacity_credited",
    "family_credit_added",
    "training_authorized_bytes",
    "authorized_unique_loss_positions",
    "optimizer_updates",
}


class AdmissionError(ValueError):
    """Raised when an authority or source-admission decision fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _require_exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == expected, f"{label} keys must be exact")
    return value


def _require_int(value: object, *, label: str, expected: int | None = None) -> int:
    _require(type(value) is int, f"{label} must be a strict integer")
    assert type(value) is int
    _require(value >= 0, f"{label} must be non-negative")
    if expected is not None:
        _require(value == expected, f"{label} mismatch")
    return value


def _require_bool(value: object, *, label: str, expected: bool | None = None) -> bool:
    _require(type(value) is bool, f"{label} must be boolean")
    assert type(value) is bool
    if expected is not None:
        _require(value is expected, f"{label} mismatch")
    return value


def _git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(prefix + raw).hexdigest()  # noqa: S324 - Git object identity, not crypto security


def _read_pinned_json(path: Path, expected_blob_sha1: str, label: str) -> dict[str, Any]:
    _require(path == EXPECTED_REQUEST_PATH, f"{label} path substitution")
    raw = path.read_bytes()
    _require(_git_blob_sha1(raw) == expected_blob_sha1, f"{label} Git blob mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdmissionError(f"{label} is not valid JSON") from exc
    _require(type(payload) is dict, f"{label} must be a JSON object")
    assert isinstance(payload, dict)
    return payload


def _validate_incumbent_authority() -> None:
    request = _read_pinned_json(REQUEST_PATH, EXPECTED_REQUEST_BLOB_SHA1, "request authority")
    _require(
        request.get("request_identity_sha256") == EXPECTED_REQUEST_IDENTITY,
        "request identity drift",
    )
    selection = _require_exact_keys(
        request.get("selection"),
        {"date", "title", "title_name", "url"},
        "request selection",
    )
    _require(selection.get("date") == "2026-09-03", "request date drift")
    _require_int(selection.get("title"), label="request title", expected=1)
    _require(selection.get("title_name") == "General Provisions", "request title name drift")
    _require(selection.get("url") == EXPECTED_URL, "request URL drift")

    authority = request.get("authority")
    _require(type(authority) is dict, "request authority block missing")
    assert isinstance(authority, dict)
    _require(authority.get("source_id") == EXPECTED_SOURCE_ID, "request source drift")
    _require(authority.get("family_id") == EXPECTED_FAMILY_ID, "request family drift")
    _require(authority.get("stratum") == "en", "request stratum drift")

    claims = request.get("claims")
    _require(type(claims) is dict, "request claims block missing")
    assert isinstance(claims, dict)
    for key in (
        "authorized_unique_loss_positions",
        "canonical_capacity_credit_bytes",
        "family_credit",
        "optimizer_updates",
        "training_authorized_bytes",
    ):
        _require_int(claims.get(key), label=f"request claims.{key}", expected=0)
    for key in (
        "final_test_accessed",
        "learned_20m_claim",
        "model_training_executed",
        "paid_compute_authorized",
        "research_corpus_v1_released",
        "tokenizer_fit_authorized",
    ):
        _require_bool(claims.get(key), label=f"request claims.{key}", expected=False)

    _require(MATERIALIZER_PATH == EXPECTED_MATERIALIZER_PATH, "materializer path substitution")
    materializer_raw = MATERIALIZER_PATH.read_bytes()
    _require(
        _git_blob_sha1(materializer_raw) == EXPECTED_MATERIALIZER_BLOB_SHA1,
        "materializer Git blob mismatch",
    )


def validate_policy(policy: dict[str, Any], *, verify_incumbent: bool = True) -> dict[str, Any]:
    root = _require_exact_keys(policy, ROOT_KEYS, "policy root")
    _require(root["schema_version"] == SCHEMA, "unexpected schema")
    _require(root["execution_profile"] == "LOCAL_FREE", "execution profile drift")

    authority = _require_exact_keys(root["project_authority"], AUTHORITY_KEYS, "project_authority")
    for key in (
        "swarm_control_issue",
        "ownership_issue",
        "parent_issue",
        "source_request_pr",
        "terminal_execution_evidence_issue",
    ):
        _require_int(authority[key], label=f"project_authority.{key}", expected=EXPECTED_AUTHORITY[key])
    for key in (
        "request_path",
        "request_git_blob_sha1",
        "materializer_path",
        "materializer_git_blob_sha1",
    ):
        _require(authority[key] == EXPECTED_AUTHORITY[key], f"project_authority.{key} drift")

    binding = _require_exact_keys(root["candidate_binding"], BINDING_KEYS, "candidate_binding")
    for key, expected in EXPECTED_BINDING.items():
        if key == "title":
            _require_int(binding[key], label="candidate_binding.title", expected=expected)
        else:
            _require(binding[key] == expected, f"candidate_binding.{key} drift")

    evidence = root["primary_public_evidence"]
    _require(type(evidence) is list, "primary_public_evidence must be a list")
    _require(len(evidence) == len(EXPECTED_EVIDENCE), "primary evidence count drift")
    for index, (actual, expected) in enumerate(zip(evidence, EXPECTED_EVIDENCE, strict=True)):
        item = _require_exact_keys(actual, EVIDENCE_KEYS, f"primary_public_evidence[{index}]")
        _require(item == expected, f"primary_public_evidence[{index}] drift")

    admission = _require_exact_keys(root["admission_policy"], POLICY_KEYS, "admission_policy")
    for key, expected in EXPECTED_POLICY.items():
        if key in NUMERIC_ZERO_POLICY_KEYS:
            _require_int(admission[key], label=f"admission_policy.{key}", expected=0)
        elif type(expected) is bool:
            _require_bool(admission[key], label=f"admission_policy.{key}", expected=expected)
        else:
            _require(admission[key] == expected, f"admission_policy.{key} drift")

    truth = _require_exact_keys(root["truth_boundary"], TRUTH_KEYS, "truth_boundary")
    for key, expected in EXPECTED_TRUTH.items():
        if key in NUMERIC_ZERO_TRUTH_KEYS:
            _require_int(truth[key], label=f"truth_boundary.{key}", expected=0)
        elif type(expected) is bool:
            _require_bool(truth[key], label=f"truth_boundary.{key}", expected=expected)
        else:
            _require(truth[key] == expected, f"truth_boundary.{key} drift")

    _require(
        admission["payload_source_admission_executed"] is truth["payload_source_admission_executed"],
        "payload execution truth mismatch",
    )
    if verify_incumbent:
        _validate_incumbent_authority()
    return policy


def load_policy(path: Path = POLICY_PATH, *, verify_incumbent: bool = True) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdmissionError("policy is not valid JSON") from exc
    _require(type(payload) is dict, "policy must be a JSON object")
    assert isinstance(payload, dict)
    return validate_policy(payload, verify_incumbent=verify_incumbent)


def assess_unit(unit: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    validate_policy(policy)
    item = _require_exact_keys(unit, UNIT_KEYS, "unit")
    _require(
        item["request_identity_sha256"] == EXPECTED_REQUEST_IDENTITY,
        "unit request identity drift",
    )
    _require(item["source_id"] == EXPECTED_SOURCE_ID, "unit source drift")
    _require(item["family_id"] == EXPECTED_FAMILY_ID, "unit family drift")
    for key in UNIT_KEYS - {"request_identity_sha256", "source_id", "family_id"}:
        _require_bool(item[key], label=f"unit.{key}")

    rejection_order = (
        ("external_incorporated_by_reference", "external_incorporated_by_reference"),
        ("contractor_or_private_authorship", "contractor_or_private_authorship"),
        ("transferred_copyright_notice_or_status", "transferred_copyright_notice_or_status"),
        ("image_or_media_payload", "image_or_media_payload"),
        ("ambiguous_provenance", "ambiguous_provenance"),
    )
    for key, reason in rejection_order:
        if item[key]:
            return {"decision": "NOT_SOURCE_ADMITTED", "reason": reason}

    if not item["embedded_in_ecfr_xml"]:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "not_embedded_in_ecfr_xml"}
    if not item["federal_regulatory_text"]:
        return {"decision": "HOLD_PROVENANCE_REVIEW", "reason": "not_attributed_federal_regulatory_text"}

    return {
        "decision": "CONDITIONAL_SOURCE_ADMISSION",
        "reason": "embedded_attributed_federal_regulatory_text",
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "training_eligible": False,
        "legal_conclusion_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    print(
        json.dumps(
            {
                "schema_version": policy["schema_version"],
                "status": "SOURCE_POLICY_READY_ZERO_CREDIT",
                "request_identity_sha256": EXPECTED_REQUEST_IDENTITY,
                "training_authorized_bytes": 0,
                "authorized_unique_loss_positions": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
