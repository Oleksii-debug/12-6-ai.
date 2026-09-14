from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "12-6.the-stack-v2-provenance-feasibility.v1"
EXPECTED_DATASET_ID = "bigcode/the-stack-v2"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SPDXISH = re.compile(r"^[A-Za-z0-9.+-]+$")

REQUIRED_PROVENANCE_FIELDS = (
    "repo_name",
    "revision_id",
    "snapshot_id",
    "blob_id",
    "content_id",
    "path",
)

BLOCKING_RIGHTS_DECISIONS = {
    "UNKNOWN",
    "REVIEW_REQUIRED",
    "NOT_APPROVED",
    "RESTRICTED",
    "AMBIGUOUS",
}
APPROVED_RIGHTS_DECISION = "APPROVED_FOR_MODEL_TRAINING_BY_PROJECT_REVIEW"


class PolicyValidationError(ValueError):
    """Raised when the feasibility policy weakens its fail-closed authority boundary."""


@dataclass(frozen=True)
class FeasibilityDecision:
    status: str
    training_authorized: bool
    reasons: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return self.status == "SOURCE_FEASIBLE_FOR_D03_ADMISSION_REVIEW"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "training_authorized": self.training_authorized,
            "reasons": list(self.reasons),
        }


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{field} must be an object")
    return value


def _sequence_of_strings(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PolicyValidationError(f"{field} must be an array of strings")
    result = tuple(value)
    if not result or any(not _non_empty_string(item) for item in result):
        raise PolicyValidationError(f"{field} must be a non-empty array of strings")
    return result  # type: ignore[return-value]


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema") != SCHEMA:
        raise PolicyValidationError(f"schema must be {SCHEMA}")
    if policy.get("status") != "FEASIBILITY_ONLY_NOT_TRAINING_AUTHORIZED":
        raise PolicyValidationError("policy status must remain non-authorizing")

    upstream = _mapping(policy.get("upstream"), "upstream")
    if upstream.get("dataset_id") != EXPECTED_DATASET_ID:
        raise PolicyValidationError("unexpected The Stack v2 dataset identity")
    revision = upstream.get("dataset_revision")
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        raise PolicyValidationError(
            "upstream.dataset_revision must be an immutable 40-hex revision"
        )
    if upstream.get("dataset_license_label") != "other":
        raise PolicyValidationError("dataset-level license label must remain 'other'")
    if upstream.get("gated") is not True:
        raise PolicyValidationError("The Stack v2 gated-access fact must remain true")
    if upstream.get("bulk_access_requires_agreement") is not True:
        raise PolicyValidationError("bulk access must require an explicit agreement")
    parties = set(_sequence_of_strings(upstream.get("bulk_access_parties"), "bulk_access_parties"))
    if parties != {"Software Heritage", "Inria"}:
        raise PolicyValidationError("bulk access parties must be Software Heritage and Inria")
    if upstream.get("source_licenses_control") is not True:
        raise PolicyValidationError("source repository licenses must remain controlling")
    if upstream.get("removal_refresh_required") is not True:
        raise PolicyValidationError("validated-removal refresh requirement cannot be disabled")

    truth = _mapping(policy.get("truth_boundaries"), "truth_boundaries")
    required_false = (
        "canonical_training_authorized",
        "bulk_download_authorized",
        "dataset_license_grants_source_rights",
        "software_heritage_access_terms_grant_source_rights",
        "inferred_license_grants_source_rights",
    )
    for field in required_false:
        if truth.get(field) is not False:
            raise PolicyValidationError(f"truth_boundaries.{field} must remain false")
    required_true = (
        "source_license_review_required",
        "privacy_review_required",
        "evaluation_firewall_required",
        "decontamination_required",
    )
    for field in required_true:
        if truth.get(field) is not True:
            raise PolicyValidationError(f"truth_boundaries.{field} must remain true")

    required_fields = set(
        _sequence_of_strings(
            policy.get("required_source_record_fields"), "required_source_record_fields"
        )
    )
    mandatory = {
        "dataset_revision",
        *REQUIRED_PROVENANCE_FIELDS,
        "source_license_spdx",
        "source_license_review",
        "access_evidence",
        "privacy_review_status",
        "evaluation_firewall_status",
        "decontamination_status",
        "removal_sync_revision",
    }
    if not mandatory.issubset(required_fields):
        missing = sorted(mandatory - required_fields)
        raise PolicyValidationError(f"required source fields weakened: {missing}")

    admission = _mapping(policy.get("admission"), "admission")
    if admission.get("terminal_training_authority") != "NEVER_GRANTED_BY_THIS_POLICY":
        raise PolicyValidationError("this policy cannot be terminal training authority")
    if admission.get("approved_rights_decision") != APPROVED_RIGHTS_DECISION:
        raise PolicyValidationError("approved rights decision token drifted")

    agreement = admission.get("project_bulk_access_agreement_ref")
    if agreement is not None and not _non_empty_string(agreement):
        raise PolicyValidationError(
            "project bulk agreement ref must be null or a non-empty durable ref"
        )


def load_policy(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PolicyValidationError("policy JSON must be an object")
    validate_policy(data)
    return data


def _record_required_string(
    record: Mapping[str, Any], field: str, reasons: list[str]
) -> str | None:
    value = record.get(field)
    if not _non_empty_string(value):
        reasons.append(f"missing_or_empty:{field}")
        return None
    return str(value)


def evaluate_source_feasibility(
    policy: Mapping[str, Any], record: Mapping[str, Any]
) -> FeasibilityDecision:
    """Evaluate one source record without ever granting canonical training authority.

    A positive result means only that the record has enough project-reviewed rights,
    provenance, access, privacy, decontamination and evaluation-firewall evidence to
    proceed to the separate D03 admission authority. It never means that the source
    has been admitted to training.
    """

    validate_policy(policy)
    upstream = _mapping(policy["upstream"], "upstream")
    admission = _mapping(policy["admission"], "admission")
    reasons: list[str] = []

    expected_revision = str(upstream["dataset_revision"])
    dataset_revision = _record_required_string(record, "dataset_revision", reasons)
    if dataset_revision is not None and dataset_revision != expected_revision:
        reasons.append("dataset_revision_drift")

    for field in REQUIRED_PROVENANCE_FIELDS:
        _record_required_string(record, field, reasons)

    revision_id = record.get("revision_id")
    snapshot_id = record.get("snapshot_id")
    content_id = record.get("content_id")
    blob_id = record.get("blob_id")
    for field, value in (
        ("revision_id", revision_id),
        ("snapshot_id", snapshot_id),
        ("content_id", content_id),
        ("blob_id", blob_id),
    ):
        if _non_empty_string(value) and not _HEX40.fullmatch(str(value)):
            reasons.append(f"invalid_swh_hex_identity:{field}")

    source_license = _record_required_string(record, "source_license_spdx", reasons)
    if source_license is not None:
        if source_license.upper() in {"NOASSERTION", "NONE", "UNKNOWN"}:
            reasons.append("source_license_unknown")
        elif not _SPDXISH.fullmatch(source_license):
            reasons.append("source_license_not_spdxish")

    license_review = record.get("source_license_review")
    if not isinstance(license_review, Mapping):
        reasons.append("missing_or_invalid:source_license_review")
    else:
        review_status = license_review.get("decision")
        if review_status != admission["approved_rights_decision"]:
            if review_status in BLOCKING_RIGHTS_DECISIONS or not _non_empty_string(review_status):
                reasons.append("source_rights_not_approved")
            else:
                reasons.append("unrecognized_source_rights_decision")
        if not _non_empty_string(license_review.get("evidence_ref")):
            reasons.append("missing_source_license_evidence_ref")
        if license_review.get("basis") in {
            "DATASET_LICENSE",
            "SWH_ACCESS_TERMS",
            "DETECTED_LICENSE_ONLY",
            "GHA_LICENSE_ONLY",
        }:
            reasons.append("forbidden_blanket_or_inferred_license_basis")

    access = record.get("access_evidence")
    if not isinstance(access, Mapping):
        reasons.append("missing_or_invalid:access_evidence")
    else:
        access_mode = access.get("mode")
        agreement_ref = access.get("agreement_ref")
        configured_ref = admission.get("project_bulk_access_agreement_ref")
        if access_mode != "SWH_S3_BULK":
            reasons.append("content_access_mode_not_qualified")
        if configured_ref is None:
            reasons.append("project_bulk_access_agreement_not_recorded")
        elif agreement_ref != configured_ref:
            reasons.append("bulk_access_agreement_mismatch")
        if access.get("terms_acknowledged") is not True:
            reasons.append("software_heritage_terms_not_acknowledged")

    expected_statuses = {
        "privacy_review_status": "PASSED",
        "evaluation_firewall_status": "BOUND",
        "decontamination_status": "PASSED",
    }
    for field, expected in expected_statuses.items():
        value = record.get(field)
        if value != expected:
            reasons.append(f"{field}_not_{expected.lower()}")

    removal_sync = _record_required_string(record, "removal_sync_revision", reasons)
    if removal_sync is not None and removal_sync != expected_revision:
        reasons.append("validated_removal_sync_is_stale")

    if record.get("dataset_license_authorizes_source") is True:
        reasons.append("dataset_license_blanket_authorization_forbidden")
    if record.get("training_authorized") is True:
        reasons.append("record_cannot_self_authorize_training")

    if reasons:
        return FeasibilityDecision(
            status="BLOCKED_FAIL_CLOSED",
            training_authorized=False,
            reasons=tuple(sorted(set(reasons))),
        )
    return FeasibilityDecision(
        status="SOURCE_FEASIBLE_FOR_D03_ADMISSION_REVIEW",
        training_authorized=False,
        reasons=("separate_d03_training_admission_still_required",),
    )


def evaluate_records(
    policy: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validate_policy(policy)
    decisions = [evaluate_source_feasibility(policy, record) for record in records]
    return {
        "schema": "12-6.the-stack-v2-provenance-evaluation.v1",
        "dataset_revision": policy["upstream"]["dataset_revision"],
        "record_count": len(decisions),
        "feasible_for_d03_review": sum(decision.feasible for decision in decisions),
        "blocked": sum(not decision.feasible for decision in decisions),
        "canonical_training_authorized": False,
        "decisions": [decision.as_dict() for decision in decisions],
    }
