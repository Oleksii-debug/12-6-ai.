import copy
import json
from pathlib import Path

import pytest

from twelve_six.the_stack_v2_policy import (
    PolicyValidationError,
    evaluate_records,
    evaluate_source_feasibility,
    validate_policy,
)

POLICY_PATH = Path("configs/research/the_stack_v2_provenance_feasibility_v1.json")


def _policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _qualified_policy():
    policy = _policy()
    policy["admission"]["project_bulk_access_agreement_ref"] = (
        "evidence/legal/swh-inria-agreement-001"
    )
    return policy


def _record():
    revision = "e565caa3a78c2423bd374333a472b049eb090e47"
    return {
        "dataset_revision": revision,
        "repo_name": "example/project",
        "revision_id": "1" * 40,
        "snapshot_id": "2" * 40,
        "blob_id": "3" * 40,
        "content_id": "4" * 40,
        "path": "src/example.py",
        "source_license_spdx": "MIT",
        "source_license_review": {
            "decision": "APPROVED_FOR_MODEL_TRAINING_BY_PROJECT_REVIEW",
            "basis": "HUMAN_REVIEWED_SOURCE_LICENSE",
            "evidence_ref": "evidence/legal/example-project-license-review.json",
        },
        "access_evidence": {
            "mode": "SWH_S3_BULK",
            "agreement_ref": "evidence/legal/swh-inria-agreement-001",
            "terms_acknowledged": True,
        },
        "privacy_review_status": "PASSED",
        "evaluation_firewall_status": "BOUND",
        "decontamination_status": "PASSED",
        "removal_sync_revision": revision,
    }


def test_canonical_policy_is_fail_closed_and_currently_not_bulk_qualified():
    policy = _policy()
    validate_policy(policy)
    assert policy["truth_boundaries"]["canonical_training_authorized"] is False
    assert policy["truth_boundaries"]["bulk_download_authorized"] is False
    assert policy["admission"]["project_bulk_access_agreement_ref"] is None


def test_complete_reviewed_record_only_reaches_separate_d03_review():
    decision = evaluate_source_feasibility(_qualified_policy(), _record())
    assert decision.status == "SOURCE_FEASIBLE_FOR_D03_ADMISSION_REVIEW"
    assert decision.feasible is True
    assert decision.training_authorized is False
    assert decision.reasons == ("separate_d03_training_admission_still_required",)


def test_live_canonical_policy_blocks_content_until_bulk_agreement_is_recorded():
    decision = evaluate_source_feasibility(_policy(), _record())
    assert decision.status == "BLOCKED_FAIL_CLOSED"
    assert "project_bulk_access_agreement_not_recorded" in decision.reasons
    assert decision.training_authorized is False


@pytest.mark.parametrize(
    "field",
    ["repo_name", "revision_id", "snapshot_id", "blob_id", "content_id", "path"],
)
def test_missing_provenance_identity_fails_closed(field):
    record = _record()
    record.pop(field)
    decision = evaluate_source_feasibility(_qualified_policy(), record)
    assert decision.status == "BLOCKED_FAIL_CLOSED"
    assert any(field in reason for reason in decision.reasons)


def test_dataset_license_cannot_substitute_for_source_license_review():
    record = _record()
    record["source_license_review"] = {
        "decision": "APPROVED_FOR_MODEL_TRAINING_BY_PROJECT_REVIEW",
        "basis": "DATASET_LICENSE",
        "evidence_ref": "dataset-card",
    }
    record["dataset_license_authorizes_source"] = True
    decision = evaluate_source_feasibility(_qualified_policy(), record)
    assert "forbidden_blanket_or_inferred_license_basis" in decision.reasons
    assert "dataset_license_blanket_authorization_forbidden" in decision.reasons
    assert decision.training_authorized is False


@pytest.mark.parametrize("basis", ["SWH_ACCESS_TERMS", "DETECTED_LICENSE_ONLY", "GHA_LICENSE_ONLY"])
def test_access_or_detected_license_signal_cannot_be_source_rights_authority(basis):
    record = _record()
    record["source_license_review"]["basis"] = basis
    decision = evaluate_source_feasibility(_qualified_policy(), record)
    assert "forbidden_blanket_or_inferred_license_basis" in decision.reasons


@pytest.mark.parametrize("license_id", ["NOASSERTION", "NONE", "UNKNOWN", ""])
def test_unknown_or_missing_source_license_fails_closed(license_id):
    record = _record()
    record["source_license_spdx"] = license_id
    decision = evaluate_source_feasibility(_qualified_policy(), record)
    assert decision.status == "BLOCKED_FAIL_CLOSED"


def test_dataset_revision_and_removal_sync_must_match_immutable_upstream():
    record = _record()
    record["dataset_revision"] = "a" * 40
    record["removal_sync_revision"] = "a" * 40
    decision = evaluate_source_feasibility(_qualified_policy(), record)
    assert "dataset_revision_drift" in decision.reasons
    assert "validated_removal_sync_is_stale" in decision.reasons


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("privacy_review_status", "UNKNOWN"),
        ("evaluation_firewall_status", "UNBOUND"),
        ("decontamination_status", "NOT_RUN"),
    ],
)
def test_privacy_evaluation_and_decontamination_are_independent_fail_closed_gates(field, value):
    record = _record()
    record[field] = value
    decision = evaluate_source_feasibility(_qualified_policy(), record)
    assert decision.status == "BLOCKED_FAIL_CLOSED"
    assert any(field in reason for reason in decision.reasons)


def test_record_cannot_self_declare_training_authority():
    record = _record()
    record["training_authorized"] = True
    decision = evaluate_source_feasibility(_qualified_policy(), record)
    assert "record_cannot_self_authorize_training" in decision.reasons
    assert decision.training_authorized is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("truth_boundaries", "canonical_training_authorized"), True),
        (("truth_boundaries", "bulk_download_authorized"), True),
        (("truth_boundaries", "dataset_license_grants_source_rights"), True),
        (("truth_boundaries", "software_heritage_access_terms_grant_source_rights"), True),
        (("upstream", "source_licenses_control"), False),
        (("upstream", "bulk_access_requires_agreement"), False),
    ],
)
def test_policy_truth_boundary_weakening_is_rejected(path, value):
    policy = copy.deepcopy(_policy())
    policy[path[0]][path[1]] = value
    with pytest.raises(PolicyValidationError):
        validate_policy(policy)


def test_required_source_fields_cannot_be_silently_removed():
    policy = _policy()
    policy["required_source_record_fields"].remove("source_license_review")
    with pytest.raises(PolicyValidationError, match="weakened"):
        validate_policy(policy)


def test_batch_evaluation_never_emits_canonical_training_authority():
    result = evaluate_records(_qualified_policy(), [_record(), {"repo_name": "broken"}])
    assert result["record_count"] == 2
    assert result["feasible_for_d03_review"] == 1
    assert result["blocked"] == 1
    assert result["canonical_training_authorized"] is False
    assert all(decision["training_authorized"] is False for decision in result["decisions"])
