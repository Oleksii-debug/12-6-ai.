from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from twelve_six.integration.dependency_security import (
    DependencySecurityError,
    build_lock_sbom,
    build_security_evidence,
    unique_components,
    validate_security_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "1" * 40
GENERATED_AT = "2026-08-24T12:00:00Z"
NOW = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)


def _scan_records(sbom, *, unresolved: str | None = None, vulnerable: str | None = None):
    licenses = {}
    advisories = {}
    for component in unique_components(sbom):
        key = component["key"]
        licenses[key] = {
            "status": "UNRESOLVED" if key == unresolved else "DECLARED",
            "license_expression": None if key == unresolved else "TEST-DECLARED",
            "license_field_present": False,
            "license_field_sha256": None,
            "license_classifiers": [],
            "metadata_sha256": "2" * 64,
        }
        advisories[key] = {
            "status": "QUERIED",
            "response_sha256": "3" * 64,
            "vulnerabilities": (
                [{"id": "OSV-TEST-1", "aliases": [], "modified": None}]
                if key == vulnerable
                else []
            ),
        }
    return licenses, advisories


def _evidence(*, unresolved: str | None = None, vulnerable: str | None = None):
    sbom = build_lock_sbom(root=ROOT, source_sha=SOURCE_SHA)
    licenses, advisories = _scan_records(sbom, unresolved=unresolved, vulnerable=vulnerable)
    evidence = build_security_evidence(
        sbom=sbom,
        generated_at=GENERATED_AT,
        license_records=licenses,
        advisory_records=advisories,
        scan_sources={
            "osv": {
                "status": "SUCCESS",
                "endpoint": "https://api.osv.dev/v1/querybatch",
                "retrieved_at": GENERATED_AT,
                "batch_response_sha256": "4" * 64,
            },
            "pypi": {
                "status": "SUCCESS",
                "endpoint": "https://pypi.org/pypi",
                "retrieved_at": GENERATED_AT,
                "response_set_sha256": "5" * 64,
            },
        },
    )
    return sbom, evidence


def test_committed_locks_build_deterministic_source_bound_sbom() -> None:
    first = build_lock_sbom(root=ROOT, source_sha=SOURCE_SHA)
    second = build_lock_sbom(root=ROOT, source_sha=SOURCE_SHA)
    assert first == second
    assert first["sbom_sha256"] == second["sbom_sha256"]
    assert set(first["profiles"]) == {"linux-aarch64", "linux-x86_64"}
    union = unique_components(first)
    assert any(item["key"] == "torch==2.13.0" for item in union)
    assert all(item["profiles"] for item in union)


def test_clean_observations_validate_without_claiming_audit_authority() -> None:
    _, evidence = _evidence()
    validated = validate_security_evidence(
        root=ROOT,
        evidence=evidence,
        expected_source_sha=SOURCE_SHA,
        max_age_hours=24,
        now=NOW,
    )
    assert validated["status"] == "EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS"
    assert validated["truth_boundary"] == {
        "audit_verdict": False,
        "license_approval": False,
        "vulnerability_risk_acceptance": False,
        "promotion_authority": False,
    }


def test_unresolved_license_is_explicit_review_required() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha=SOURCE_SHA)
    key = unique_components(sbom)[0]["key"]
    _, evidence = _evidence(unresolved=key)
    assert evidence["status"] == "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
    validate_security_evidence(
        root=ROOT,
        evidence=evidence,
        expected_source_sha=SOURCE_SHA,
        now=NOW,
    )


def test_vulnerability_is_explicit_review_required() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha=SOURCE_SHA)
    key = unique_components(sbom)[0]["key"]
    _, evidence = _evidence(vulnerable=key)
    assert evidence["status"] == "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
    validate_security_evidence(
        root=ROOT,
        evidence=evidence,
        expected_source_sha=SOURCE_SHA,
        now=NOW,
    )


def test_tampered_evidence_is_rejected() -> None:
    _, evidence = _evidence()
    evidence["components"][0]["version"] = "999.0"
    with pytest.raises(DependencySecurityError, match="self-hash mismatch"):
        validate_security_evidence(
            root=ROOT,
            evidence=evidence,
            expected_source_sha=SOURCE_SHA,
            now=NOW,
        )


def test_wrong_source_sha_is_rejected() -> None:
    _, evidence = _evidence()
    with pytest.raises(DependencySecurityError, match="source SHA mismatch"):
        validate_security_evidence(
            root=ROOT,
            evidence=evidence,
            expected_source_sha="a" * 40,
            now=NOW,
        )


def test_stale_evidence_is_rejected() -> None:
    _, evidence = _evidence()
    with pytest.raises(DependencySecurityError, match="is stale"):
        validate_security_evidence(
            root=ROOT,
            evidence=evidence,
            expected_source_sha=SOURCE_SHA,
            max_age_hours=1,
            now=NOW,
        )


def test_missing_component_scan_is_rejected() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha=SOURCE_SHA)
    licenses, advisories = _scan_records(sbom)
    licenses.pop(next(iter(licenses)))
    with pytest.raises(DependencySecurityError, match="component set"):
        build_security_evidence(
            sbom=sbom,
            generated_at=GENERATED_AT,
            license_records=licenses,
            advisory_records=advisories,
            scan_sources={
                "osv": {"status": "SUCCESS"},
                "pypi": {"status": "SUCCESS"},
            },
        )


def test_incomplete_scan_source_is_rejected() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha=SOURCE_SHA)
    licenses, advisories = _scan_records(sbom)
    with pytest.raises(DependencySecurityError, match="source is incomplete"):
        build_security_evidence(
            sbom=sbom,
            generated_at=GENERATED_AT,
            license_records=licenses,
            advisory_records=advisories,
            scan_sources={
                "osv": {"status": "ERROR"},
                "pypi": {"status": "SUCCESS"},
            },
        )
