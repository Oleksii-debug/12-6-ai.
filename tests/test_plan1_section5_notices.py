"""Lock-bound third-party notice and license-review negative cases."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from twelve_six.integration.dependency_security import build_lock_sbom, unique_components
from twelve_six.integration.release_fixture import (
    FixtureIntegrityError, build_notice_inventory, verify_notice_inventory,
)

ROOT = Path(__file__).resolve().parents[1]


def _test_observations(sbom):
    return {
        item["key"]: {
            "status": "UNRESOLVED", "license_expression": None,
            "metadata_sha256": "a" * 64,
        }
        for item in unique_components(sbom)
    }


def test_notice_inventory_is_complete_deterministic_not_release_approval() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha="1" * 40)
    observations = _test_observations(sbom)
    first = build_notice_inventory(sbom, observations)
    assert first == build_notice_inventory(sbom, observations)
    assert len(first["notices"]) == len(unique_components(sbom))
    assert set(sbom["profiles"]) == {"linux-x86_64", "linux-aarch64", "windows-x86_64"}
    assert first["release_approved"] is False
    assert all(item["approval"] == "UNRESOLVED" for item in first["notices"])
    assert all(item["full_license_text_verified"] is False for item in first["notices"])
    verify_notice_inventory(sbom, first)


def test_unknown_license_and_missing_publisher_metadata_fail_closed() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha="1" * 40)
    observations = _test_observations(sbom)
    key = next(iter(observations))
    observations.pop(key)
    with pytest.raises(FixtureIntegrityError, match="coverage"):
        build_notice_inventory(sbom, observations)
    observations = _test_observations(sbom)
    observations[key]["metadata_sha256"] = "untrusted"
    with pytest.raises(FixtureIntegrityError, match="metadata digest"):
        build_notice_inventory(sbom, observations)


def test_notice_tampering_or_false_license_text_approval_rejected() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha="1" * 40)
    data = build_notice_inventory(sbom, _test_observations(sbom))
    wrong = copy.deepcopy(data)
    wrong["notices"][0]["component"] = "forged==0"
    with pytest.raises(FixtureIntegrityError, match="hash mismatch"):
        verify_notice_inventory(sbom, wrong)
    wrong = copy.deepcopy(data)
    wrong["notices"][0]["full_license_text_verified"] = True
    with pytest.raises(FixtureIntegrityError):
        verify_notice_inventory(sbom, wrong)


def test_no_automatic_permissive_approval() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha="1" * 40)
    observations = _test_observations(sbom)
    key = next(iter(observations))
    observations[key] = {
        "status": "DECLARED", "license_expression": "MIT",
        "metadata_sha256": "b" * 64,
    }
    notices = build_notice_inventory(sbom, observations)
    assert notices["release_approved"] is False
    assert next(item for item in notices["notices"] if item["component"] == key)["approval"] == (
        "CANDIDATE_PERMISSIVE_REVIEW_REQUIRED"
    )
    verify_notice_inventory(sbom, notices)
