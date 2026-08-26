from __future__ import annotations

import hashlib

import pytest

from twelve_six.verification.cluster_safe_split_v03 import (
    BoundaryRecord,
    SplitSafetyEvidenceError,
    adversarial_self_test,
    audit_records,
    compare_clean_roots,
)


def _record(
    boundary: str,
    record_id: str,
    source_id: str,
    source_family: str,
    content: str,
    cluster_id: str,
) -> BoundaryRecord:
    return BoundaryRecord(
        boundary=boundary,
        record_id=record_id,
        source_id=source_id,
        source_family=source_family,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        cluster_id=cluster_id,
        size_bytes=len(content.encode()),
    )


def _clean() -> list[BoundaryRecord]:
    return [
        _record("train", "t1", "ua-a", "ua.family.a", "alpha", "c1"),
        _record("train", "t2", "en-a", "en.family.a", "beta", "c2"),
        _record("validation", "v1", "ua-b", "ua.family.b", "gamma", "c3"),
        _record("reserved", "r1", "eval", "reserved.eval", "delta", "c4"),
    ]


def test_adversarial_self_test_passes_all_cases() -> None:
    report = adversarial_self_test()
    assert report["verdict"] == "PASS_ADVERSARIAL_FIXTURES"
    assert report["fixture_count"] == 6
    assert all(report["checks"].values())


def test_exact_and_near_duplicate_crossings_fail() -> None:
    records = _clean()
    records.append(
        BoundaryRecord(
            boundary="validation",
            record_id="v-exact",
            source_id="en-b",
            source_family="en.family.b",
            content_sha256=records[0].content_sha256,
            cluster_id=records[0].cluster_id,
            size_bytes=records[0].size_bytes,
        )
    )
    report = audit_records(records)
    assert report["verdict"] == "FAIL_SPLIT_SAFETY"
    assert report["exact_content_cross_boundary_count"] == 1
    assert report["cluster_cross_boundary_count"] == 1


def test_record_identity_reuse_fails() -> None:
    records = _clean()
    records.append(_record("reserved", "t1", "eval-2", "reserved.eval", "unique", "c9"))
    report = audit_records(records)
    assert report["verdict"] == "FAIL_SPLIT_SAFETY"
    assert report["record_identity_cross_boundary_count"] == 1


def test_assignment_and_family_distribution_are_input_order_deterministic() -> None:
    first = audit_records(_clean())
    second = audit_records(list(reversed(_clean())))
    assert first["canonical_assignment_sha256"] == second["canonical_assignment_sha256"]
    assert (
        first["source_family_distribution_sha256"]
        == second["source_family_distribution_sha256"]
    )


def test_clean_root_comparison_detects_assignment_drift() -> None:
    first = audit_records(_clean())
    second = dict(first)
    assert compare_clean_roots(first, second)["verdict"] == "PASS_DETERMINISTIC_ASSIGNMENT"
    second["canonical_assignment_sha256"] = "0" * 64
    comparison = compare_clean_roots(first, second)
    assert comparison["verdict"] == "FAIL_NONDETERMINISTIC_ASSIGNMENT"
    assert comparison["mismatched_fields"] == ["canonical_assignment_sha256"]


def test_mapping_fails_closed_without_family_or_cluster_evidence() -> None:
    content_sha = hashlib.sha256(b"text").hexdigest()
    with pytest.raises(SplitSafetyEvidenceError, match="source_family"):
        BoundaryRecord.from_mapping(
            "train",
            {
                "id": "missing-family",
                "source_id": "source",
                "content_sha256": content_sha,
                "near_duplicate_cluster_id": "cluster",
            },
        )
    with pytest.raises(SplitSafetyEvidenceError, match="near_duplicate_cluster_id"):
        BoundaryRecord.from_mapping(
            "train",
            {
                "id": "missing-cluster",
                "source_id": "source",
                "source_family": "family",
                "content_sha256": content_sha,
            },
        )
