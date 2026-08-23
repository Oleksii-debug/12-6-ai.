from __future__ import annotations

import hashlib

import pytest

from twelve_six.data.external_sources import (
    RIGHTS_APPROVED,
    RIGHTS_REVIEW_REQUIRED,
    ExternalDataContractError,
    ExternalSourceSpec,
    ReservedSetSpec,
    RightsDecision,
    SnapshotSpec,
    build_external_source_registry,
    build_reserved_fingerprint_registry,
    contamination_report,
    validate_external_source_registry,
    validate_reserved_fingerprint_registry,
    verify_local_snapshot,
)
from twelve_six.data.scalable_ingestion import DataTroveParquetPlan, ScalableIngestionError


def _rights(status: str = RIGHTS_APPROVED, *, license_id: str = "CC-BY-4.0") -> RightsDecision:
    return RightsDecision(
        status=status,
        license_id=license_id,
        terms_url="https://example.invalid/terms",
        allows_model_training=status == RIGHTS_APPROVED,
        allows_derivatives=True,
        allows_redistribution=True,
        policy_ref="policy://data-rights/review-1",
        reviewed_at="2026-08-23T00:00:00Z",
        reviewer_ref="role://data-rights-owner",
    )


def _snapshot(payload: bytes = b"immutable fixture") -> SnapshotSpec:
    return SnapshotSpec(
        uri="hf://datasets/example/source/snapshots/v1/source.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        retrieved_at="2026-08-23T00:00:00Z",
        upstream_version="v1",
        retrieval_method="manual_fixture",
    )


def _source(
    source_id: str = "source-a",
    *,
    source_version: str = "v1",
    rights: RightsDecision | None = None,
) -> ExternalSourceSpec:
    return ExternalSourceSpec(
        source_id=source_id,
        source_version=source_version,
        provider="example-provider",
        source_url="https://example.invalid/source",
        source_kind="jsonl",
        purpose="pretraining",
        synthetic=False,
        benchmark_material=False,
        held_out=False,
        snapshot=_snapshot(),
        rights=rights or _rights(),
    )


def test_training_requires_explicit_rights_approval() -> None:
    source = _source(rights=_rights(RIGHTS_REVIEW_REQUIRED))
    with pytest.raises(ExternalDataContractError, match="rights are not approved"):
        source.assert_training_eligible()


def test_noassertion_cannot_be_approved_for_training() -> None:
    with pytest.raises(ExternalDataContractError, match="NOASSERTION"):
        _rights(RIGHTS_APPROVED, license_id="NOASSERTION")


def test_registry_identity_is_order_independent_across_versions() -> None:
    a1 = _source("a", source_version="v1")
    a2 = _source("a", source_version="v2")
    b1 = _source("b", source_version="v1")
    first = build_external_source_registry([a2, b1, a1])
    second = build_external_source_registry([a1, a2, b1])
    assert first == second
    assert validate_external_source_registry(first) == (a1, a2, b1)
    assert len(first["registry_identity_sha256"]) == 64


def test_registry_identity_tamper_fails_closed() -> None:
    registry = build_external_source_registry([_source()])
    registry["registry_identity_sha256"] = "0" * 64
    with pytest.raises(ExternalDataContractError, match="identity mismatch"):
        validate_external_source_registry(registry)


def test_reserved_registry_identity_tamper_fails_closed() -> None:
    registry = build_reserved_fingerprint_registry([])
    registry["registry_identity_sha256"] = "0" * 64
    with pytest.raises(ExternalDataContractError, match="identity mismatch"):
        validate_reserved_fingerprint_registry(registry)


def test_snapshot_uri_rejects_secret_or_unstable_query() -> None:
    with pytest.raises(ExternalDataContractError, match="credentials/query"):
        SnapshotSpec(
            uri="s3://bucket/raw.jsonl?token=secret",
            sha256="0" * 64,
            size_bytes=1,
            retrieved_at="2026-08-23T00:00:00Z",
            upstream_version="v1",
            retrieval_method="test",
        )


def test_local_snapshot_verification_is_fail_closed(tmp_path) -> None:
    payload = b"immutable fixture"
    path = tmp_path / "snapshot.bin"
    path.write_bytes(payload)
    verify_local_snapshot(_snapshot(payload), path)
    path.write_bytes(payload + b"tampered")
    with pytest.raises(ExternalDataContractError, match="size mismatch"):
        verify_local_snapshot(_snapshot(payload), path)


def test_reserved_registry_detects_source_and_content_overlap() -> None:
    blocked = hashlib.sha256(b"reserved").hexdigest()
    reserved = build_reserved_fingerprint_registry(
        [
            ReservedSetSpec(
                set_id="eval-a",
                version="v1",
                source_id="benchmark-source",
                purpose="benchmark",
                normalized_sha256=(blocked,),
            )
        ]
    )
    report = contamination_report(
        [
            {"id": "one", "source_id": "benchmark-source", "content_sha256": "1" * 64},
            {"id": "two", "source_id": "train-source", "content_sha256": blocked},
        ],
        reserved,
    )
    assert report["source_id_overlap_count"] == 1
    assert report["content_sha256_overlap_count"] == 1
    assert report["source_id_overlap_record_ids"] == ["one"]
    assert report["content_sha256_overlap_record_ids"] == ["two"]


def test_datatrove_plan_is_deterministic_and_pinned() -> None:
    plan = DataTroveParquetPlan(
        source_id="source-a",
        source_version="v1",
        snapshot_sha256="1" * 64,
        registry_identity_sha256="2" * 64,
        input_uri="hf://datasets/example/raw/v1",
        input_format="jsonl",
        output_uri="hf://buckets/example/staged/v1",
        logging_uri="hf://buckets/example/logs/v1",
        tasks=8,
        workers=4,
    )
    assert plan.manifest() == plan.manifest()
    assert plan.manifest()["datatrove_version"] == "0.10.0"
    assert len(plan.manifest()["plan_sha256"]) == 64


def test_datatrove_plan_rejects_workers_above_tasks() -> None:
    with pytest.raises(ScalableIngestionError, match="workers cannot exceed tasks"):
        DataTroveParquetPlan(
            source_id="source-a",
            source_version="v1",
            snapshot_sha256="1" * 64,
            registry_identity_sha256="2" * 64,
            input_uri="input",
            input_format="jsonl",
            output_uri="output",
            logging_uri="logs",
            tasks=1,
            workers=2,
        )
