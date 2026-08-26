from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from twelve_six.data.external_sources import (
    RIGHTS_APPROVED,
    RIGHTS_REVIEW_REQUIRED,
    ExternalSourceSpec,
    RightsDecision,
    SnapshotSpec,
    build_external_source_registry,
)
from twelve_six.data.source_acquisition import (
    RetrievalCheckpoint,
    RetrievedChunk,
    SourceAcquisitionError,
    SourceRetrievalPlan,
    assert_checkpoint_compatible,
    build_local_checkpoint,
    build_retrieval_inventory,
    plan_from_registered_source,
    verify_and_stage_local_mirror,
)


def _registry(payload: bytes, *, rights_status: str = RIGHTS_REVIEW_REQUIRED):
    snapshot = SnapshotSpec(
        uri="https://example.invalid/snapshots/source-a/v1/source.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        retrieved_at="2026-08-24T00:00:00Z",
        upstream_version="v1",
        retrieval_method="synthetic_local_fixture",
    )
    rights = RightsDecision(
        status=rights_status,
        license_id="CC-BY-4.0",
        terms_url="https://example.invalid/terms",
        allows_model_training=rights_status == RIGHTS_APPROVED,
        allows_derivatives=True,
        allows_redistribution=True,
        policy_ref="policy://rights/source-a-v1",
        reviewed_at="2026-08-24T00:00:00Z",
        reviewer_ref="role://data-rights-owner",
    )
    source = ExternalSourceSpec(
        source_id="source-a",
        source_version="v1",
        provider="example-provider",
        source_url="https://example.invalid/source-a",
        source_kind="binary_fixture",
        purpose="pretraining",
        synthetic=True,
        benchmark_material=False,
        held_out=False,
        snapshot=snapshot,
        rights=rights,
    )
    return build_external_source_registry([source])


def _plan(
    tmp_path: Path,
    payload: bytes,
    *,
    rights_status: str = RIGHTS_REVIEW_REQUIRED,
    chunk_size_bytes: int = 4,
) -> SourceRetrievalPlan:
    return plan_from_registered_source(
        _registry(payload, rights_status=rights_status),
        "source-a",
        "v1",
        (tmp_path / "staged.bin").resolve().as_uri(),
        chunk_size_bytes=chunk_size_bytes,
    )


def test_plan_observes_rights_without_granting_training_approval(tmp_path: Path) -> None:
    plan = _plan(tmp_path, b"abcdefgh", rights_status=RIGHTS_REVIEW_REQUIRED)
    manifest = plan.manifest()
    assert manifest["rights_status_observed"] == RIGHTS_REVIEW_REQUIRED
    assert manifest["training_eligibility_evaluated"] is False
    assert manifest["rights_semantics"] == "OBSERVED_ONLY_NOT_APPROVAL"


def test_plan_requires_exact_registered_source_version(tmp_path: Path) -> None:
    with pytest.raises(SourceAcquisitionError, match="not found"):
        plan_from_registered_source(
            _registry(b"x"),
            "source-a",
            "v2",
            (tmp_path / "staged.bin").resolve().as_uri(),
        )


def test_plan_rejects_unstable_source_uri_and_uri_alias() -> None:
    common = {
        "source_registry_identity_sha256": "1" * 64,
        "source_id": "source-a",
        "source_version": "v1",
        "expected_sha256": "2" * 64,
        "expected_size_bytes": 1,
        "upstream_version": "v1",
        "retrieval_method": "fixture",
        "rights_status_observed": RIGHTS_REVIEW_REQUIRED,
    }
    with pytest.raises(SourceAcquisitionError, match="credentials/query"):
        SourceRetrievalPlan(
            **common,
            source_uri="https://example.invalid/a?token=secret",
            destination_uri="file:///tmp/b",
        )
    with pytest.raises(SourceAcquisitionError, match="must differ"):
        SourceRetrievalPlan(
            **common,
            source_uri="file:///tmp/a",
            destination_uri="file:///tmp/a",
        )


def test_exact_local_stage_publishes_only_verified_bytes(tmp_path: Path) -> None:
    payload = b"abcdefghij"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(payload)
    plan = _plan(tmp_path, payload, chunk_size_bytes=4)
    receipt = verify_and_stage_local_mirror(plan, source_path)
    assert (tmp_path / "staged.bin").read_bytes() == payload
    assert receipt.verified_sha256 == hashlib.sha256(payload).hexdigest()
    assert receipt.chunk_count == 3
    assert receipt.manifest() == receipt.manifest()
    assert receipt.manifest()["training_eligibility_evaluated"] is False


def test_hash_mismatch_does_not_publish_final_destination(tmp_path: Path) -> None:
    expected = b"abcdefgh"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"abcdEfgh")
    plan = _plan(tmp_path, expected, chunk_size_bytes=4)
    with pytest.raises(SourceAcquisitionError, match="SHA-256 mismatch"):
        verify_and_stage_local_mirror(plan, source_path)
    assert not (tmp_path / "staged.bin").exists()
    assert (tmp_path / "staged.bin.partial").exists()


def test_size_overflow_does_not_publish_final_destination(tmp_path: Path) -> None:
    expected = b"abcdefgh"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(expected + b"x")
    plan = _plan(tmp_path, expected, chunk_size_bytes=4)
    with pytest.raises(SourceAcquisitionError, match="exceeds expected"):
        verify_and_stage_local_mirror(plan, source_path)
    assert not (tmp_path / "staged.bin").exists()


def test_checkpoint_is_contiguous_and_bound_to_exact_plan(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    plan = _plan(tmp_path, payload, chunk_size_bytes=4)
    partial = tmp_path / "staged.bin.partial"
    partial.write_bytes(payload[:4])
    checkpoint = build_local_checkpoint(plan, partial)
    assert checkpoint.next_offset_bytes == 4
    assert_checkpoint_compatible(plan, checkpoint)
    other_plan = SourceRetrievalPlan(
        **{
            **plan.__dict__,
            "destination_uri": (tmp_path / "other.bin").resolve().as_uri(),
        }
    )
    with pytest.raises(SourceAcquisitionError, match="different plan"):
        assert_checkpoint_compatible(other_plan, checkpoint)


def test_checkpoint_rejects_index_gap_and_partial_chunk_boundary() -> None:
    with pytest.raises(SourceAcquisitionError, match="indexes must be contiguous"):
        RetrievalCheckpoint(
            "1" * 64,
            8,
            4,
            (RetrievedChunk(1, 0, 4, "2" * 64),),
        )
    with pytest.raises(SourceAcquisitionError, match="full chunk boundary"):
        RetrievalCheckpoint(
            "1" * 64,
            8,
            4,
            (RetrievedChunk(0, 0, 3, "2" * 64),),
        )


def test_resume_from_verified_partial_reaches_exact_final_bytes(tmp_path: Path) -> None:
    payload = b"abcdefghijkl"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(payload)
    plan = _plan(tmp_path, payload, chunk_size_bytes=4)
    (tmp_path / "staged.bin.partial").write_bytes(payload[:8])
    receipt = verify_and_stage_local_mirror(plan, source_path, resume=True)
    assert receipt.chunk_count == 3
    assert (tmp_path / "staged.bin").read_bytes() == payload


def test_resume_rejects_tampered_partial_prefix(tmp_path: Path) -> None:
    payload = b"abcdefghijkl"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(payload)
    plan = _plan(tmp_path, payload, chunk_size_bytes=4)
    (tmp_path / "staged.bin.partial").write_bytes(b"abcdWXYZ")
    with pytest.raises(SourceAcquisitionError, match="does not match source prefix"):
        verify_and_stage_local_mirror(plan, source_path, resume=True)
    assert not (tmp_path / "staged.bin").exists()


def test_resume_false_rejects_existing_partial(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(payload)
    plan = _plan(tmp_path, payload, chunk_size_bytes=4)
    (tmp_path / "staged.bin.partial").write_bytes(payload[:4])
    with pytest.raises(SourceAcquisitionError, match="resume=false"):
        verify_and_stage_local_mirror(plan, source_path, resume=False)


def test_inventory_is_deterministic_and_rejects_duplicate_source_version(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(payload)
    registry = _registry(payload)
    plan = plan_from_registered_source(
        registry,
        "source-a",
        "v1",
        (tmp_path / "staged.bin").resolve().as_uri(),
        chunk_size_bytes=4,
    )
    receipt = verify_and_stage_local_mirror(plan, source_path)
    first = build_retrieval_inventory(registry, [receipt])
    assert first == build_retrieval_inventory(registry, [receipt])
    assert first["rights_semantics"] == "INVENTORY_IS_NOT_TRAINING_APPROVAL"
    with pytest.raises(SourceAcquisitionError, match="duplicate source version"):
        build_retrieval_inventory(registry, [receipt, receipt])
