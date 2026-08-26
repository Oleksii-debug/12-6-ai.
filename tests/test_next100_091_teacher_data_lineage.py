from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from twelve_six.postbase import (
    BaseCorpusBoundaryError,
    ConfidentlyWrongTeacher,
    CorrectTeacher,
    DeterministicJudge,
    ExactMatchVerifier,
    MockCritic,
    MockStudent,
    TeacherStudentDataFactory,
    VersionedDatasetCurator,
    make_task,
)
from twelve_six.postbase.lineage import (
    LINEAGE_TRAINING_USE,
    LineageIntegrityError,
    LineageMutationError,
    SyntheticDatasetLineage,
)


def _factory(curator: VersionedDatasetCurator, *, teacher=None) -> TeacherStudentDataFactory:
    return TeacherStudentDataFactory(
        student=MockStudent("3"),
        teachers=(teacher or CorrectTeacher("4"),),
        critic=MockCritic(),
        verifier=ExactMatchVerifier(
            {
                "arithmetic-1": "4",
                "arithmetic-2": "4",
                "arithmetic-bad": "4",
            }
        ),
        judge=DeterministicJudge(),
        curator=curator,
    )


def _evidence():
    curator = VersionedDatasetCurator("lineage-proof", "factory-v1")
    good = _factory(curator)
    first = good.run(make_task("arithmetic-1", "What is 2 + 2?"))
    second = good.run(make_task("arithmetic-2", "What is 3 + 1?"))
    rejected = _factory(curator, teacher=ConfidentlyWrongTeacher("5")).run(
        make_task("arithmetic-bad", "What is 2 + 2?")
    )
    assert first.record is not None and second.record is not None
    assert rejected.accepted is False and rejected.record is None
    return first.record, second.record, rejected


def test_version_manifest_captures_required_immutable_lineage_evidence() -> None:
    first, _, rejected = _evidence()
    lineage = SyntheticDatasetLineage("lineage-proof")

    snapshot = lineage.create_version(
        "v1",
        accepted_records=(first,),
        rejected_results=(rejected,),
    )
    manifest = snapshot.manifest()

    assert manifest["dataset_version"] == "v1"
    assert manifest["parent_version"] is None
    assert manifest["parent_manifest_sha256"] == "0" * 64
    assert manifest["accepted_record_hashes"] == (snapshot.accepted_records[0].record_sha256,)
    assert manifest["accepted_record_ids"] == (first.record_id,)
    assert manifest["critic_identities"] == (first.critic_review.provenance.actor_id,)
    assert manifest["verifier_identities"] == tuple(
        sorted({item.verifier_id for item in first.verifications})
    )
    assert manifest["source_proposal_ids"] == tuple(
        sorted(item.contribution_id for item in first.teacher_proposals)
    )
    assert len(manifest["rejection_log"]) == 1
    assert manifest["rejection_log"][0]["task_id"] == "arithmetic-bad"
    assert len(manifest["manifest_sha256"]) == 64
    assert lineage.verify_version("v1") is True
    assert lineage.verify_history() is True


def test_delete_creates_new_version_without_rewriting_parent_history() -> None:
    first, _, rejected = _evidence()
    lineage = SyntheticDatasetLineage("lineage-proof")
    v1 = lineage.create_version("v1", accepted_records=(first,), rejected_results=(rejected,))
    v1_manifest_before = lineage.manifest("v1")

    v2 = lineage.delete_records("v2", (first.record_id,), parent_version="v1")

    assert v2.parent_version == "v1"
    assert v2.parent_manifest_sha256 == v1.manifest_sha256
    assert v2.accepted_records == ()
    assert v2.operation.kind == "DELETE"
    assert v2.operation.deleted_record_ids == (first.record_id,)
    assert lineage.manifest("v1") == v1_manifest_before
    assert lineage.read("v1").accepted_record_ids == (first.record_id,)
    assert lineage.read("v2").accepted_record_ids == ()


def test_supersession_creates_new_version_and_preserves_source_identity_chain() -> None:
    first, second, _ = _evidence()
    lineage = SyntheticDatasetLineage("lineage-proof")
    v1 = lineage.create_version("v1", accepted_records=(first,))

    v2 = lineage.supersede_records(
        "v2",
        {first.record_id: second},
        parent_version="v1",
    )

    assert v1.accepted_record_ids == (first.record_id,)
    assert v2.accepted_record_ids == (second.record_id,)
    assert v2.operation.kind == "SUPERSEDE"
    assert v2.operation.superseded_records == ((first.record_id, second.record_id),)
    assert v2.parent_manifest_sha256 == v1.manifest_sha256
    assert lineage.verify_history() is True


def test_rollback_is_new_version_and_verified_read_binds_expected_manifest_hash() -> None:
    first, _, rejected = _evidence()
    lineage = SyntheticDatasetLineage("lineage-proof")
    v1 = lineage.create_version("v1", accepted_records=(first,), rejected_results=(rejected,))
    v2 = lineage.delete_records("v2", (first.record_id,))

    v3 = lineage.rollback("v3", "v1", parent_version="v2")

    assert v3.parent_version == "v2"
    assert v3.parent_manifest_sha256 == v2.manifest_sha256
    assert v3.accepted_record_ids == v1.accepted_record_ids
    assert v3.operation.kind == "ROLLBACK"
    assert v3.operation.rollback_target_version == "v1"
    assert v3.operation.rollback_target_manifest_sha256 == v1.manifest_sha256
    assert lineage.read("v3", expected_manifest_sha256=v3.manifest_sha256) == v3
    with pytest.raises(LineageIntegrityError, match="manifest SHA-256 mismatch"):
        lineage.read("v3", expected_manifest_sha256="f" * 64)


def test_external_manifest_mutation_cannot_rewrite_stored_version() -> None:
    first, _, _ = _evidence()
    lineage = SyntheticDatasetLineage("lineage-proof")
    snapshot = lineage.create_version("v1", accepted_records=(first,))

    exposed = lineage.manifest("v1")
    exposed["canonical_base_training_eligible"] = True
    exposed["accepted_record_hashes"] = ()
    exposed["manifest_sha256"] = "f" * 64

    stored = lineage.manifest("v1")
    assert stored["canonical_base_training_eligible"] is False
    assert stored["accepted_record_hashes"] == snapshot.accepted_record_hashes
    assert stored["manifest_sha256"] == snapshot.manifest_sha256
    assert lineage.verify_version("v1") is True
    with pytest.raises(FrozenInstanceError):
        snapshot.dataset_version = "rewritten"  # type: ignore[misc]


def test_existing_version_name_cannot_be_reused_for_history_rewrite() -> None:
    first, second, _ = _evidence()
    lineage = SyntheticDatasetLineage("lineage-proof")
    lineage.create_version("v1", accepted_records=(first,))

    with pytest.raises(LineageMutationError, match="immutable history cannot be rewritten"):
        lineage.derive_version("v1", accepted_records=(second,))


def test_canonical_base_eligibility_is_permanently_false_across_operations() -> None:
    first, second, rejected = _evidence()
    lineage = SyntheticDatasetLineage("lineage-proof")
    lineage.create_version("v1", accepted_records=(first,), rejected_results=(rejected,))
    lineage.derive_version("v2", accepted_records=(second,))
    lineage.delete_records("v3", (first.record_id,))
    lineage.rollback("v4", "v1")

    for version in lineage.versions:
        manifest = lineage.manifest(version)
        assert manifest["classification"] == "POSTBASE/EXPERIMENTAL"
        assert manifest["base_corpus_evidence"] is False
        assert manifest["canonical_base_training_eligible"] is False
        assert manifest["training_use"] == LINEAGE_TRAINING_USE
    with pytest.raises(BaseCorpusBoundaryError, match="permanently POSTBASE/EXPERIMENTAL"):
        lineage.as_base_corpus_evidence("v4")
