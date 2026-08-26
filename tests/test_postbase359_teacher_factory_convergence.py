from __future__ import annotations

from dataclasses import replace

import pytest

from twelve_six.postbase import (
    AcceptanceError,
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


def _factory(
    curator: VersionedDatasetCurator,
    *,
    teacher=None,
    critic=None,
    verifier=None,
) -> TeacherStudentDataFactory:
    return TeacherStudentDataFactory(
        student=MockStudent("3"),
        teachers=(teacher or CorrectTeacher("4"),),
        critic=critic or MockCritic(),
        verifier=verifier
        or ExactMatchVerifier(
            {
                "arithmetic-1": "4",
                "arithmetic-2": "4",
            }
        ),
        judge=DeterministicJudge(),
        curator=curator,
    )


def test_confidently_wrong_teacher_is_rejected_before_dataset_admission() -> None:
    curator = VersionedDatasetCurator("postbase359-proof", "v1")
    factory = _factory(curator, teacher=ConfidentlyWrongTeacher("5"))

    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))

    assert result.accepted is False
    assert result.record is None
    assert result.teacher_proposals[0].confidence == 1.0
    assert result.verifications[0].status.value == "CONTRADICTED"
    assert curator.records == ()
    assert curator.manifest()["dataset_revision"] == 0


def test_critic_must_be_independent_from_teacher_identity() -> None:
    curator = VersionedDatasetCurator("postbase359-proof", "v1")
    teacher = CorrectTeacher("4")
    critic = MockCritic()
    critic.adapter_id = teacher.adapter_id

    with pytest.raises(ValueError, match="pairwise independent"):
        _factory(curator, teacher=teacher, critic=critic)


def test_verifier_must_be_independent_from_critic_identity() -> None:
    curator = VersionedDatasetCurator("postbase359-proof", "v1")
    critic = MockCritic()
    verifier = ExactMatchVerifier({"arithmetic-1": "4"})
    verifier.adapter_id = critic.adapter_id

    with pytest.raises(ValueError, match="pairwise independent"):
        _factory(curator, critic=critic, verifier=verifier)


def test_provenance_enforces_proposal_critique_verify_decide_curate_order() -> None:
    curator = VersionedDatasetCurator("postbase359-proof", "v1")
    result = _factory(curator).run(make_task("arithmetic-1", "What is 2 + 2?"))

    assert result.accepted is True
    assert result.record is not None
    proposal = result.teacher_proposals[0]
    verification = result.verifications[0]

    assert result.critic_review.provenance.parent_ids == (
        result.student_answer.contribution_id,
        proposal.contribution_id,
    )
    assert verification.provenance.parent_ids == (
        proposal.contribution_id,
        result.critic_review.contribution_id,
    )
    assert result.judge_decision.provenance.parent_ids == (
        result.critic_review.contribution_id,
        verification.contribution_id,
    )
    assert result.record.curator_provenance.parent_ids[-1] == (
        result.record.parent_dataset_manifest_sha256
    )


def test_curator_rejects_forged_verification_that_skips_independent_critique() -> None:
    curator = VersionedDatasetCurator("postbase359-proof", "v1")
    result = _factory(curator).run(make_task("arithmetic-1", "What is 2 + 2?"))
    assert result.record is not None

    verification = result.verifications[0]
    forged_provenance = replace(
        verification.provenance,
        parent_ids=(verification.subject_proposal_id,),
    )
    forged_verification = replace(verification, provenance=forged_provenance)

    with pytest.raises(AcceptanceError, match="does not follow independent critique"):
        curator.curate(
            task=result.task,
            student=result.student_answer,
            proposals=result.teacher_proposals,
            critic=result.critic_review,
            verifications=(forged_verification,),
            decision=result.judge_decision,
        )


def test_dataset_snapshots_are_sha_linked_and_base_ineligible() -> None:
    curator = VersionedDatasetCurator("postbase359-proof", "v1")
    root = curator.manifest()
    factory = _factory(curator)

    first = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))
    second = factory.run(make_task("arithmetic-2", "What is 3 + 1?"))

    assert first.record is not None and second.record is not None
    history = curator.manifest_history
    assert [item["dataset_revision"] for item in history] == [0, 1, 2]
    assert history[1]["parent_manifest_sha256"] == root["manifest_sha256"]
    assert history[2]["parent_manifest_sha256"] == history[1]["manifest_sha256"]
    assert first.record.parent_dataset_manifest_sha256 == root["manifest_sha256"]
    assert second.record.parent_dataset_manifest_sha256 == history[1]["manifest_sha256"]

    for manifest in history:
        assert manifest["classification"] == "POSTBASE/EXPERIMENTAL"
        assert manifest["base_corpus_evidence"] is False
        assert manifest["canonical_base_training_eligible"] is False
        assert manifest["training_use"] == "POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY"
    assert first.record.base_corpus_evidence is False
    assert first.record.canonical_base_training_eligible is False
    assert second.record.canonical_base_training_eligible is False


def test_successor_dataset_version_chains_to_prior_manifest() -> None:
    v1 = VersionedDatasetCurator("postbase359-proof", "v1")
    result = _factory(v1).run(make_task("arithmetic-1", "What is 2 + 2?"))
    assert result.accepted is True
    v1_manifest = v1.manifest()

    v2 = VersionedDatasetCurator(
        "postbase359-proof",
        "v2",
        parent_manifest_sha256=v1_manifest["manifest_sha256"],
    )
    v2_root = v2.manifest()

    assert v2_root["dataset_version"] == "v2"
    assert v2_root["dataset_revision"] == 0
    assert v2_root["parent_manifest_sha256"] == v1_manifest["manifest_sha256"]
    assert v2_root["canonical_base_training_eligible"] is False


def test_returned_manifest_mutation_cannot_rewrite_stored_version_history() -> None:
    curator = VersionedDatasetCurator("postbase359-proof", "v1")
    result = _factory(curator).run(make_task("arithmetic-1", "What is 2 + 2?"))
    assert result.accepted is True

    exposed = curator.manifest()
    expected_sha = exposed["manifest_sha256"]
    exposed["canonical_base_training_eligible"] = True
    exposed["manifest_sha256"] = "f" * 64

    stored = curator.manifest()
    assert stored["canonical_base_training_eligible"] is False
    assert stored["manifest_sha256"] == expected_sha
