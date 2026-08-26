from __future__ import annotations

from dataclasses import replace

import pytest

from twelve_six.postbase import (
    AcceptanceError,
    BaseCorpusBoundaryError,
    ConfidentlyWrongTeacher,
    CorrectTeacher,
    DATASET_CLASSIFICATION,
    Decision,
    DeterministicJudge,
    DisagreeingTeacherA,
    DisagreeingTeacherB,
    ExactMatchVerifier,
    MockCritic,
    MockStudent,
    Role,
    TeacherStudentDataFactory,
    VerificationStatus,
    VersionedDatasetCurator,
    canonical_json,
    make_task,
)


def _factory(*teachers):
    curator = VersionedDatasetCurator("postbase259-proof", "v1")
    return TeacherStudentDataFactory(
        student=MockStudent("3"),
        teachers=teachers,
        critic=MockCritic(),
        verifier=ExactMatchVerifier({"arithmetic-1": "4"}),
        judge=DeterministicJudge(),
        curator=curator,
    ), curator


def test_correct_teacher_requires_verification_then_enters_postbase_dataset() -> None:
    factory, curator = _factory(CorrectTeacher())
    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))

    assert result.accepted is True
    assert result.record is not None
    assert result.record.classification == DATASET_CLASSIFICATION
    assert result.record.base_corpus_evidence is False
    assert result.record.training_use == "POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY"
    assert result.judge_decision.decision is Decision.ACCEPT
    assert result.verifications[0].status is VerificationStatus.SUPPORTED
    assert curator.records == (result.record,)


def test_confidently_wrong_teacher_is_rejected_when_verifier_contradicts() -> None:
    factory, curator = _factory(ConfidentlyWrongTeacher())
    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))

    assert result.accepted is False
    assert result.record is None
    assert result.verifications[0].status is VerificationStatus.CONTRADICTED
    assert result.judge_decision.decision is Decision.REJECT
    assert curator.records == ()


def test_two_disagreeing_teachers_resolve_only_to_independently_supported_proposal() -> None:
    factory, _ = _factory(DisagreeingTeacherA("4"), DisagreeingTeacherB("5"))
    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))

    assert result.accepted is True
    assert result.record is not None
    selected = next(
        proposal
        for proposal in result.teacher_proposals
        if proposal.contribution_id == result.judge_decision.selected_proposal_id
    )
    assert selected.proposed_answer == "4"
    statuses = {
        proposal.proposed_answer: verification.status
        for proposal, verification in zip(
            result.teacher_proposals, result.verifications, strict=True
        )
    }
    assert statuses == {
        "4": VerificationStatus.SUPPORTED,
        "5": VerificationStatus.CONTRADICTED,
    }


def test_teacher_output_alone_cannot_be_curated() -> None:
    factory, curator = _factory(CorrectTeacher())
    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))
    assert result.record is not None

    with pytest.raises(AcceptanceError, match="teacher output alone"):
        curator.curate(
            task=result.task,
            student=result.student_answer,
            proposals=result.teacher_proposals,
            critic=result.critic_review,
            verifications=(),
            decision=result.judge_decision,
        )


def test_curator_rechecks_contradiction_even_if_a_bad_judge_accepts() -> None:
    factory, curator = _factory(ConfidentlyWrongTeacher())
    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))
    proposal = result.teacher_proposals[0]
    forged = replace(
        result.judge_decision,
        decision=Decision.ACCEPT,
        selected_proposal_id=proposal.contribution_id,
    )

    with pytest.raises(AcceptanceError, match="contradicts"):
        curator.curate(
            task=result.task,
            student=result.student_answer,
            proposals=result.teacher_proposals,
            critic=result.critic_review,
            verifications=result.verifications,
            decision=forged,
        )


def test_all_required_roles_have_exact_provenance() -> None:
    factory, _ = _factory(CorrectTeacher())
    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))
    assert result.record is not None

    assert result.student_answer.provenance.role is Role.STUDENT
    assert all(item.provenance.role is Role.TEACHER for item in result.teacher_proposals)
    assert result.critic_review.provenance.role is Role.CRITIC
    assert all(
        item.provenance.role is Role.DETERMINISTIC_VERIFIER for item in result.verifications
    )
    assert result.judge_decision.provenance.role is Role.JUDGE
    assert result.record.curator_provenance.role is Role.DATASET_CURATOR

    provenance_ids = [
        result.task.provenance.provenance_id,
        result.student_answer.provenance.provenance_id,
        *(item.provenance.provenance_id for item in result.teacher_proposals),
        result.critic_review.provenance.provenance_id,
        *(item.provenance.provenance_id for item in result.verifications),
        result.judge_decision.provenance.provenance_id,
        result.record.curator_provenance.provenance_id,
    ]
    assert all(len(item) == 64 for item in provenance_ids)
    assert all(set(item) <= set("0123456789abcdef") for item in provenance_ids)


def test_identical_local_run_has_identical_record_identity() -> None:
    first, _ = _factory(CorrectTeacher())
    second, _ = _factory(CorrectTeacher())
    task = make_task("arithmetic-1", "What is 2 + 2?")

    result_a = first.run(task)
    result_b = second.run(task)
    assert result_a.record is not None and result_b.record is not None
    assert result_a.record.record_id == result_b.record.record_id
    assert canonical_json(result_a.record) == canonical_json(result_b.record)


def test_postbase_dataset_cannot_masquerade_as_base_corpus_evidence() -> None:
    factory, curator = _factory(CorrectTeacher())
    result = factory.run(make_task("arithmetic-1", "What is 2 + 2?"))
    assert result.record is not None

    manifest = curator.manifest()
    assert manifest["classification"] == "POSTBASE/EXPERIMENTAL"
    assert manifest["base_corpus_evidence"] is False
    assert manifest["canonical_base_training_eligible"] is False
    with pytest.raises(BaseCorpusBoundaryError, match="not Base corpus evidence"):
        curator.as_base_corpus_evidence()


def test_verifier_must_be_independent_from_teacher_identity() -> None:
    teacher = CorrectTeacher()
    verifier = ExactMatchVerifier({"arithmetic-1": "4"})
    verifier.adapter_id = teacher.adapter_id
    with pytest.raises(ValueError, match="independent"):
        TeacherStudentDataFactory(
            student=MockStudent("3"),
            teachers=(teacher,),
            critic=MockCritic(),
            verifier=verifier,
            judge=DeterministicJudge(),
            curator=VersionedDatasetCurator("proof", "v1"),
        )
