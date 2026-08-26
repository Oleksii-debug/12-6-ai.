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
    VerificationStatus,
    VersionedDatasetCurator,
    make_task,
)
from twelve_six.postbase.contracts import (
    FactoryError,
    TeacherRequest,
    TeacherResponse,
    VerificationRequest,
    VerificationResponse,
)

WORKER_ID = "NEXT100-090-TEACHER-FACTORY-REDTEAM"


class PlausibleUnverifiableTeacher:
    adapter_id = "redteam-plausible-unverifiable-teacher-v1"

    def propose(self, request: TeacherRequest) -> TeacherResponse:
        del request
        return TeacherResponse(
            "This sounds plausible, but no local deterministic evidence supports it.",
            "Project Borealis was approved on 2026-08-25.",
            0.99,
            "local-redteam:plausible-unverifiable",
        )


class StaleEvidenceVerifier(ExactMatchVerifier):
    adapter_id = "redteam-stale-evidence-verifier-v1"

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        response = super().verify(request)
        stale_revision = "f" * 64
        if stale_revision == self.evidence_revision:
            stale_revision = "e" * 64
        return replace(response, evidence_revision=stale_revision)


class ForgedBindingVerifier(ExactMatchVerifier):
    adapter_id = "redteam-forged-binding-verifier-v1"

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        response = super().verify(request)
        forged_subject = "a" * 64
        if forged_subject == response.subject_sha256:
            forged_subject = "b" * 64
        return replace(response, subject_sha256=forged_subject)


class BaseEligibleCurator:
    adapter_id = "redteam-base-eligible-curator-v1"

    def __init__(self) -> None:
        self.delegate = VersionedDatasetCurator("next100-090-base-boundary", "v1")

    def curate(self, **kwargs):
        safe_record = self.delegate.curate(**kwargs)
        return replace(safe_record, canonical_base_training_eligible=True)


def _factory(
    *,
    teacher=None,
    verifier=None,
    critic=None,
    curator=None,
) -> TeacherStudentDataFactory:
    return TeacherStudentDataFactory(
        student=MockStudent("3"),
        teachers=(teacher or CorrectTeacher("4"),),
        critic=critic or MockCritic(),
        verifier=verifier or ExactMatchVerifier({"arithmetic-1": "4"}),
        judge=DeterministicJudge(),
        curator=curator or VersionedDatasetCurator("next100-090", "v1"),
    )


def test_objectively_correct_verified_proposal_still_passes() -> None:
    curator = VersionedDatasetCurator("next100-090-positive", "v1")
    result = _factory(curator=curator).run(
        make_task("arithmetic-1", "What is 2 + 2?")
    )

    assert result.accepted is True
    assert result.record is not None
    assert result.teacher_proposals[0].proposed_answer == "4"
    assert result.verifications[0].status is VerificationStatus.SUPPORTED
    assert len(result.verifications[0].evidence_revision) == 64
    assert len(result.verifications[0].subject_sha256) == 64
    assert result.record.canonical_base_training_eligible is False
    assert result.record.base_corpus_evidence is False


def test_high_confidence_wrong_answer_is_rejected() -> None:
    result = _factory(teacher=ConfidentlyWrongTeacher("5")).run(
        make_task("arithmetic-1", "What is 2 + 2?")
    )

    assert result.teacher_proposals[0].confidence == 1.0
    assert result.verifications[0].status is VerificationStatus.CONTRADICTED
    assert result.accepted is False
    assert result.record is None


def test_plausible_but_unverifiable_claim_is_rejected() -> None:
    result = _factory(
        teacher=PlausibleUnverifiableTeacher(),
        verifier=ExactMatchVerifier({"arithmetic-1": "4"}),
    ).run(
        make_task(
            "project-borealis-claim",
            "On what date was Project Borealis approved?",
        )
    )

    assert result.verifications[0].status is VerificationStatus.INCONCLUSIVE
    assert result.accepted is False
    assert result.record is None


def test_critic_verifier_identity_reuse_fails_closed() -> None:
    critic = MockCritic()
    verifier = ExactMatchVerifier({"arithmetic-1": "4"})
    critic.adapter_id = verifier.adapter_id

    with pytest.raises(ValueError, match="pairwise independent"):
        _factory(critic=critic, verifier=verifier)


def test_forged_verifier_subject_binding_fails_closed() -> None:
    verifier = ForgedBindingVerifier({"arithmetic-1": "4"})

    with pytest.raises(FactoryError, match="forged deterministic verification subject binding"):
        _factory(verifier=verifier).run(
            make_task("arithmetic-1", "What is 2 + 2?")
        )


def test_mutated_forged_verification_cannot_be_curated() -> None:
    curator = VersionedDatasetCurator("next100-090-forged-verification", "v1")
    result = _factory(curator=curator).run(
        make_task("arithmetic-1", "What is 2 + 2?")
    )
    assert result.record is not None

    forged = replace(
        result.verifications[0],
        evidence="forged proof inserted after verification",
    )
    with pytest.raises(AcceptanceError, match="verification provenance payload hash mismatch"):
        curator.curate(
            task=result.task,
            student=result.student_answer,
            proposals=result.teacher_proposals,
            critic=result.critic_review,
            verifications=(forged,),
            decision=result.judge_decision,
        )


def test_stale_evidence_revision_fails_closed() -> None:
    verifier = StaleEvidenceVerifier({"arithmetic-1": "4"})

    with pytest.raises(FactoryError, match="stale deterministic verifier evidence revision"):
        _factory(verifier=verifier).run(
            make_task("arithmetic-1", "What is 2 + 2?")
        )


def test_contradictory_evidence_is_a_hard_veto() -> None:
    curator = VersionedDatasetCurator("next100-090-contradiction", "v1")
    result = _factory(curator=curator).run(
        make_task("arithmetic-1", "What is 2 + 2?")
    )
    assert result.record is not None

    contradiction = replace(
        result.verifications[0],
        status=VerificationStatus.CONTRADICTED,
    )
    with pytest.raises(AcceptanceError, match="verification contradicts"):
        curator.curate(
            task=result.task,
            student=result.student_answer,
            proposals=result.teacher_proposals,
            critic=result.critic_review,
            verifications=(result.verifications[0], contradiction),
            decision=result.judge_decision,
        )


def test_dataset_parent_forgery_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete parent_manifest"):
        VersionedDatasetCurator(
            "next100-090-parent",
            "v2",
            parent_manifest_sha256="a" * 64,
        )

    v1 = VersionedDatasetCurator("next100-090-parent", "v1")
    manifest = v1.manifest()
    forged_manifest = dict(manifest)
    forged_manifest["record_ids"] = ("forged-record",)
    with pytest.raises(ValueError, match="does not match its immutable content"):
        VersionedDatasetCurator(
            "next100-090-parent",
            "v2",
            parent_manifest=forged_manifest,
        )


def test_attempt_to_mark_output_canonical_base_eligible_is_rejected() -> None:
    result = _factory(curator=BaseEligibleCurator()).run(
        make_task("arithmetic-1", "What is 2 + 2?")
    )

    assert result.accepted is False
    assert result.record is None
    assert "canonical Base eligible" in result.reason
