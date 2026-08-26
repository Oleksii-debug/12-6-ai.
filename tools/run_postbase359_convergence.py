from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    sha256_json,
)

WORKER_ID = "POSTBASE-359-TEACHER-FACTORY-CONVERGENCE"
PARENT_CANDIDATE_HEAD = "5ec7bc917b506273751f0efa8d1048431bcafc8d"


def _factory(curator: VersionedDatasetCurator, teacher) -> TeacherStudentDataFactory:
    return TeacherStudentDataFactory(
        student=MockStudent("3"),
        teachers=(teacher,),
        critic=MockCritic(),
        verifier=ExactMatchVerifier({"arithmetic-1": "4"}),
        judge=DeterministicJudge(),
        curator=curator,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    accepted_curator = VersionedDatasetCurator("postbase359-proof", "v1")
    accepted = _factory(accepted_curator, CorrectTeacher("4")).run(
        make_task("arithmetic-1", "What is 2 + 2?")
    )

    rejected_curator = VersionedDatasetCurator("postbase359-proof-wrong", "v1")
    rejected = _factory(rejected_curator, ConfidentlyWrongTeacher("5")).run(
        make_task("arithmetic-1", "What is 2 + 2?")
    )

    assert accepted.accepted and accepted.record is not None
    assert not rejected.accepted and rejected.record is None
    assert rejected.teacher_proposals[0].confidence == 1.0
    assert rejected.verifications[0].status.value == "CONTRADICTED"
    assert rejected_curator.records == ()

    accepted_manifest = accepted_curator.manifest()
    root_manifest = accepted_curator.manifest_history[0]
    assert accepted.record.parent_dataset_manifest_sha256 == root_manifest["manifest_sha256"]
    assert accepted_manifest["parent_manifest_sha256"] == root_manifest["manifest_sha256"]
    assert accepted_manifest["canonical_base_training_eligible"] is False
    assert accepted.record.canonical_base_training_eligible is False

    proposal = accepted.teacher_proposals[0]
    verification = accepted.verifications[0]
    assert accepted.critic_review.provenance.parent_ids == (
        accepted.student_answer.contribution_id,
        proposal.contribution_id,
    )
    assert verification.provenance.parent_ids == (
        proposal.contribution_id,
        accepted.critic_review.contribution_id,
    )
    assert accepted.judge_decision.provenance.parent_ids == (
        accepted.critic_review.contribution_id,
        verification.contribution_id,
    )

    base_boundary_blocked = False
    try:
        accepted_curator.as_base_corpus_evidence()
    except BaseCorpusBoundaryError:
        base_boundary_blocked = True
    assert base_boundary_blocked

    report = {
        "schema": "12-6.postbase359-teacher-factory-convergence.v1",
        "worker_id": WORKER_ID,
        "parent_candidate_head": PARENT_CANDIDATE_HEAD,
        "execution_mode": "LOCAL_FREE",
        "external_teacher_calls": 0,
        "provider_sdk_dependencies": [],
        "canonical_base_training_executed": False,
        "canonical_base_training_eligible": False,
        "base_corpus_evidence": False,
        "flow": [
            "teacher_proposal",
            "independent_critique",
            "deterministic_verification",
            "accept_reject",
            "versioned_postbase_dataset",
        ],
        "accepted_fixture": {
            "accepted": accepted.accepted,
            "record_id": accepted.record.record_id,
            "record_revision": accepted.record.dataset_revision,
            "manifest_sha256": accepted_manifest["manifest_sha256"],
        },
        "confidently_wrong_fixture": {
            "teacher_confidence": rejected.teacher_proposals[0].confidence,
            "accepted": rejected.accepted,
            "verification_status": rejected.verifications[0].status.value,
            "curated_records": len(rejected_curator.records),
        },
        "base_boundary_blocked": base_boundary_blocked,
    }
    report["report_sha256"] = sha256_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
