from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.postbase import (
    BaseCorpusBoundaryError,
    ConfidentlyWrongTeacher,
    CorrectTeacher,
    DeterministicJudge,
    DisagreeingTeacherA,
    DisagreeingTeacherB,
    ExactMatchVerifier,
    MockCritic,
    MockStudent,
    TeacherStudentDataFactory,
    VersionedDatasetCurator,
    make_task,
    sha256_json,
)


def _run(*teachers):
    curator = VersionedDatasetCurator("postbase259-local-proof", "v1")
    factory = TeacherStudentDataFactory(
        student=MockStudent("3"),
        teachers=teachers,
        critic=MockCritic(),
        verifier=ExactMatchVerifier({"arithmetic-1": "4"}),
        judge=DeterministicJudge(),
        curator=curator,
    )
    return factory.run(make_task("arithmetic-1", "What is 2 + 2?")), curator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    correct, correct_curator = _run(CorrectTeacher())
    wrong, wrong_curator = _run(ConfidentlyWrongTeacher())
    disagreement, disagreement_curator = _run(DisagreeingTeacherA(), DisagreeingTeacherB())

    base_boundary_blocked = False
    try:
        correct_curator.as_base_corpus_evidence()
    except BaseCorpusBoundaryError:
        base_boundary_blocked = True

    report = {
        "schema": "12-6.postbase259-proof.v1",
        "worker_id": "POSTBASE-259-TEACHER-STUDENT-DATA-FACTORY-V1",
        "execution_mode": "LOCAL_FREE",
        "foreign_teacher_calls": 0,
        "provider_dependencies": [],
        "canonical_base_training_executed": False,
        "correct_teacher": {
            "accepted": correct.accepted,
            "record_id": correct.record.record_id if correct.record else None,
            "manifest": correct_curator.manifest(),
        },
        "confidently_wrong_teacher": {
            "accepted": wrong.accepted,
            "verification_status": wrong.verifications[0].status.value,
            "curated_records": len(wrong_curator.records),
        },
        "two_disagreeing_teachers": {
            "accepted": disagreement.accepted,
            "statuses": [item.status.value for item in disagreement.verifications],
            "selected_proposal_id": disagreement.judge_decision.selected_proposal_id,
            "curated_records": len(disagreement_curator.records),
        },
        "base_boundary_blocked": base_boundary_blocked,
    }
    report["report_sha256"] = sha256_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert correct.accepted and correct.record is not None
    assert correct.record.base_corpus_evidence is False
    assert not wrong.accepted and not wrong_curator.records
    assert disagreement.accepted and disagreement.record is not None
    assert base_boundary_blocked


if __name__ == "__main__":
    main()
