from __future__ import annotations

from typing import Any, Sequence

from .contracts import (
    AcceptanceError,
    BaseCorpusBoundaryError,
    CONVERGENCE_WORKER_ID,
    CriticAdapter,
    CriticReview,
    DATASET_CLASSIFICATION,
    DatasetCuratorAdapter,
    DatasetRecord,
    Decision,
    DeterministicVerifierAdapter,
    FactoryError,
    FactoryResult,
    JudgeAdapter,
    JudgeDecision,
    Role,
    StudentAdapter,
    StudentAnswer,
    TaskRecord,
    TeacherAdapter,
    TeacherProposal,
    TeacherRequest,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
    provenance,
    sha256_json,
)

_GENESIS_MANIFEST_SHA256 = "0" * 64


class DeterministicJudge:
    adapter_id = "postbase259-deterministic-judge-v1"

    def decide(
        self,
        task: TaskRecord,
        student: StudentAnswer,
        proposals: Sequence[TeacherProposal],
        critic: CriticReview,
        verifications: Sequence[VerificationResult],
    ) -> tuple[Decision, str | None, tuple[str, ...], str, tuple[str, ...]]:
        del task, student, critic
        eligible: list[TeacherProposal] = []
        support_ids: list[str] = []
        rejected: list[str] = []
        for proposal in proposals:
            checks = [v for v in verifications if v.subject_proposal_id == proposal.contribution_id]
            supported = any(v.status is VerificationStatus.SUPPORTED for v in checks)
            contradicted = any(v.status is VerificationStatus.CONTRADICTED for v in checks)
            if supported and not contradicted:
                eligible.append(proposal)
                support_ids.extend(
                    v.contribution_id for v in checks if v.status is VerificationStatus.SUPPORTED
                )
            else:
                rejected.append(proposal.contribution_id)
        if not eligible:
            return (
                Decision.REJECT,
                None,
                tuple(sorted(rejected)),
                "No proposal has independent deterministic support without contradiction.",
                tuple(sorted(set(support_ids))),
            )
        if len({p.proposed_answer for p in eligible}) > 1:
            return (
                Decision.REJECT,
                None,
                tuple(sorted(p.contribution_id for p in proposals)),
                "Multiple independently supported proposals still disagree.",
                tuple(sorted(set(support_ids))),
            )
        selected = min(eligible, key=lambda p: p.contribution_id)
        rejected.extend(p.contribution_id for p in eligible if p is not selected)
        return (
            Decision.ACCEPT,
            selected.contribution_id,
            tuple(sorted(set(rejected))),
            "Selected proposal is independently supported and not contradicted.",
            tuple(sorted(set(support_ids))),
        )


class VersionedDatasetCurator:
    adapter_id = "postbase259-dataset-curator-v1"

    def __init__(
        self,
        dataset_name: str,
        dataset_version: str,
        *,
        parent_manifest_sha256: str | None = None,
    ) -> None:
        if not dataset_name or not dataset_version:
            raise ValueError("dataset_name and dataset_version are required")
        parent = parent_manifest_sha256 or _GENESIS_MANIFEST_SHA256
        if len(parent) != 64 or any(ch not in "0123456789abcdef" for ch in parent):
            raise ValueError("parent_manifest_sha256 must be a lowercase SHA-256 hex digest")
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.parent_manifest_sha256 = parent
        self._records: list[DatasetRecord] = []
        self._manifest_history: list[dict[str, Any]] = [
            self._build_manifest(dataset_revision=0, parent_manifest_sha256=parent)
        ]

    @property
    def records(self) -> tuple[DatasetRecord, ...]:
        return tuple(self._records)

    @property
    def manifest_history(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._manifest_history)

    def _build_manifest(
        self,
        *,
        dataset_revision: int,
        parent_manifest_sha256: str,
    ) -> dict[str, Any]:
        body = {
            "schema": "12-6.postbase-synthetic-dataset-manifest.v2",
            "worker_id": CONVERGENCE_WORKER_ID,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_revision": dataset_revision,
            "parent_manifest_sha256": parent_manifest_sha256,
            "classification": DATASET_CLASSIFICATION,
            "base_corpus_evidence": False,
            "canonical_base_training_eligible": False,
            "training_use": "POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY",
            "record_ids": tuple(r.record_id for r in self._records),
        }
        return {**body, "manifest_sha256": sha256_json(body)}

    def _selected(
        self,
        task: TaskRecord,
        student: StudentAnswer,
        proposals: Sequence[TeacherProposal],
        critic: CriticReview,
        verifications: Sequence[VerificationResult],
        decision: JudgeDecision,
    ) -> TeacherProposal:
        if decision.decision is not Decision.ACCEPT or not decision.selected_proposal_id:
            raise AcceptanceError("judge did not accept a proposal")
        selected = next(
            (p for p in proposals if p.contribution_id == decision.selected_proposal_id), None
        )
        if selected is None:
            raise AcceptanceError("judge selected an unknown teacher proposal")
        checks = [v for v in verifications if v.subject_proposal_id == selected.contribution_id]
        if not checks:
            raise AcceptanceError("teacher output alone cannot enter accepted training data")
        if any(v.status is VerificationStatus.CONTRADICTED for v in checks):
            raise AcceptanceError("deterministic verification contradicts the selected proposal")
        support = [v for v in checks if v.status is VerificationStatus.SUPPORTED]
        if not support:
            raise AcceptanceError("selected proposal lacks independent deterministic support")
        teachers = {p.teacher_id for p in proposals}
        if any(v.verifier_id in teachers for v in support):
            raise AcceptanceError("teacher cannot serve as its own independent verifier")
        if any(v.verifier_id == critic.provenance.actor_id for v in support):
            raise AcceptanceError("critic cannot serve as the deterministic verifier")
        if any(v.provenance.role is not Role.DETERMINISTIC_VERIFIER for v in support):
            raise AcceptanceError("support did not come from a deterministic verifier")
        expected_critic_parents = (student.contribution_id, *(p.contribution_id for p in proposals))
        if critic.provenance.parent_ids != expected_critic_parents:
            raise AcceptanceError("critic provenance does not follow all teacher proposals")
        for verification in checks:
            if verification.provenance.parent_ids != (
                selected.contribution_id,
                critic.contribution_id,
            ):
                raise AcceptanceError("verification provenance does not follow independent critique")
        expected_judge_parents = (
            critic.contribution_id,
            *(v.contribution_id for v in verifications),
        )
        if decision.provenance.parent_ids != expected_judge_parents:
            raise AcceptanceError("judge provenance does not follow deterministic verification")
        if student.provenance.role is not Role.STUDENT:
            raise AcceptanceError("invalid student provenance")
        if critic.provenance.role is not Role.CRITIC:
            raise AcceptanceError("invalid critic provenance")
        if decision.provenance.role is not Role.JUDGE:
            raise AcceptanceError("invalid judge provenance")
        ids = {
            student.task_id,
            critic.task_id,
            decision.task_id,
            *(p.task_id for p in proposals),
            *(v.task_id for v in verifications),
        }
        if ids != {task.task_id}:
            raise AcceptanceError("cross-task provenance contamination detected")
        return selected

    def curate(
        self,
        *,
        task: TaskRecord,
        student: StudentAnswer,
        proposals: Sequence[TeacherProposal],
        critic: CriticReview,
        verifications: Sequence[VerificationResult],
        decision: JudgeDecision,
    ) -> DatasetRecord:
        self._selected(task, student, proposals, critic, verifications, decision)
        previous_manifest = self._manifest_history[-1]
        dataset_revision = int(previous_manifest["dataset_revision"]) + 1
        parent_manifest_sha256 = str(previous_manifest["manifest_sha256"])
        payload = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_revision": dataset_revision,
            "parent_dataset_manifest_sha256": parent_manifest_sha256,
            "classification": DATASET_CLASSIFICATION,
            "base_corpus_evidence": False,
            "canonical_base_training_eligible": False,
            "training_use": "POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY",
            "task_id": task.task_id,
            "student": student.contribution_id,
            "teachers": tuple(p.contribution_id for p in proposals),
            "critic": critic.contribution_id,
            "verifications": tuple(v.contribution_id for v in verifications),
            "judge": decision.contribution_id,
        }
        parents = (
            task.provenance.provenance_id,
            student.contribution_id,
            *(p.contribution_id for p in proposals),
            critic.contribution_id,
            *(v.contribution_id for v in verifications),
            decision.contribution_id,
            parent_manifest_sha256,
        )
        prov = provenance(
            Role.DATASET_CURATOR,
            self.adapter_id,
            "LOCAL_CURATION",
            f"{self.dataset_name}:{self.dataset_version}:r{dataset_revision}",
            parents,
            payload,
        )
        record = DatasetRecord(
            sha256_json({"payload": payload, "provenance": prov}),
            self.dataset_name,
            self.dataset_version,
            dataset_revision,
            parent_manifest_sha256,
            DATASET_CLASSIFICATION,
            False,
            False,
            "POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY",
            task,
            student,
            tuple(proposals),
            critic,
            tuple(verifications),
            decision,
            prov,
        )
        self._records.append(record)
        self._manifest_history.append(
            self._build_manifest(
                dataset_revision=dataset_revision,
                parent_manifest_sha256=parent_manifest_sha256,
            )
        )
        return record

    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest_history[-1])

    def as_base_corpus_evidence(self) -> None:
        raise BaseCorpusBoundaryError(
            "POSTBASE synthetic datasets are not Base corpus evidence "
            "and cannot be exported as such"
        )


class TeacherStudentDataFactory:
    def __init__(
        self,
        *,
        student: StudentAdapter,
        teachers: Sequence[TeacherAdapter],
        critic: CriticAdapter,
        verifier: DeterministicVerifierAdapter,
        judge: JudgeAdapter,
        curator: DatasetCuratorAdapter,
    ) -> None:
        if not teachers:
            raise ValueError("at least one teacher is required")
        teacher_ids = [t.adapter_id for t in teachers]
        if len(teacher_ids) != len(set(teacher_ids)):
            raise ValueError("teacher adapter IDs must be unique")
        actor_ids = [
            student.adapter_id,
            *teacher_ids,
            critic.adapter_id,
            verifier.adapter_id,
            judge.adapter_id,
            curator.adapter_id,
        ]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError(
                "student, teacher, critic, verifier, judge, and curator adapter IDs "
                "must be pairwise independent"
            )
        self.student = student
        self.teachers = tuple(teachers)
        self.critic = critic
        self.verifier = verifier
        self.judge = judge
        self.curator = curator

    def run(self, task: TaskRecord) -> FactoryResult:
        answer = self.student.answer(task.task_id, task.prompt)
        payload = {"task_id": task.task_id, "answer": answer}
        student_prov = provenance(
            Role.STUDENT,
            self.student.adapter_id,
            "LOCAL_ADAPTER",
            self.student.adapter_id,
            (task.provenance.provenance_id,),
            payload,
        )
        student_id = sha256_json({"payload": payload, "provenance": student_prov})
        student = StudentAnswer(student_id, task.task_id, answer, student_prov)

        proposals: list[TeacherProposal] = []
        for teacher in self.teachers:
            request = TeacherRequest(task.task_id, task.prompt, answer, student_id)
            response = teacher.propose(request)
            if not 0.0 <= response.confidence <= 1.0:
                raise FactoryError("teacher confidence must be in [0, 1]")
            payload = {
                "task_id": task.task_id,
                "student_answer_id": student_id,
                "teacher_id": teacher.adapter_id,
                "critique": response.critique,
                "proposed_answer": response.proposed_answer,
                "confidence": response.confidence,
            }
            prov = provenance(
                Role.TEACHER,
                teacher.adapter_id,
                "TEACHER_ADAPTER",
                response.source_ref,
                (task.provenance.provenance_id, student_id),
                payload,
            )
            proposal_id = sha256_json({"payload": payload, "provenance": prov})
            proposals.append(
                TeacherProposal(
                    proposal_id,
                    task.task_id,
                    student_id,
                    teacher.adapter_id,
                    response.critique,
                    response.proposed_answer,
                    response.confidence,
                    prov,
                )
            )

        findings = self.critic.review(task, student, proposals)
        critic_payload = {
            "task_id": task.task_id,
            "student_answer_id": student_id,
            "proposal_ids": tuple(p.contribution_id for p in proposals),
            "findings": findings,
        }
        critic_prov = provenance(
            Role.CRITIC,
            self.critic.adapter_id,
            "LOCAL_ADAPTER",
            self.critic.adapter_id,
            (student_id, *(p.contribution_id for p in proposals)),
            critic_payload,
        )
        critic = CriticReview(
            sha256_json({"payload": critic_payload, "provenance": critic_prov}),
            task.task_id,
            findings,
            critic_prov,
        )

        verifications: list[VerificationResult] = []
        for proposal in proposals:
            result = self.verifier.verify(
                VerificationRequest(task.task_id, task.prompt, answer, proposal, critic)
            )
            payload = {
                "task_id": task.task_id,
                "proposal_id": proposal.contribution_id,
                "critic_review_id": critic.contribution_id,
                "verifier_id": self.verifier.adapter_id,
                "status": result.status,
                "evidence": result.evidence,
            }
            prov = provenance(
                Role.DETERMINISTIC_VERIFIER,
                self.verifier.adapter_id,
                "DETERMINISTIC_LOCAL_VERIFIER",
                result.source_ref,
                (proposal.contribution_id, critic.contribution_id),
                payload,
            )
            verifications.append(
                VerificationResult(
                    sha256_json({"payload": payload, "provenance": prov}),
                    task.task_id,
                    proposal.contribution_id,
                    self.verifier.adapter_id,
                    result.status,
                    result.evidence,
                    prov,
                )
            )

        value, selected, rejected, rationale, support_ids = self.judge.decide(
            task, student, proposals, critic, verifications
        )
        payload = {
            "task_id": task.task_id,
            "decision": value,
            "selected": selected,
            "rejected": rejected,
            "rationale": rationale,
            "support_ids": support_ids,
        }
        judge_prov = provenance(
            Role.JUDGE,
            self.judge.adapter_id,
            "LOCAL_JUDGE",
            self.judge.adapter_id,
            (critic.contribution_id, *(v.contribution_id for v in verifications)),
            payload,
        )
        decision = JudgeDecision(
            sha256_json({"payload": payload, "provenance": judge_prov}),
            task.task_id,
            value,
            selected,
            rejected,
            rationale,
            support_ids,
            judge_prov,
        )
        if value is Decision.REJECT:
            return FactoryResult(
                False,
                rationale,
                None,
                task,
                student,
                tuple(proposals),
                critic,
                tuple(verifications),
                decision,
            )
        try:
            record = self.curator.curate(
                task=task,
                student=student,
                proposals=proposals,
                critic=critic,
                verifications=verifications,
                decision=decision,
            )
        except AcceptanceError as exc:
            return FactoryResult(
                False,
                str(exc),
                None,
                task,
                student,
                tuple(proposals),
                critic,
                tuple(verifications),
                decision,
            )
        return FactoryResult(
            True,
            "accepted into versioned POSTBASE/EXPERIMENTAL synthetic dataset",
            record,
            task,
            student,
            tuple(proposals),
            critic,
            tuple(verifications),
            decision,
        )
