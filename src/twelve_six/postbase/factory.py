from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    CONVERGENCE_WORKER_ID,
    DATASET_CLASSIFICATION,
    AcceptanceError,
    BaseCorpusBoundaryError,
    CriticAdapter,
    CriticReview,
    DatasetCuratorAdapter,
    DatasetRecord,
    Decision,
    DeterministicVerifierAdapter,
    FactoryError,
    FactoryResult,
    JudgeAdapter,
    JudgeDecision,
    Provenance,
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
    verification_subject_sha256,
)

_GENESIS_MANIFEST_SHA256 = "0" * 64
_TRAINING_USE = "POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY"
_MANIFEST_SCHEMA = "12-6.postbase-synthetic-dataset-manifest.v2"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _validate_provenance(prov: Provenance, payload: Any, label: str) -> None:
    payload_sha256 = sha256_json(payload)
    if prov.payload_sha256 != payload_sha256:
        raise AcceptanceError(f"{label} provenance payload hash mismatch")
    body = {
        "role": prov.role,
        "actor_id": prov.actor_id,
        "source_kind": prov.source_kind,
        "source_ref": prov.source_ref,
        "parent_ids": prov.parent_ids,
        "payload_sha256": prov.payload_sha256,
    }
    if prov.provenance_id != sha256_json(body):
        raise AcceptanceError(f"{label} provenance identity mismatch")


def _validate_contribution_id(contribution_id: str, payload: Any, prov: Provenance, label: str) -> None:
    if contribution_id != sha256_json({"payload": payload, "provenance": prov}):
        raise AcceptanceError(f"{label} contribution identity mismatch")


def _task_payload(task: TaskRecord) -> dict[str, Any]:
    return {"task_id": task.task_id, "prompt": task.prompt}


def _student_payload(student: StudentAnswer) -> dict[str, Any]:
    return {"task_id": student.task_id, "answer": student.answer}


def _proposal_payload(proposal: TeacherProposal) -> dict[str, Any]:
    return {
        "task_id": proposal.task_id,
        "student_answer_id": proposal.student_answer_id,
        "teacher_id": proposal.teacher_id,
        "critique": proposal.critique,
        "proposed_answer": proposal.proposed_answer,
        "confidence": proposal.confidence,
    }


def _critic_payload(
    critic: CriticReview,
    student: StudentAnswer,
    proposals: Sequence[TeacherProposal],
) -> dict[str, Any]:
    return {
        "task_id": critic.task_id,
        "student_answer_id": student.contribution_id,
        "proposal_ids": tuple(p.contribution_id for p in proposals),
        "findings": critic.findings,
    }


def _verification_payload(
    verification: VerificationResult,
    critic: CriticReview,
) -> dict[str, Any]:
    return {
        "task_id": verification.task_id,
        "proposal_id": verification.subject_proposal_id,
        "critic_review_id": critic.contribution_id,
        "verifier_id": verification.verifier_id,
        "status": verification.status,
        "evidence": verification.evidence,
        "evidence_revision": verification.evidence_revision,
        "subject_sha256": verification.subject_sha256,
    }


def _judge_payload(decision: JudgeDecision) -> dict[str, Any]:
    return {
        "task_id": decision.task_id,
        "decision": decision.decision,
        "selected": decision.selected_proposal_id,
        "rejected": decision.rejected_proposal_ids,
        "rationale": decision.rationale,
        "support_ids": decision.support_ids,
    }


def _validate_chain_integrity(
    task: TaskRecord,
    student: StudentAnswer,
    proposals: Sequence[TeacherProposal],
    critic: CriticReview,
    verifications: Sequence[VerificationResult],
    decision: JudgeDecision,
) -> None:
    task_payload = _task_payload(task)
    _validate_provenance(task.provenance, task_payload, "task")

    student_payload = _student_payload(student)
    _validate_provenance(student.provenance, student_payload, "student")
    _validate_contribution_id(student.contribution_id, student_payload, student.provenance, "student")
    if student.provenance.role is not Role.STUDENT:
        raise AcceptanceError("invalid student provenance")
    if student.provenance.parent_ids != (task.provenance.provenance_id,):
        raise AcceptanceError("student provenance does not follow task")

    proposal_by_id = {proposal.contribution_id: proposal for proposal in proposals}
    if len(proposal_by_id) != len(proposals):
        raise AcceptanceError("duplicate teacher proposal identity detected")
    for proposal in proposals:
        proposal_payload = _proposal_payload(proposal)
        _validate_provenance(proposal.provenance, proposal_payload, "teacher proposal")
        _validate_contribution_id(
            proposal.contribution_id,
            proposal_payload,
            proposal.provenance,
            "teacher proposal",
        )
        if proposal.provenance.role is not Role.TEACHER:
            raise AcceptanceError("invalid teacher provenance")
        if proposal.provenance.actor_id != proposal.teacher_id:
            raise AcceptanceError("teacher identity does not match proposal provenance")
        if proposal.student_answer_id != student.contribution_id:
            raise AcceptanceError("teacher proposal is bound to the wrong student answer")
        if proposal.provenance.parent_ids != (
            task.provenance.provenance_id,
            student.contribution_id,
        ):
            raise AcceptanceError("teacher proposal provenance does not follow task and student")

    critic_payload = _critic_payload(critic, student, proposals)
    _validate_provenance(critic.provenance, critic_payload, "critic")
    _validate_contribution_id(critic.contribution_id, critic_payload, critic.provenance, "critic")
    if critic.provenance.role is not Role.CRITIC:
        raise AcceptanceError("invalid critic provenance")

    verification_ids: set[str] = set()
    for verification in verifications:
        if verification.contribution_id in verification_ids:
            raise AcceptanceError("duplicate verification identity detected")
        verification_ids.add(verification.contribution_id)
        proposal = proposal_by_id.get(verification.subject_proposal_id)
        if proposal is None:
            raise AcceptanceError("verification targets an unknown teacher proposal")
        if not _is_sha256(verification.evidence_revision):
            raise AcceptanceError("verification evidence revision is missing or malformed")
        request = VerificationRequest(
            task.task_id,
            task.prompt,
            student.answer,
            proposal,
            critic,
        )
        if verification.subject_sha256 != verification_subject_sha256(request):
            raise AcceptanceError("verification is not bound to the exact proposal and critique")
        verification_payload = _verification_payload(verification, critic)
        _validate_provenance(
            verification.provenance,
            verification_payload,
            "verification",
        )
        _validate_contribution_id(
            verification.contribution_id,
            verification_payload,
            verification.provenance,
            "verification",
        )
        if verification.provenance.role is not Role.DETERMINISTIC_VERIFIER:
            raise AcceptanceError("support did not come from a deterministic verifier")
        if verification.provenance.actor_id != verification.verifier_id:
            raise AcceptanceError("verifier identity does not match verification provenance")
        if verification.provenance.parent_ids != (
            proposal.contribution_id,
            critic.contribution_id,
        ):
            raise AcceptanceError("verification provenance does not follow independent critique")

    judge_payload = _judge_payload(decision)
    _validate_provenance(decision.provenance, judge_payload, "judge")
    _validate_contribution_id(decision.contribution_id, judge_payload, decision.provenance, "judge")
    if decision.provenance.role is not Role.JUDGE:
        raise AcceptanceError("invalid judge provenance")
    expected_judge_parents = (
        critic.contribution_id,
        *(v.contribution_id for v in verifications),
    )
    if decision.provenance.parent_ids != expected_judge_parents:
        raise AcceptanceError("judge provenance does not follow deterministic verification")
    proposal_ids = set(proposal_by_id)
    if decision.selected_proposal_id is not None and decision.selected_proposal_id not in proposal_ids:
        raise AcceptanceError("judge selected an unknown teacher proposal")
    if not set(decision.rejected_proposal_ids) <= proposal_ids:
        raise AcceptanceError("judge rejected an unknown teacher proposal")
    supported_ids = {
        item.contribution_id
        for item in verifications
        if item.status is VerificationStatus.SUPPORTED
    }
    if not set(decision.support_ids) <= supported_ids:
        raise AcceptanceError("judge support list contains non-supported verification")

    ids = {
        student.task_id,
        critic.task_id,
        decision.task_id,
        *(p.task_id for p in proposals),
        *(v.task_id for v in verifications),
    }
    if ids != {task.task_id}:
        raise AcceptanceError("cross-task provenance contamination detected")


def _manifest_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "worker_id",
        "dataset_name",
        "dataset_version",
        "dataset_revision",
        "parent_manifest_sha256",
        "classification",
        "base_corpus_evidence",
        "canonical_base_training_eligible",
        "training_use",
        "record_ids",
        "manifest_sha256",
    }
    if set(manifest) != required:
        raise ValueError("parent manifest schema is not exact")
    body = {key: manifest[key] for key in required if key != "manifest_sha256"}
    return body


def _validated_parent_manifest_sha256(
    manifest: Mapping[str, Any],
    *,
    dataset_name: str,
) -> str:
    body = _manifest_body(manifest)
    digest = manifest["manifest_sha256"]
    if not _is_sha256(digest) or digest != sha256_json(body):
        raise ValueError("parent manifest SHA-256 does not match its immutable content")
    if body["schema"] != _MANIFEST_SCHEMA or body["worker_id"] != CONVERGENCE_WORKER_ID:
        raise ValueError("parent manifest is not a POSTBASE-359 synthetic dataset manifest")
    if body["dataset_name"] != dataset_name:
        raise ValueError("parent manifest belongs to a different dataset")
    if not isinstance(body["dataset_revision"], int) or body["dataset_revision"] < 0:
        raise ValueError("parent manifest dataset revision is invalid")
    if not _is_sha256(body["parent_manifest_sha256"]):
        raise ValueError("parent manifest parent identity is invalid")
    if body["classification"] != DATASET_CLASSIFICATION:
        raise BaseCorpusBoundaryError("parent manifest escaped POSTBASE/EXPERIMENTAL classification")
    if body["base_corpus_evidence"] is not False:
        raise BaseCorpusBoundaryError("parent manifest cannot be Base corpus evidence")
    if body["canonical_base_training_eligible"] is not False:
        raise BaseCorpusBoundaryError("parent manifest cannot be canonical Base training eligible")
    if body["training_use"] != _TRAINING_USE:
        raise BaseCorpusBoundaryError("parent manifest has an unsafe training-use classification")
    return str(digest)


def _validate_record_postconditions(
    record: DatasetRecord,
    *,
    task: TaskRecord,
    student: StudentAnswer,
    proposals: Sequence[TeacherProposal],
    critic: CriticReview,
    verifications: Sequence[VerificationResult],
    decision: JudgeDecision,
) -> None:
    if record.classification != DATASET_CLASSIFICATION:
        raise AcceptanceError("curator returned a record outside POSTBASE/EXPERIMENTAL")
    if record.base_corpus_evidence is not False:
        raise AcceptanceError("curator attempted to mark synthetic output as Base corpus evidence")
    if record.canonical_base_training_eligible is not False:
        raise AcceptanceError("curator attempted to mark synthetic output canonical Base eligible")
    if record.training_use != _TRAINING_USE:
        raise AcceptanceError("curator returned an unsafe training-use classification")
    if not _is_sha256(record.parent_dataset_manifest_sha256):
        raise AcceptanceError("curator returned an invalid dataset parent identity")
    if record.task != task or record.student_answer != student:
        raise AcceptanceError("curator mutated task or student content")
    if record.teacher_proposals != tuple(proposals):
        raise AcceptanceError("curator mutated teacher proposal content")
    if record.critic_review != critic or record.verifications != tuple(verifications):
        raise AcceptanceError("curator mutated critique or verification content")
    if record.judge_decision != decision:
        raise AcceptanceError("curator mutated judge content")
    if record.curator_provenance.role is not Role.DATASET_CURATOR:
        raise AcceptanceError("invalid curator provenance")
    if not record.curator_provenance.parent_ids:
        raise AcceptanceError("curator provenance is missing dataset parent")
    if record.curator_provenance.parent_ids[-1] != record.parent_dataset_manifest_sha256:
        raise AcceptanceError("curator provenance does not bind the dataset parent")
    payload = {
        "dataset_name": record.dataset_name,
        "dataset_version": record.dataset_version,
        "dataset_revision": record.dataset_revision,
        "parent_dataset_manifest_sha256": record.parent_dataset_manifest_sha256,
        "classification": record.classification,
        "base_corpus_evidence": record.base_corpus_evidence,
        "canonical_base_training_eligible": record.canonical_base_training_eligible,
        "training_use": record.training_use,
        "task_id": task.task_id,
        "student": student.contribution_id,
        "teachers": tuple(p.contribution_id for p in proposals),
        "critic": critic.contribution_id,
        "verifications": tuple(v.contribution_id for v in verifications),
        "judge": decision.contribution_id,
    }
    _validate_provenance(record.curator_provenance, payload, "curator")
    if record.record_id != sha256_json({"payload": payload, "provenance": record.curator_provenance}):
        raise AcceptanceError("curated record identity mismatch")


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
        parent_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        if not dataset_name or not dataset_version:
            raise ValueError("dataset_name and dataset_version are required")
        if parent_manifest is None:
            if parent_manifest_sha256 not in (None, _GENESIS_MANIFEST_SHA256):
                raise ValueError(
                    "non-genesis parent requires the complete parent_manifest for hash validation"
                )
            parent = _GENESIS_MANIFEST_SHA256
        else:
            parent = _validated_parent_manifest_sha256(
                parent_manifest,
                dataset_name=dataset_name,
            )
            if parent_manifest_sha256 is not None and parent_manifest_sha256 != parent:
                raise ValueError("parent_manifest_sha256 does not match parent_manifest")
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
            "schema": _MANIFEST_SCHEMA,
            "worker_id": CONVERGENCE_WORKER_ID,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_revision": dataset_revision,
            "parent_manifest_sha256": parent_manifest_sha256,
            "classification": DATASET_CLASSIFICATION,
            "base_corpus_evidence": False,
            "canonical_base_training_eligible": False,
            "training_use": _TRAINING_USE,
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
        _validate_chain_integrity(task, student, proposals, critic, verifications, decision)
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
            "training_use": _TRAINING_USE,
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
            _TRAINING_USE,
            task,
            student,
            tuple(proposals),
            critic,
            tuple(verifications),
            decision,
            prov,
        )
        _validate_record_postconditions(
            record,
            task=task,
            student=student,
            proposals=proposals,
            critic=critic,
            verifications=verifications,
            decision=decision,
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
        evidence_revision = getattr(verifier, "evidence_revision", None)
        if not _is_sha256(evidence_revision):
            raise ValueError("deterministic verifier must expose a valid evidence_revision SHA-256")
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
            request = VerificationRequest(task.task_id, task.prompt, answer, proposal, critic)
            result = self.verifier.verify(request)
            current_revision = getattr(self.verifier, "evidence_revision", None)
            if not _is_sha256(current_revision):
                raise FactoryError("deterministic verifier evidence revision became invalid")
            if result.evidence_revision != current_revision:
                raise FactoryError("stale deterministic verifier evidence revision")
            expected_subject_sha256 = verification_subject_sha256(request)
            if result.subject_sha256 != expected_subject_sha256:
                raise FactoryError("forged deterministic verification subject binding")
            payload = {
                "task_id": task.task_id,
                "proposal_id": proposal.contribution_id,
                "critic_review_id": critic.contribution_id,
                "verifier_id": self.verifier.adapter_id,
                "status": result.status,
                "evidence": result.evidence,
                "evidence_revision": result.evidence_revision,
                "subject_sha256": result.subject_sha256,
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
                    result.evidence_revision,
                    result.subject_sha256,
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
            _validate_record_postconditions(
                record,
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
