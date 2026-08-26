from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

WORKER_ID = "POSTBASE-259-TEACHER-STUDENT-DATA-FACTORY-V1"
SCHEMA_VERSION = "12-6.postbase259-teacher-student-factory.v1"
DATASET_CLASSIFICATION = "POSTBASE/EXPERIMENTAL"


class FactoryError(RuntimeError):
    pass


class AcceptanceError(FactoryError):
    pass


class BaseCorpusBoundaryError(FactoryError):
    pass


class Role(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    CRITIC = "critic"
    DETERMINISTIC_VERIFIER = "deterministic_verifier"
    JUDGE = "judge"
    DATASET_CURATOR = "dataset_curator"


class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class Provenance:
    provenance_id: str
    role: Role
    actor_id: str
    source_kind: str
    source_ref: str
    parent_ids: tuple[str, ...]
    payload_sha256: str


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    prompt: str
    provenance: Provenance


@dataclass(frozen=True)
class StudentAnswer:
    contribution_id: str
    task_id: str
    answer: str
    provenance: Provenance


@dataclass(frozen=True)
class TeacherProposal:
    contribution_id: str
    task_id: str
    student_answer_id: str
    teacher_id: str
    critique: str
    proposed_answer: str
    confidence: float
    provenance: Provenance


@dataclass(frozen=True)
class CriticReview:
    contribution_id: str
    task_id: str
    findings: str
    provenance: Provenance


@dataclass(frozen=True)
class VerificationResult:
    contribution_id: str
    task_id: str
    subject_proposal_id: str
    verifier_id: str
    status: VerificationStatus
    evidence: str
    provenance: Provenance


@dataclass(frozen=True)
class JudgeDecision:
    contribution_id: str
    task_id: str
    decision: Decision
    selected_proposal_id: str | None
    rejected_proposal_ids: tuple[str, ...]
    rationale: str
    support_ids: tuple[str, ...]
    provenance: Provenance


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    dataset_name: str
    dataset_version: str
    classification: str
    base_corpus_evidence: bool
    training_use: str
    task: TaskRecord
    student_answer: StudentAnswer
    teacher_proposals: tuple[TeacherProposal, ...]
    critic_review: CriticReview
    verifications: tuple[VerificationResult, ...]
    judge_decision: JudgeDecision
    curator_provenance: Provenance


@dataclass(frozen=True)
class FactoryResult:
    accepted: bool
    reason: str
    record: DatasetRecord | None
    task: TaskRecord
    student_answer: StudentAnswer
    teacher_proposals: tuple[TeacherProposal, ...]
    critic_review: CriticReview
    verifications: tuple[VerificationResult, ...]
    judge_decision: JudgeDecision


@dataclass(frozen=True)
class TeacherRequest:
    task_id: str
    prompt: str
    student_answer: str
    student_answer_id: str


@dataclass(frozen=True)
class TeacherResponse:
    critique: str
    proposed_answer: str
    confidence: float
    source_ref: str


@dataclass(frozen=True)
class VerificationRequest:
    task_id: str
    prompt: str
    student_answer: str
    proposal: TeacherProposal


@dataclass(frozen=True)
class VerificationResponse:
    status: VerificationStatus
    evidence: str
    source_ref: str


class StudentAdapter(Protocol):
    adapter_id: str

    def answer(self, task_id: str, prompt: str) -> str: ...


class TeacherAdapter(Protocol):
    """Future provider adapter boundary; intentionally contains no provider credentials."""

    adapter_id: str

    def propose(self, request: TeacherRequest) -> TeacherResponse: ...


class CriticAdapter(Protocol):
    adapter_id: str

    def review(
        self,
        task: TaskRecord,
        student: StudentAnswer,
        proposals: Sequence[TeacherProposal],
    ) -> str: ...


class DeterministicVerifierAdapter(Protocol):
    adapter_id: str

    def verify(self, request: VerificationRequest) -> VerificationResponse: ...


class JudgeAdapter(Protocol):
    adapter_id: str

    def decide(
        self,
        task: TaskRecord,
        student: StudentAnswer,
        proposals: Sequence[TeacherProposal],
        critic: CriticReview,
        verifications: Sequence[VerificationResult],
    ) -> tuple[Decision, str | None, tuple[str, ...], str, tuple[str, ...]]: ...


class DatasetCuratorAdapter(Protocol):
    adapter_id: str

    def curate(
        self,
        *,
        task: TaskRecord,
        student: StudentAnswer,
        proposals: Sequence[TeacherProposal],
        critic: CriticReview,
        verifications: Sequence[VerificationResult],
        decision: JudgeDecision,
    ) -> DatasetRecord: ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def provenance(
    role: Role,
    actor_id: str,
    source_kind: str,
    source_ref: str,
    parents: Sequence[str],
    payload: Any,
) -> Provenance:
    payload_sha256 = sha256_json(payload)
    body = {
        "role": role,
        "actor_id": actor_id,
        "source_kind": source_kind,
        "source_ref": source_ref,
        "parent_ids": tuple(parents),
        "payload_sha256": payload_sha256,
    }
    return Provenance(
        sha256_json(body),
        role,
        actor_id,
        source_kind,
        source_ref,
        tuple(parents),
        payload_sha256,
    )


def make_task(task_id: str, prompt: str, source_ref: str = "local-task-fixture") -> TaskRecord:
    payload = {"task_id": task_id, "prompt": prompt}
    prov = provenance(
        Role.DATASET_CURATOR,
        "task-intake",
        "LOCAL_TASK",
        source_ref,
        (),
        payload,
    )
    return TaskRecord(task_id, prompt, prov)
