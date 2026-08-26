from __future__ import annotations

from typing import Mapping, Sequence

from .contracts import (
    StudentAnswer,
    TaskRecord,
    TeacherProposal,
    TeacherRequest,
    TeacherResponse,
    VerificationRequest,
    VerificationResponse,
    VerificationStatus,
)


class MockStudent:
    adapter_id = "mock-student-v1"

    def __init__(self, answer: str) -> None:
        self.answer_text = answer

    def answer(self, task_id: str, prompt: str) -> str:
        del task_id, prompt
        return self.answer_text


class _StaticTeacher:
    def __init__(
        self,
        adapter_id: str,
        critique: str,
        proposed_answer: str,
        confidence: float,
    ) -> None:
        self.adapter_id = adapter_id
        self.response = TeacherResponse(
            critique,
            proposed_answer,
            confidence,
            f"local-fixture:{adapter_id}",
        )

    def propose(self, request: TeacherRequest) -> TeacherResponse:
        del request
        return self.response


class CorrectTeacher(_StaticTeacher):
    def __init__(self, answer: str = "4") -> None:
        super().__init__(
            "mock-correct-teacher-v1",
            "Student answer does not match the deterministic result.",
            answer,
            0.95,
        )


class ConfidentlyWrongTeacher(_StaticTeacher):
    def __init__(self, answer: str = "5") -> None:
        super().__init__(
            "mock-confidently-wrong-teacher-v1",
            "I am certain this unsupported answer is correct.",
            answer,
            1.0,
        )


class DisagreeingTeacherA(_StaticTeacher):
    def __init__(self, answer: str = "4") -> None:
        super().__init__(
            "mock-disagreeing-teacher-a-v1",
            "Teacher A proposes one candidate.",
            answer,
            0.8,
        )


class DisagreeingTeacherB(_StaticTeacher):
    def __init__(self, answer: str = "5") -> None:
        super().__init__(
            "mock-disagreeing-teacher-b-v1",
            "Teacher B proposes a conflicting candidate.",
            answer,
            0.9,
        )


class MockCritic:
    adapter_id = "mock-critic-v1"

    def review(
        self,
        task: TaskRecord,
        student: StudentAnswer,
        proposals: Sequence[TeacherProposal],
    ) -> str:
        del task, student
        if len({p.proposed_answer for p in proposals}) > 1:
            return "Teacher disagreement detected; deterministic verification must resolve it."
        return "Teacher proposals agree; deterministic verification is still mandatory."


class ExactMatchVerifier:
    adapter_id = "mock-deterministic-exact-match-verifier-v1"

    def __init__(self, expected_by_task: Mapping[str, str]) -> None:
        self.expected_by_task = dict(expected_by_task)

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        expected = self.expected_by_task.get(request.task_id)
        source = f"local-fixture:exact-match:{request.task_id}"
        if expected is None:
            return VerificationResponse(
                VerificationStatus.INCONCLUSIVE,
                "No deterministic expected value is registered for this task.",
                source,
            )
        if request.proposal.proposed_answer == expected:
            return VerificationResponse(
                VerificationStatus.SUPPORTED,
                f"Exact deterministic match to registered value {expected!r}.",
                source,
            )
        return VerificationResponse(
            VerificationStatus.CONTRADICTED,
            f"Registered deterministic value is {expected!r}, not "
            f"{request.proposal.proposed_answer!r}.",
            source,
        )
