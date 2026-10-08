from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Mapping, Protocol, Sequence


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool: str
    arguments: Mapping[str, JsonValue]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool: str
    ok: bool
    output: Mapping[str, JsonValue] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: int = 0


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    calls: tuple[ToolCall, ...]
    summary: str = ""


@dataclass(frozen=True)
class ExecutionRecord:
    trace_id: str
    proposal_id: str
    results: tuple[ToolResult, ...]
    cancelled: bool


@dataclass(frozen=True)
class Verification:
    passed: bool
    code: str
    summary: str
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRun:
    trace_id: str
    proposal: Proposal
    execution: ExecutionRecord
    verification: Verification
    workspace: Path


@dataclass(frozen=True)
class Selection:
    selected_index: int | None
    reason: str


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


class Proposer(Protocol):
    def propose(self, *, goal: str, trace_id: str, tool_names: Sequence[str]) -> Proposal: ...


class Tool(Protocol):
    @property
    def name(self) -> str: ...

    def execute(
        self,
        call: ToolCall,
        *,
        workspace: Path,
        cancellation: CancellationToken,
    ) -> ToolResult: ...


class Verifier(Protocol):
    def verify(
        self,
        *,
        goal: str,
        workspace: Path,
        execution: ExecutionRecord,
        cancellation: CancellationToken,
    ) -> Verification: ...


class Selector(Protocol):
    def select(self, candidates: Sequence[CandidateRun]) -> Selection: ...


class BrowserMCPAdapter(Protocol):
    def call(
        self,
        method: str,
        arguments: Mapping[str, JsonValue],
        *,
        timeout_seconds: float,
        cancellation: CancellationToken,
    ) -> Mapping[str, JsonValue]: ...
