from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .contracts import (
    CandidateRun,
    CancellationToken,
    ExecutionRecord,
    JsonValue,
    Proposal,
    Proposer,
    Selection,
    Selector,
    Tool,
    ToolResult,
    Verification,
    Verifier,
)
from .workspace import ensure_workspace


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if not self._tools:
            raise ValueError("at least one tool is required")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)


class Executor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        proposal: Proposal,
        *,
        trace_id: str,
        workspace: Path,
        cancellation: CancellationToken,
    ) -> ExecutionRecord:
        root = ensure_workspace(workspace)
        results: list[ToolResult] = []
        for call in proposal.calls:
            if cancellation.cancelled:
                break
            tool = self.registry.get(call.tool)
            if tool is None:
                result = ToolResult(
                    call_id=call.call_id,
                    tool=call.tool,
                    ok=False,
                    error_code="unknown_tool",
                    error_message="tool is not registered",
                )
            else:
                result = tool.execute(call, workspace=root, cancellation=cancellation)
            results.append(result)
            self._append_log(root, trace_id, proposal.proposal_id, result)
            if not result.ok:
                break
        return ExecutionRecord(
            trace_id=trace_id,
            proposal_id=proposal.proposal_id,
            results=tuple(results),
            cancelled=cancellation.cancelled,
        )

    @staticmethod
    def _append_log(workspace: Path, trace_id: str, proposal_id: str, result: ToolResult) -> None:
        log_dir = workspace / ".agent_runtime"
        log_dir.mkdir(exist_ok=True)
        record = {
            "schema": "12-6.agent-execution-log.v1",
            "trace_id": trace_id,
            "proposal_id": proposal_id,
            "result": asdict(result),
        }
        with (log_dir / "execution.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


class PassingFirstSelector:
    def select(self, candidates: Sequence[CandidateRun]) -> Selection:
        for index, candidate in enumerate(candidates):
            if candidate.verification.passed:
                return Selection(selected_index=index, reason="first independently verified candidate")
        return Selection(selected_index=None, reason="no candidate passed independent verification")


class DatasetBuilder:
    schema = "12-6.agent-episode.v1"

    def build(self, candidate: CandidateRun) -> dict[str, JsonValue]:
        return {
            "schema": self.schema,
            "trace_id": candidate.trace_id,
            "proposal": {
                "proposal_id": candidate.proposal.proposal_id,
                "summary": candidate.proposal.summary,
                "calls": [asdict(call) for call in candidate.proposal.calls],
            },
            "execution": {
                "cancelled": candidate.execution.cancelled,
                "results": [asdict(result) for result in candidate.execution.results],
            },
            "verification": asdict(candidate.verification),
            "training_authority": "DATASET_RECORD_ONLY_NOT_MODEL_UPDATE",
        }


class AgentRuntime:
    def __init__(
        self,
        *,
        executor: Executor,
        verifier: Verifier,
        selector: Selector | None = None,
        dataset_builder: DatasetBuilder | None = None,
    ) -> None:
        self.executor = executor
        self.verifier = verifier
        self.selector = selector or PassingFirstSelector()
        self.dataset_builder = dataset_builder or DatasetBuilder()

    def run_candidate(
        self,
        *,
        goal: str,
        proposer: Proposer,
        workspace: Path,
        trace_id: str | None = None,
        cancellation: CancellationToken | None = None,
    ) -> CandidateRun:
        token = cancellation or CancellationToken()
        trace = trace_id or uuid.uuid4().hex
        proposal = proposer.propose(
            goal=goal,
            trace_id=trace,
            tool_names=self.executor.registry.names,
        )
        execution = self.executor.execute(
            proposal,
            trace_id=trace,
            workspace=workspace,
            cancellation=token,
        )
        if token.cancelled or any(not result.ok for result in execution.results):
            verification = Verification(
                passed=False,
                code="execution_failed",
                summary="executor did not complete a clean tool sequence",
            )
        else:
            verification = self.verifier.verify(
                goal=goal,
                workspace=workspace,
                execution=execution,
                cancellation=token,
            )
        return CandidateRun(
            trace_id=trace,
            proposal=proposal,
            execution=execution,
            verification=verification,
            workspace=workspace.resolve(),
        )

    def select(self, candidates: Sequence[CandidateRun]) -> Selection:
        return self.selector.select(candidates)

    def build_dataset_record(self, candidate: CandidateRun) -> Mapping[str, JsonValue]:
        return self.dataset_builder.build(candidate)
