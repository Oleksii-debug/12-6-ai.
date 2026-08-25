from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from twelve_six.agent_runtime import (
    AgentRuntime,
    BrowserMCPTool,
    CancellationToken,
    DeterministicMockMCP,
    Executor,
    FileTool,
    PassingFirstSelector,
    Proposal,
    TerminalPolicy,
    TerminalTool,
    ToolCall,
    ToolRegistry,
    Verification,
)
from twelve_six.agent_runtime.toy_workflow import ScriptedToyProposer, ToyVerifier


class _FixedProposer:
    def __init__(self, proposal: Proposal) -> None:
        self.proposal = proposal

    def propose(self, *, goal: str, trace_id: str, tool_names: tuple[str, ...]) -> Proposal:
        del goal, trace_id, tool_names
        return self.proposal


class _PassVerifier:
    def verify(self, *, goal, workspace, execution, cancellation):
        del goal, workspace, execution, cancellation
        return Verification(True, "ok", "verified")


def test_workspace_escape_is_rejected(tmp_path: Path) -> None:
    call = ToolCall("escape", "files", {"op": "write_text", "path": "../escape.txt", "text": "x"})
    result = FileTool().execute(call, workspace=tmp_path, cancellation=CancellationToken())
    assert not result.ok
    assert result.error_code == "files_error"
    assert not (tmp_path.parent / "escape.txt").exists()


def test_terminal_policy_rejects_unlisted_program(tmp_path: Path) -> None:
    terminal = TerminalTool(TerminalPolicy(frozenset({Path(sys.executable).name})))
    call = ToolCall("blocked", "terminal", {"argv": ["definitely-not-allowed"]})
    result = terminal.execute(call, workspace=tmp_path, cancellation=CancellationToken())
    assert not result.ok
    assert result.error_code == "terminal_error"


def test_terminal_timeout_is_structured(tmp_path: Path) -> None:
    terminal = TerminalTool(
        TerminalPolicy(frozenset({Path(sys.executable).name}), max_timeout_seconds=0.2, poll_seconds=0.01)
    )
    call = ToolCall(
        "slow",
        "terminal",
        {"argv": [sys.executable, "-c", "import time; time.sleep(2)"], "timeout_seconds": 0.05},
    )
    result = terminal.execute(call, workspace=tmp_path, cancellation=CancellationToken())
    assert not result.ok
    assert result.error_code == "timeout"


def test_terminal_cancellation_is_structured(tmp_path: Path) -> None:
    terminal = TerminalTool(
        TerminalPolicy(frozenset({Path(sys.executable).name}), max_timeout_seconds=2, poll_seconds=0.01)
    )
    token = CancellationToken()
    call = ToolCall(
        "cancel",
        "terminal",
        {"argv": [sys.executable, "-c", "import time; time.sleep(2)"], "timeout_seconds": 2},
    )
    result_box = []

    def run() -> None:
        result_box.append(terminal.execute(call, workspace=tmp_path, cancellation=token))

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.05)
    token.cancel()
    thread.join(timeout=2)
    assert result_box
    assert result_box[0].error_code == "cancelled"


def test_deterministic_browser_mcp_seam(tmp_path: Path) -> None:
    tool = BrowserMCPTool(DeterministicMockMCP({"page.title": {"title": "fixture"}}))
    call = ToolCall("mcp", "browser_mcp", {"method": "page.title", "arguments": {}})
    result = tool.execute(call, workspace=tmp_path, cancellation=CancellationToken())
    assert result.ok
    assert result.output == {"title": "fixture"}


def test_executor_stops_after_failed_tool_and_logs_trace(tmp_path: Path) -> None:
    registry = ToolRegistry([FileTool()])
    proposal = Proposal(
        "p1",
        (
            ToolCall("bad", "missing", {}),
            ToolCall("never", "files", {"op": "write_text", "path": "never.txt", "text": "no"}),
        ),
    )
    execution = Executor(registry).execute(
        proposal,
        trace_id="trace-1",
        workspace=tmp_path,
        cancellation=CancellationToken(),
    )
    assert len(execution.results) == 1
    assert execution.results[0].error_code == "unknown_tool"
    log = (tmp_path / ".agent_runtime" / "execution.jsonl").read_text(encoding="utf-8")
    payload = json.loads(log)
    assert payload["trace_id"] == "trace-1"
    assert payload["proposal_id"] == "p1"


def test_selector_requires_independent_verification(tmp_path: Path) -> None:
    proposal = Proposal("p", ())
    runtime = AgentRuntime(executor=Executor(ToolRegistry([FileTool()])), verifier=_PassVerifier())
    candidate = runtime.run_candidate(
        goal="noop",
        proposer=_FixedProposer(proposal),
        workspace=tmp_path,
        trace_id="trace",
    )
    assert PassingFirstSelector().select([candidate]).selected_index == 0
    record = runtime.build_dataset_record(candidate)
    assert record["training_authority"] == "DATASET_RECORD_ONLY_NOT_MODEL_UPDATE"


def test_real_isolated_toy_development_workflow(tmp_path: Path) -> None:
    if __import__("shutil").which("git") is None:
        pytest.skip("git is required for the toy Git workflow")
    registry = ToolRegistry(
        [
            FileTool(),
            TerminalTool(TerminalPolicy(frozenset({Path(sys.executable).name}), max_timeout_seconds=10)),
            __import__("twelve_six.agent_runtime", fromlist=["GitTool"]).GitTool(),
        ]
    )
    runtime = AgentRuntime(executor=Executor(registry), verifier=ToyVerifier())
    candidate = runtime.run_candidate(
        goal="Create a tested integer addition module.",
        proposer=ScriptedToyProposer(),
        workspace=tmp_path,
        trace_id="toy-test-trace",
    )
    assert candidate.verification.passed
    assert (tmp_path / "calculator.py").is_file()
    assert (tmp_path / "test_calculator.py").is_file()
    assert (tmp_path / ".git").is_dir()
    assert len(candidate.execution.results) == 5
    assert all(result.ok for result in candidate.execution.results)
