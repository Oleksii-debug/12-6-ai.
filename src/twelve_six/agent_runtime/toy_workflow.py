from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .contracts import CancellationToken, ExecutionRecord, Proposal, ToolCall, Verification
from .runtime import AgentRuntime, Executor, ToolRegistry
from .tools import FileTool, GitTool, TerminalPolicy, TerminalTool


class ScriptedToyProposer:
    """Deterministic model stand-in. It does not load or modify any model weights."""

    def propose(self, *, goal: str, trace_id: str, tool_names: Sequence[str]) -> Proposal:
        del goal, trace_id
        required = {"files", "terminal", "git"}
        if not required.issubset(tool_names):
            raise RuntimeError("toy proposer requires files, terminal, and git tools")
        calculator = "def add(a: int, b: int) -> int:\n    return a + b\n"
        test = (
            "import unittest\n\n"
            "from calculator import add\n\n\n"
            "class CalculatorTest(unittest.TestCase):\n"
            "    def test_add(self) -> None:\n"
            "        self.assertEqual(add(20, 22), 42)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return Proposal(
            proposal_id="toy-proposal-v1",
            summary="create and test a tiny calculator module inside the isolated workspace",
            calls=(
                ToolCall("git-init", "git", {"op": "init"}),
                ToolCall("write-code", "files", {"op": "write_text", "path": "calculator.py", "text": calculator}),
                ToolCall("write-test", "files", {"op": "write_text", "path": "test_calculator.py", "text": test}),
                ToolCall(
                    "unit-test",
                    "terminal",
                    {"argv": [sys.executable, "-m", "unittest", "-q"], "cwd": ".", "timeout_seconds": 10},
                ),
                ToolCall("git-status", "git", {"op": "status"}),
            ),
        )


class ToyVerifier:
    def verify(
        self,
        *,
        goal: str,
        workspace: Path,
        execution: ExecutionRecord,
        cancellation: CancellationToken,
    ) -> Verification:
        del goal, execution
        if cancellation.cancelled:
            return Verification(False, "cancelled", "verification cancelled")
        code = workspace / "calculator.py"
        test = workspace / "test_calculator.py"
        if not code.is_file() or not test.is_file():
            return Verification(False, "missing_files", "expected development artifacts are missing")
        import subprocess

        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "-q"],
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        return Verification(
            passed=completed.returncode == 0,
            code="tests_pass" if completed.returncode == 0 else "tests_fail",
            summary="independent verifier reran the toy unit test",
            details={"exit_code": completed.returncode},
        )


def build_runtime() -> AgentRuntime:
    allowed = frozenset({Path(sys.executable).name})
    registry = ToolRegistry(
        [
            FileTool(),
            TerminalTool(TerminalPolicy(allowed_programs=allowed, max_timeout_seconds=10.0)),
            GitTool(),
        ]
    )
    return AgentRuntime(executor=Executor(registry), verifier=ToyVerifier())


def run(workspace: Path) -> dict[str, object]:
    runtime = build_runtime()
    candidate = runtime.run_candidate(
        goal="Create a tested integer addition module.",
        proposer=ScriptedToyProposer(),
        workspace=workspace,
        trace_id="toy-dev-trace-v1",
    )
    selection = runtime.select([candidate])
    dataset_record = runtime.build_dataset_record(candidate)
    return {
        "schema": "12-6.agent-toy-development-run.v1",
        "trace_id": candidate.trace_id,
        "workspace": str(candidate.workspace),
        "verification": asdict(candidate.verification),
        "selection": asdict(selection),
        "dataset_record": dataset_record,
        "base_behavior_modified": False,
        "external_model_weights_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated deterministic 12-6 agent toy workflow")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.workspace is not None:
        args.workspace.mkdir(parents=True, exist_ok=True)
        report = run(args.workspace)
    else:
        with tempfile.TemporaryDirectory(prefix="twelve-six-agent-") as tmp:
            report = run(Path(tmp))
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if report["verification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
