from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import BrowserMCPAdapter, CancellationToken, JsonValue, ToolCall, ToolResult
from .workspace import WorkspaceViolation, resolve_workspace_path


def _result(
    call: ToolCall,
    started: float,
    *,
    ok: bool,
    output: Mapping[str, JsonValue] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool=call.tool,
        ok=ok,
        output=output or {},
        error_code=error_code,
        error_message=error_message,
        duration_ms=max(0, int((time.monotonic() - started) * 1000)),
    )


class FileTool:
    name = "files"

    def execute(
        self,
        call: ToolCall,
        *,
        workspace: Path,
        cancellation: CancellationToken,
    ) -> ToolResult:
        started = time.monotonic()
        if cancellation.cancelled:
            return _result(call, started, ok=False, error_code="cancelled", error_message="cancelled")
        try:
            op = call.arguments.get("op")
            raw_path = call.arguments.get("path")
            if not isinstance(op, str) or not isinstance(raw_path, str):
                raise ValueError("op and path must be strings")
            path = resolve_workspace_path(workspace, raw_path)
            if op == "write_text":
                text = call.arguments.get("text")
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                return _result(
                    call,
                    started,
                    ok=True,
                    output={"path": raw_path, "bytes": len(text.encode("utf-8"))},
                )
            if op == "read_text":
                return _result(
                    call,
                    started,
                    ok=True,
                    output={"path": raw_path, "text": path.read_text(encoding="utf-8")},
                )
            if op == "exists":
                return _result(call, started, ok=True, output={"path": raw_path, "exists": path.exists()})
            raise ValueError(f"unsupported files op: {op}")
        except (OSError, ValueError, WorkspaceViolation) as exc:
            return _result(
                call,
                started,
                ok=False,
                error_code="files_error",
                error_message=str(exc),
            )


@dataclass(frozen=True)
class TerminalPolicy:
    allowed_programs: frozenset[str]
    max_timeout_seconds: float = 30.0
    poll_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not self.allowed_programs:
            raise ValueError("allowed_programs must not be empty")
        if self.max_timeout_seconds <= 0 or self.poll_seconds <= 0:
            raise ValueError("timeouts must be positive")


class TerminalTool:
    name = "terminal"

    def __init__(self, policy: TerminalPolicy) -> None:
        self.policy = policy

    def execute(
        self,
        call: ToolCall,
        *,
        workspace: Path,
        cancellation: CancellationToken,
    ) -> ToolResult:
        started = time.monotonic()
        try:
            argv = call.arguments.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                raise ValueError("argv must be a non-empty list[str]")
            program = Path(argv[0]).name
            if program not in self.policy.allowed_programs:
                raise ValueError(f"program is not allowed: {program}")
            cwd_arg = call.arguments.get("cwd", ".")
            if not isinstance(cwd_arg, str):
                raise ValueError("cwd must be a string")
            cwd = resolve_workspace_path(workspace, cwd_arg)
            if not cwd.is_dir():
                raise ValueError("cwd must exist and be a directory")
            requested_timeout = call.arguments.get("timeout_seconds", self.policy.max_timeout_seconds)
            if isinstance(requested_timeout, bool) or not isinstance(requested_timeout, (int, float)):
                raise ValueError("timeout_seconds must be numeric")
            timeout = float(requested_timeout)
            if timeout <= 0:
                raise ValueError("timeout_seconds must be positive")
            timeout = min(timeout, self.policy.max_timeout_seconds)
            if cancellation.cancelled:
                return _result(call, started, ok=False, error_code="cancelled", error_message="cancelled")

            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                start_new_session=(os.name != "nt"),
            )
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if cancellation.wait(self.policy.poll_seconds):
                    self._terminate(proc)
                    stdout, stderr = proc.communicate()
                    return _result(
                        call,
                        started,
                        ok=False,
                        output={"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode},
                        error_code="cancelled",
                        error_message="command cancelled",
                    )
                if time.monotonic() >= deadline:
                    self._terminate(proc)
                    stdout, stderr = proc.communicate()
                    return _result(
                        call,
                        started,
                        ok=False,
                        output={"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode},
                        error_code="timeout",
                        error_message="command timed out",
                    )
            stdout, stderr = proc.communicate()
            return _result(
                call,
                started,
                ok=proc.returncode == 0,
                output={"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode},
                error_code=None if proc.returncode == 0 else "nonzero_exit",
                error_message=None if proc.returncode == 0 else "command exited non-zero",
            )
        except (OSError, ValueError, WorkspaceViolation) as exc:
            return _result(call, started, ok=False, error_code="terminal_error", error_message=str(exc))

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except OSError:
                pass


class GitTool:
    name = "git"
    _OPS: dict[str, tuple[str, ...]] = {
        "init": ("init", "--quiet"),
        "status": ("status", "--short"),
        "diff": ("diff", "--"),
        "diff_cached": ("diff", "--cached", "--"),
        "add_all": ("add", "--all"),
    }

    def execute(
        self,
        call: ToolCall,
        *,
        workspace: Path,
        cancellation: CancellationToken,
    ) -> ToolResult:
        started = time.monotonic()
        op = call.arguments.get("op")
        if not isinstance(op, str) or op not in self._OPS:
            return _result(
                call,
                started,
                ok=False,
                error_code="git_error",
                error_message="unsupported git op",
            )
        if shutil.which("git") is None:
            return _result(
                call,
                started,
                ok=False,
                error_code="git_unavailable",
                error_message="git executable not found",
            )
        tool = TerminalTool(TerminalPolicy(frozenset({"git"}), max_timeout_seconds=10.0))
        terminal_call = ToolCall(
            call_id=call.call_id,
            tool="terminal",
            arguments={"argv": ["git", *self._OPS[op]], "cwd": ".", "timeout_seconds": 10},
        )
        result = tool.execute(terminal_call, workspace=workspace, cancellation=cancellation)
        return ToolResult(
            call_id=call.call_id,
            tool=call.tool,
            ok=result.ok,
            output=result.output,
            error_code=result.error_code,
            error_message=result.error_message,
            duration_ms=result.duration_ms,
        )


class BrowserMCPTool:
    name = "browser_mcp"

    def __init__(self, adapter: BrowserMCPAdapter, *, max_timeout_seconds: float = 10.0) -> None:
        if max_timeout_seconds <= 0:
            raise ValueError("max_timeout_seconds must be positive")
        self.adapter = adapter
        self.max_timeout_seconds = max_timeout_seconds

    def execute(
        self,
        call: ToolCall,
        *,
        workspace: Path,
        cancellation: CancellationToken,
    ) -> ToolResult:
        del workspace
        started = time.monotonic()
        try:
            method = call.arguments.get("method")
            arguments = call.arguments.get("arguments", {})
            timeout = call.arguments.get("timeout_seconds", self.max_timeout_seconds)
            if not isinstance(method, str) or not isinstance(arguments, dict):
                raise ValueError("method must be string and arguments must be object")
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError("timeout_seconds must be positive numeric")
            if cancellation.cancelled:
                return _result(call, started, ok=False, error_code="cancelled", error_message="cancelled")
            payload = self.adapter.call(
                method,
                arguments,
                timeout_seconds=min(float(timeout), self.max_timeout_seconds),
                cancellation=cancellation,
            )
            return _result(call, started, ok=True, output=payload)
        except (RuntimeError, ValueError) as exc:
            return _result(
                call,
                started,
                ok=False,
                error_code="browser_mcp_error",
                error_message=str(exc),
            )


class DeterministicMockMCP:
    def __init__(self, responses: Mapping[str, Mapping[str, JsonValue]]) -> None:
        self.responses = dict(responses)

    def call(
        self,
        method: str,
        arguments: Mapping[str, JsonValue],
        *,
        timeout_seconds: float,
        cancellation: CancellationToken,
    ) -> Mapping[str, JsonValue]:
        del arguments, timeout_seconds
        if cancellation.cancelled:
            raise RuntimeError("cancelled")
        if method not in self.responses:
            raise RuntimeError(f"no deterministic response for method: {method}")
        return json.loads(json.dumps(self.responses[method], sort_keys=True))
