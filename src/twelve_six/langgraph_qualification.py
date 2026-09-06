"""Fail-closed LangGraph qualification and project-owned task-state contract."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/langchain-ai/langgraph"
UPSTREAM_TAG = "1.2.11"
UPSTREAM_COMMIT = "644815f9e5bc52ad8f7a5227a456227e9c3e639b"
UPSTREAM_LICENSE = "MIT"
UPSTREAM_LICENSE_BLOB = "fc0602feecdd6748623c852ab534e1ca612673c7"
PYPI_PACKAGE = "langgraph==1.2.11"
PYPI_SDIST_SHA256 = "9ecfe11e50d338b34b15cf4d8a442642de103e8ae6971320efba84e4542eb363"
UPSTREAM_LIB_PYPROJECT_BLOB = "e3d95cfec83fb9ce4a887c264a060f4513be2e9d"
UPSTREAM_LIB_UV_LOCK_BLOB = "a3d49f446adc664140e8fca0a8f6318be05a8dd7"

ALLOWED_STATUS = {"READY", "RUNNING", "BLOCKED", "COMPLETED", "FAILED"}
TASK_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ContractError(ValueError):
    """Raised when project-owned task state violates the contract."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_task_state(state: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "task_id", "status", "goal", "completed_steps", "pending_steps", "evidence", "checkpoint_seq"}
    if set(state) != required:
        missing = sorted(required - set(state))
        extra = sorted(set(state) - required)
        raise ContractError(f"task_state_keys mismatch missing={missing} extra={extra}")
    if state["schema_version"] != 1:
        raise ContractError("unsupported task-state schema_version")
    if not isinstance(state["task_id"], str) or not TASK_ID_RE.fullmatch(state["task_id"]):
        raise ContractError("invalid task_id")
    if state["status"] not in ALLOWED_STATUS:
        raise ContractError("invalid task status")
    if not isinstance(state["goal"], str) or not state["goal"]:
        raise ContractError("goal must be non-empty")
    for key in ("completed_steps", "pending_steps", "evidence"):
        if not isinstance(state[key], list) or any(not isinstance(x, str) for x in state[key]):
            raise ContractError(f"{key} must be list[str]")
    if not isinstance(state["checkpoint_seq"], int) or state["checkpoint_seq"] < 0:
        raise ContractError("checkpoint_seq must be a non-negative integer")
    if len(set(state["completed_steps"])) != len(state["completed_steps"]):
        raise ContractError("duplicate completed step")
    if set(state["completed_steps"]) & set(state["pending_steps"]):
        raise ContractError("step cannot be both completed and pending")
    return state


def atomic_write_json(path: Path, value: dict[str, Any]) -> str:
    payload = canonical_json(validate_task_state(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return sha256_bytes(payload)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = json.loads(handle.read().decode("utf-8"))
    return validate_task_state(data)


def project_transition(state: dict[str, Any]) -> dict[str, Any]:
    validate_task_state(state)
    if state["status"] != "READY":
        raise ContractError("fixture transition expects READY")
    if not state["pending_steps"]:
        return {**state, "status": "COMPLETED", "checkpoint_seq": state["checkpoint_seq"] + 1}
    step = state["pending_steps"][0]
    return {
        **state,
        "status": "RUNNING",
        "completed_steps": [*state["completed_steps"], step],
        "pending_steps": state["pending_steps"][1:],
        "evidence": [*state["evidence"], f"done:{step}"],
        "checkpoint_seq": state["checkpoint_seq"] + 1,
    }


def benchmark_project_checkpoint(iterations: int = 200) -> dict[str, float | int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    with tempfile.TemporaryDirectory(prefix="twelve-six-langgraph-bench-") as tmp:
        path = Path(tmp) / "task.json"
        state = {
            "schema_version": 1,
            "task_id": "bench-task",
            "status": "READY",
            "goal": "bounded checkpoint benchmark",
            "completed_steps": [],
            "pending_steps": ["prepare"],
            "evidence": [],
            "checkpoint_seq": 0,
        }
        start = perf_counter()
        for i in range(iterations):
            candidate = {**state, "checkpoint_seq": i}
            atomic_write_json(path, candidate)
            state = read_json(path)
        elapsed = perf_counter() - start
    return {"iterations": iterations, "elapsed_seconds": elapsed, "ops_per_second": iterations / elapsed}


def environment_snapshot() -> dict[str, Any]:
    commands = {}
    for name in ("python", "pip", "uv", "poetry", "pdm", "conda", "git", "nvidia-smi"):
        try:
            completed = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=5)
            commands[name] = {"available": completed.returncode == 0, "version": (completed.stdout or completed.stderr).strip()}
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            commands[name] = {"available": False, "version": str(exc)}
    packages = {}
    for package in ("langgraph", "torch", "numpy", "pytest", "safetensors", "ruff", "transformers", "tokenizers"):
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[package] = None
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "commands": commands, "packages": packages}


def run_real_langgraph_probe() -> dict[str, Any]:
    try:
        from typing_extensions import TypedDict
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:
        return {"executed": False, "status": "NOT_EXECUTED", "reason": "LANGGRAPH_IMPORT_UNAVAILABLE", "error": repr(exc)}

    class State(TypedDict):
        value: int

    def increment(state: State) -> dict[str, int]:
        return {"value": state["value"] + 1}

    graph = StateGraph(State)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    compiled = graph.compile()
    start = perf_counter()
    first = compiled.invoke({"value": 0})
    elapsed = perf_counter() - start
    second = compiled.invoke({"value": 0})
    return {"executed": True, "status": "COMPLETED", "first_output": first, "second_output": second, "deterministic": first == second, "elapsed_seconds": elapsed}
