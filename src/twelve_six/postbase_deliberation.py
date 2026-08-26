from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Budget:
    wall_seconds: float | None = None
    model_calls: int | None = None
    generated_tokens: int | None = None
    tool_calls: int | None = None
    candidate_branches: int | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds is not None and self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive")
        for name in ("model_calls", "generated_tokens", "candidate_branches"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tool_calls is not None and self.tool_calls < 0:
            raise ValueError("tool_calls must be non-negative")
        if self.wall_seconds is None and self.model_calls is None and self.generated_tokens is None:
            raise ValueError("wall, model-call, or generated-token limiter required")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Request:
    task: str
    stage: str
    branch_id: str
    candidate_id: str
    iteration: int
    current_text: str | None = None
    critique: str | None = None
    tool_results: tuple[str, ...] = ()
    max_generated_tokens: int | None = None
    max_tool_calls: int | None = None
    deadline_monotonic: float | None = None


@dataclass(frozen=True)
class Response:
    text: str
    generated_tokens: int
    private_scratch: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Verification:
    score: float
    confidence: float
    summary: str = ""

    def __post_init__(self) -> None:
        for name in ("score", "confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")


class ModelAdapter(Protocol):
    def generate(self, request: Request) -> Response:
        ...


class Verifier(Protocol):
    def evaluate(
        self,
        task: str,
        text: str,
        branch_id: str,
        iteration: int,
    ) -> Verification:
        ...


class ToolExecutor(Protocol):
    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class Config:
    initial_branches: int = 2
    target_score: float | None = 0.95
    min_confidence: float = 0.90
    convergence_delta: float = 1e-4
    convergence_rounds: int = 2

    def __post_init__(self) -> None:
        if self.initial_branches <= 0 or self.convergence_rounds <= 0:
            raise ValueError("branch and convergence counts must be positive")
        if self.target_score is not None and not 0 <= self.target_score <= 1:
            raise ValueError("target_score must be in [0,1]")
        if not 0 <= self.min_confidence <= 1 or self.convergence_delta < 0:
            raise ValueError("invalid confidence or convergence delta")


class AdapterContractError(RuntimeError):
    pass


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


@dataclass
class _Candidate:
    id: str
    branch: str
    parent: str | None
    iteration: int
    text: str
    scratch: str
    verification: Verification


class DeliberationController:
    """Model-agnostic, synchronous, budgeted post-Base deliberation controller."""

    worker_id = "POSTBASE-255-DELIBERATION-CONTROLLER-V1"
    schema = "12-6.postbase-deliberation-trace.v1"

    def __init__(
        self,
        model: ModelAdapter,
        verifier: Verifier,
        *,
        tools: ToolExecutor | None = None,
        config: Config | None = None,
    ):
        self.model = model
        self.verifier = verifier
        self.tools = tools
        self.config = config or Config()

    def run(self, task: str, budget: Budget) -> dict[str, Any]:
        if not task.strip():
            raise ValueError("task must be non-empty")
        started = time.monotonic()
        deadline = None if budget.wall_seconds is None else started + budget.wall_seconds
        used = {"model_calls": 0, "generated_tokens": 0, "tool_calls": 0, "candidate_branches": 0}
        trace: dict[str, Any] = {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "budget": asdict(budget),
            "config": asdict(self.config),
            "model_calls": [],
            "tool_calls": [],
            "branches_attempted": [],
            "comparisons": [],
        }
        candidates: list[_Candidate] = []
        next_id = 1

        def reserve() -> str:
            nonlocal next_id
            self._branch_ok(budget, used)
            candidate_id = f"candidate-{next_id}"
            next_id += 1
            used["candidate_branches"] += 1
            return candidate_id

        try:
            for n in range(self.config.initial_branches):
                if budget.candidate_branches is not None and used["candidate_branches"] >= budget.candidate_branches:
                    break
                self._model_ok(budget, used, deadline)
                cid = reserve()
                branch = f"branch-{n + 1}"
                response = self._stage(task, "propose", branch, cid, 0, None, None, budget, used, trace, deadline)
                candidates.append(self._candidate(task, response, cid, branch, None, 0, "propose", trace))

            if not candidates:
                raise RuntimeError("budget exhausted before any candidate completed")

            best = self._compare(candidates, trace)
            if self._target(best):
                reason = "verifier_target"
            else:
                reason = ""
                previous = best.verification.score
                stable = 0
                iteration = 1
                while not reason:
                    self._outer_ok(budget, used, deadline)
                    critique = self._stage(
                        task, "critique", best.branch, best.id, iteration,
                        best.text, None, budget, used, trace, deadline,
                    )
                    self._outer_ok(budget, used, deadline)
                    self._branch_ok(budget, used)
                    self._model_ok(budget, used, deadline)
                    cid = reserve()
                    revision = self._stage(
                        task, "revise", best.branch, cid, iteration,
                        best.text, critique.text, budget, used, trace, deadline,
                    )
                    candidates.append(
                        self._candidate(
                            task, revision, cid, best.branch, best.id,
                            iteration, "revise", trace,
                        )
                    )
                    best = self._compare(candidates, trace)
                    if self._target(best):
                        reason = "verifier_target"
                        break
                    delta = abs(best.verification.score - previous)
                    stable = stable + 1 if delta <= self.config.convergence_delta else 0
                    previous = best.verification.score
                    if stable >= self.config.convergence_rounds:
                        reason = "converged"
                    iteration += 1
        except _Stop as stop:
            if not candidates:
                raise RuntimeError(f"{stop.reason} before candidate completion") from None
            best = self._compare(candidates, trace)
            reason = stop.reason

        trace["budget_consumed"] = {**used, "wall_seconds": time.monotonic() - started}
        trace["stop_reason"] = reason
        trace["selected_final_candidate"] = best.id
        return {
            "final_text": best.text,
            "score": best.verification.score,
            "confidence": best.verification.confidence,
            "trace": trace,
        }

    def _stage(
        self, task: str, stage: str, branch: str, cid: str, iteration: int,
        current: str | None, critique: str | None, budget: Budget,
        used: dict[str, int], trace: dict[str, Any], deadline: float | None,
    ) -> Response:
        tool_results: tuple[str, ...] = ()
        while True:
            self._model_ok(budget, used, deadline)
            rt = None if budget.generated_tokens is None else budget.generated_tokens - used["generated_tokens"]
            rc = None if budget.tool_calls is None else budget.tool_calls - used["tool_calls"]
            request = Request(
                task, stage, branch, cid, iteration, current, critique,
                tool_results, rt, rc, deadline,
            )
            t0 = time.monotonic()
            response = self.model.generate(request)
            elapsed = time.monotonic() - t0
            if response.generated_tokens < 0 or (rt is not None and response.generated_tokens > rt):
                raise AdapterContractError("adapter exceeded generated-token allowance")
            if rc is not None and len(response.tool_calls) > rc:
                raise AdapterContractError("adapter exceeded tool-call allowance")
            used["model_calls"] += 1
            used["generated_tokens"] += response.generated_tokens
            trace["model_calls"].append({
                "index": used["model_calls"],
                "stage": stage,
                "branch_id": branch,
                "candidate_id": cid,
                "iteration": iteration,
                "generated_tokens": response.generated_tokens,
                "response_sha256": _hash(response.text),
                "private_scratch_sha256": _hash(response.private_scratch),
                "tool_calls_requested": len(response.tool_calls),
                "wall_seconds": elapsed,
            })
            if not response.tool_calls:
                return response
            if self.tools is None:
                raise AdapterContractError("tool requested without tool executor")
            results = list(tool_results)
            for call in response.tool_calls:
                self._wall_ok(deadline)
                self._tool_ok(budget, used)
                t0 = time.monotonic()
                ok, error = True, None
                try:
                    result = self.tools.execute(call.name, call.arguments)
                    if not isinstance(result, str):
                        raise TypeError("tool result must be str")
                except Exception as exc:
                    ok, error = False, type(exc).__name__
                    result = f"[tool_error:{error}]"
                used["tool_calls"] += 1
                trace["tool_calls"].append({
                    "index": used["tool_calls"],
                    "name": call.name,
                    "arguments_sha256": _hash(_json(call.arguments)),
                    "result_sha256": _hash(result),
                    "result_bytes": len(result.encode()),
                    "success": ok,
                    "error_type": error,
                    "wall_seconds": time.monotonic() - t0,
                })
                results.append(result)
            tool_results = tuple(results)

    def _candidate(
        self, task: str, response: Response, cid: str, branch: str,
        parent: str | None, iteration: int, stage: str, trace: dict[str, Any],
    ) -> _Candidate:
        verification = self.verifier.evaluate(task, response.text, branch, iteration)
        candidate = _Candidate(cid, branch, parent, iteration, response.text, response.private_scratch, verification)
        trace["branches_attempted"].append({
            "candidate_id": cid,
            "branch_id": branch,
            "parent_candidate_id": parent,
            "iteration": iteration,
            "stage": stage,
            "text_sha256": _hash(response.text),
            "score": verification.score,
            "confidence": verification.confidence,
            "verifier_summary": verification.summary,
        })
        return candidate

    @staticmethod
    def _compare(candidates: list[_Candidate], trace: dict[str, Any]) -> _Candidate:
        best = max(candidates, key=lambda c: (c.verification.score, c.verification.confidence, -c.iteration, c.id))
        trace["comparisons"].append({
            "round": len(trace["comparisons"]),
            "selected_candidate_id": best.id,
            "selected_score": best.verification.score,
            "rejected_candidate_ids": [c.id for c in candidates if c.id != best.id],
        })
        return best

    def _target(self, candidate: _Candidate) -> bool:
        return (
            self.config.target_score is not None
            and candidate.verification.score >= self.config.target_score
            and candidate.verification.confidence >= self.config.min_confidence
        )

    def _outer_ok(self, budget: Budget, used: dict[str, int], deadline: float | None) -> None:
        self._wall_ok(deadline)
        self._branch_ok(budget, used)
        self._model_ok(budget, used, deadline)

    @staticmethod
    def _wall_ok(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise _Stop("wall_clock_budget")

    def _model_ok(self, budget: Budget, used: dict[str, int], deadline: float | None) -> None:
        self._wall_ok(deadline)
        if budget.model_calls is not None and used["model_calls"] >= budget.model_calls:
            raise _Stop("model_call_budget")
        if budget.generated_tokens is not None and used["generated_tokens"] >= budget.generated_tokens:
            raise _Stop("generated_token_budget")

    @staticmethod
    def _tool_ok(budget: Budget, used: dict[str, int]) -> None:
        if budget.tool_calls is not None and used["tool_calls"] >= budget.tool_calls:
            raise _Stop("tool_call_budget")

    @staticmethod
    def _branch_ok(budget: Budget, used: dict[str, int]) -> None:
        if budget.candidate_branches is not None and used["candidate_branches"] >= budget.candidate_branches:
            raise _Stop("candidate_branch_budget")


class DeterministicMockAdapter:
    """LOCAL_FREE mechanics adapter; not an external teacher."""

    def generate(self, request: Request) -> Response:
        if request.stage == "propose":
            text = f"proposal {request.branch_id}: {request.task}"
        elif request.stage == "critique":
            text = f"critique {request.iteration}: verify assumptions and precision"
        elif request.stage == "revise":
            text = f"{request.current_text or ''} revision-{request.iteration}"
        else:
            raise ValueError(request.stage)
        words = text.split()
        if request.max_generated_tokens is not None:
            words = words[:request.max_generated_tokens]
        return Response(
            " ".join(words),
            len(words),
            f"internal:{request.stage}:{request.branch_id}:{request.iteration}",
        )
