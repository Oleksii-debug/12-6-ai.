from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from twelve_six.postbase_deliberation import (
    Budget,
    Config,
    DeliberationController,
    Request,
    Response,
    ToolCall,
    Verification,
)

from .authority import CapabilityGate, CapabilityUnavailableError

TRACE_SCHEMA = "12-6.local-assistant-orchestration.v1"
WORKER_ID = "NEXT100-093-LOCAL-ASSISTANT-ORCHESTRATOR"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class RunOptions:
    checkpoint: str | None = None
    expected_model_spec_sha256: str | None = None
    mock_model: bool = False
    use_hypothesis_search: bool = False
    memory_db: str | None = None
    use_mock_tools: bool = False
    max_model_calls: int = 4
    max_generated_tokens: int = 256
    max_tool_calls: int = 2
    candidate_branches: int = 2
    max_new_tokens_per_call: int = 64
    expected_answer_fixture: str | None = None

    def __post_init__(self) -> None:
        if self.mock_model and self.checkpoint is not None:
            raise ValueError("choose either mock_model or checkpoint, not both")
        if not self.mock_model and self.checkpoint is None:
            raise ValueError("checkpoint is required unless mock_model=True")
        for name in (
            "max_model_calls",
            "max_generated_tokens",
            "candidate_branches",
            "max_new_tokens_per_call",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    text: str
    trace: dict[str, Any]


class CandidateEvidenceBuilder(Protocol):
    def build(self, task: str, candidate: str, candidate_id: str) -> object: ...


class StrictVerifierBridge:
    """Explicit categorical-to-ranking policy over POSTBASE-357.

    The bridge never treats INCONCLUSIVE as positive evidence. PASS receives a
    positive ranking only when the ensemble reports deterministic correctness
    support. FAIL/CONFLICT are hard rejection scores. This policy is intentionally
    visible here rather than silently weakening POSTBASE-357 semantics.
    """

    def __init__(self, *, expected_answer_fixture: str | None = None) -> None:
        from twelve_six.postbase.verification import (
            Claim,
            ExactAnswerFixture,
            ExactAnswerFixtureVerifier,
            VerificationRequest,
            VerificationStatus,
            VerifierEnsemble,
        )

        self._Claim = Claim
        self._ExactAnswerFixture = ExactAnswerFixture
        self._VerificationRequest = VerificationRequest
        self._VerificationStatus = VerificationStatus
        self._ensemble = VerifierEnsemble((ExactAnswerFixtureVerifier(),))
        self._expected = expected_answer_fixture
        self.records: list[dict[str, object]] = []

    def evaluate(self, task: str, text: str, branch_id: str, iteration: int) -> Verification:
        del task
        claim_id = f"{branch_id}:iteration:{iteration}"
        fixtures = ()
        if self._expected is not None:
            fixtures = (
                self._ExactAnswerFixture(
                    claim_id=claim_id,
                    expected=self._expected,
                    actual=text,
                ),
            )
        request = self._VerificationRequest(
            claims=(self._Claim(claim_id=claim_id, text=text),),
            exact_fixtures=fixtures,
        )
        result = self._ensemble.verify(request)
        claim = result.claim(claim_id)
        status = result.status
        if status is self._VerificationStatus.PASS and claim.deterministic_correctness_pass:
            score, confidence = 1.0, 1.0
        elif status in {self._VerificationStatus.FAIL, self._VerificationStatus.CONFLICT}:
            score, confidence = 0.0, 1.0
        else:
            score, confidence = 0.0, 0.0
        self.records.append(
            {
                "claim_id": claim_id,
                "status": status.value,
                "deterministic_correctness_pass": claim.deterministic_correctness_pass,
                "deterministic_failure": claim.deterministic_failure,
                "reason_codes": [item.value for item in claim.reason_codes],
                "candidate_sha256": _sha256_text(text),
                "fixture_only": self._expected is not None,
            }
        )
        return Verification(score=score, confidence=confidence, summary=status.value)


class PostBaseDeliberationModelBridge:
    """Read-only bridge from POSTBASE-255 requests to terminal POSTBASE-351."""

    def __init__(
        self,
        checkpoint: str,
        *,
        expected_model_spec_sha256: str | None,
        max_new_tokens_per_call: int,
    ) -> None:
        from twelve_six.inference.contracts import GenerationConfig
        from twelve_six.postbase import ControllerGenerationRequest, PostBaseModelAdapter

        self._GenerationConfig = GenerationConfig
        self._ControllerGenerationRequest = ControllerGenerationRequest
        self._adapter = PostBaseModelAdapter.from_checkpoint(
            checkpoint,
            expected_model_spec_sha256=expected_model_spec_sha256,
        )
        self._max_new_tokens = max_new_tokens_per_call
        self.post_base_records: list[dict[str, object]] = []

    @property
    def base_evidence(self) -> dict[str, object]:
        return self._adapter.base_evidence.to_dict()

    def generate(self, request: Request) -> Response:
        remaining = request.max_generated_tokens
        max_tokens = self._max_new_tokens if remaining is None else min(self._max_new_tokens, remaining)
        prompt = self._render_prompt(request)
        output = self._adapter.generate(
            self._ControllerGenerationRequest(
                controller="deliberation",
                prompt=prompt,
                config=self._GenerationConfig(max_new_tokens=max_tokens, sample=False),
            )
        )
        self.post_base_records.append(output.post_base_evidence.to_dict())
        return Response(
            text=output.generation.text,
            generated_tokens=len(output.generation.generated_token_ids),
            private_scratch="",
            tool_calls=(),
        )

    @staticmethod
    def _render_prompt(request: Request) -> str:
        payload = {
            "task": request.task,
            "stage": request.stage,
            "branch_id": request.branch_id,
            "candidate_id": request.candidate_id,
            "iteration": request.iteration,
            "current_text": request.current_text,
            "critique": request.critique,
            "tool_results": list(request.tool_results),
        }
        return "POSTBASE_ORCHESTRATION_V1\n" + _canonical_json(payload)


class DeterministicMockModel:
    """Project-owned mechanics fixture; not a learned assistant or quality claim."""

    def generate(self, request: Request) -> Response:
        if request.task.startswith("calc:") and request.stage == "propose":
            expression = request.task[len("calc:") :].strip()
            if not request.tool_results:
                return Response(
                    text="calculator requested",
                    generated_tokens=2,
                    tool_calls=(ToolCall("calculator", {"expression": expression}),),
                )
            result = json.loads(request.tool_results[-1])
            output = result.get("output") if isinstance(result, dict) else None
            value = output.get("value") if isinstance(output, dict) else None
            return Response(text=str(value), generated_tokens=1)
        if request.stage == "critique":
            return Response(text="deterministic fixture critique", generated_tokens=3)
        if request.stage == "revise" and request.current_text is not None:
            return Response(text=request.current_text, generated_tokens=1)
        return Response(text=request.task, generated_tokens=max(1, len(request.task.split())))


class MockToolBridge:
    """POSTBASE-255 ToolExecutor seam backed only by POSTBASE-254 MockExecutor."""

    def __init__(self) -> None:
        from twelve_six.postbase.tool_protocol import MockExecutor, parse_tool_request

        self._executor = MockExecutor()
        self._parse_tool_request = parse_tool_request
        self.records: list[dict[str, object]] = []
        self._ordinal = 0

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        self._ordinal += 1
        request_id = f"next100-093-{self._ordinal:04d}"
        request = self._parse_tool_request(
            {
                "protocol_version": 1,
                "request_id": request_id,
                "tool_name": name,
                "arguments": arguments,
                "timeout_ms": 1000,
                "max_output_bytes": 65536,
            }
        )
        result = self._executor.execute(request)
        record = result.to_dict()
        self.records.append(
            {
                "request_id": request_id,
                "tool_name": name,
                "ok": result.ok,
                "request_sha256": result.provenance.request_sha256,
                "output_sha256": result.provenance.output_sha256,
                "training_eligible": result.provenance.training_eligible,
                "weight_update_eligible": result.provenance.weight_update_eligible,
            }
        )
        return _canonical_json(record)


class LocalAssistantOrchestrator:
    def __init__(self, gate: CapabilityGate | None = None) -> None:
        self.gate = gate or CapabilityGate()

    def run(self, task: str, options: RunOptions) -> OrchestrationResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        task = task.strip()
        trace: dict[str, Any] = {
            "schema": TRACE_SCHEMA,
            "worker_id": WORKER_ID,
            "execution_profile": "LOCAL_FREE",
            "task_sha256": _sha256_text(task),
            "task_bytes": len(task.encode("utf-8")),
            "authorities": self.gate.snapshot(),
            "requested_capabilities": {
                "model_adapter": not options.mock_model,
                "deliberation": True,
                "verifier": True,
                "hypothesis_search": options.use_hypothesis_search,
                "memory_rag": options.memory_db is not None,
                "mock_tools": options.use_mock_tools,
            },
            "safety": {
                "base_weights_modified": False,
                "training_executed": False,
                "optimizer_updates": 0,
                "external_llm_used": False,
                "paid_compute": False,
                "chat_personality_claim": False,
            },
        }

        # Terminal requirements are checked before any model/tool/memory execution.
        self.gate.require("deliberation")
        self.gate.require("verifier")
        if options.use_hypothesis_search:
            self.gate.require("hypothesis_search")
        if options.memory_db is not None:
            self.gate.require("memory_rag")
        if options.use_mock_tools:
            self.gate.require("mock_tools")

        memory_result: object | None = None
        if options.memory_db is not None:
            memory_result = self._retrieve_memory(task, options.memory_db)
            trace["memory"] = memory_result

        hypothesis = None
        if options.use_hypothesis_search:
            hypothesis = self._new_hypothesis_search(task)
            trace["hypothesis_before"] = hypothesis.export()

        if options.mock_model:
            model: Any = DeterministicMockModel()
            trace["model"] = {"kind": "deterministic_mock_fixture", "base_checkpoint_consumed": False}
        else:
            self.gate.require("model_adapter")
            model = PostBaseDeliberationModelBridge(
                str(options.checkpoint),
                expected_model_spec_sha256=options.expected_model_spec_sha256,
                max_new_tokens_per_call=options.max_new_tokens_per_call,
            )
            trace["model"] = {
                "kind": "terminal_postbase351_checkpoint_adapter",
                "base_checkpoint_consumed": True,
                "base_evidence": model.base_evidence,
            }

        verifier = StrictVerifierBridge(expected_answer_fixture=options.expected_answer_fixture)
        tools = MockToolBridge() if options.use_mock_tools else None
        controller = DeliberationController(
            model,
            verifier,
            tools=tools,
            config=Config(initial_branches=options.candidate_branches, target_score=1.0),
        )
        deliberation = controller.run(
            task,
            Budget(
                model_calls=options.max_model_calls,
                generated_tokens=options.max_generated_tokens,
                tool_calls=options.max_tool_calls if options.use_mock_tools else 0,
                candidate_branches=options.candidate_branches,
            ),
        )
        text = str(deliberation["final_text"])

        trace["verifier"] = {
            "policy": "PASS+deterministic correctness=>1/1; FAIL/CONFLICT=>0/1; INCONCLUSIVE=>0/0",
            "records": verifier.records,
        }
        trace["deliberation"] = deliberation["trace"]
        if tools is not None:
            trace["mock_tools"] = tools.records
        if not options.mock_model:
            trace["post_base_generation_evidence"] = model.post_base_records
        if hypothesis is not None:
            if text:
                best = hypothesis.best()
                if best is not None:
                    hypothesis.revise(best.id, text, initial_score=best.score)
            trace["hypothesis_after"] = hypothesis.export()
        trace["output_sha256"] = _sha256_text(text)
        trace["output_bytes"] = len(text.encode("utf-8"))
        return OrchestrationResult(text=text, trace=trace)

    @staticmethod
    def _new_hypothesis_search(task: str):
        from twelve_six.postbase_hypothesis import HypothesisSearch

        search = HypothesisSearch()
        search.propose(task, initial_score=0.5)
        return search

    @staticmethod
    def _retrieve_memory(task: str, memory_db: str) -> dict[str, object]:
        from twelve_six.memory_rag import LexicalRetriever, MemoryDatabase

        database = MemoryDatabase(Path(memory_db))
        try:
            result = LexicalRetriever(database).retrieve(task)
            return {
                "query_sha256": _sha256_text(result.query),
                "evidence": [
                    {
                        "memory_id": item.memory_id,
                        "content_hash": item.content_hash,
                        "source_id": item.provenance.source_id,
                        "source_version": item.provenance.source_version,
                        "version": item.version,
                        "verification": item.verification.value,
                        "lexical_score": item.lexical_score,
                        "supersedes": list(item.supersedes),
                        "superseded_by": list(item.superseded_by),
                    }
                    for item in result.evidence
                ],
                "conflicts": [asdict(item) for item in result.conflicts],
            }
        finally:
            database.close()


def write_trace(path: str | Path, trace: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CapabilityUnavailableError",
    "LocalAssistantOrchestrator",
    "OrchestrationResult",
    "RunOptions",
    "TRACE_SCHEMA",
    "WORKER_ID",
    "write_trace",
]
