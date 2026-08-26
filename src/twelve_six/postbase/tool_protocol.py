from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal, NotRequired, TypedDict


PROTOCOL_VERSION = 1
MAX_TIMEOUT_MS = 60_000
MAX_OUTPUT_BYTES = 64 * 1024
MAX_ARGUMENT_BYTES = 64 * 1024
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class Phase(str, Enum):
    MODEL_GENERATION = "model_generation"
    TOOL_EXECUTION = "tool_execution"
    TOOL_OBSERVATION = "tool_observation"
    FINAL_ANSWER = "final_answer"


class ToolName(str, Enum):
    WEB_SEARCH = "web.search"
    DOCUMENT_RETRIEVAL = "document.retrieve"
    CALCULATOR = "calculator"
    PYTHON_EXECUTION = "python.execute"
    FILESYSTEM_SANDBOX = "filesystem.sandbox"
    FUTURE_API = "api.call"


class ErrorCode(str, Enum):
    MALFORMED_REQUEST = "malformed_request"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    POLICY_DENIED = "policy_denied"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    NOT_FOUND = "not_found"
    EXECUTION_ERROR = "execution_error"


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class WebSearchArguments(TypedDict):
    query: str
    top_k: int


class DocumentRetrievalArguments(TypedDict):
    document_id: str
    query: str | None
    max_chunks: int


class CalculatorArguments(TypedDict):
    expression: str


class PythonExecutionArguments(TypedDict):
    code: str
    inputs: dict[str, JsonValue]


class FilesystemSandboxArguments(TypedDict):
    operation: Literal["read", "write", "list"]
    path: str
    content: NotRequired[str | None]


class FutureApiArguments(TypedDict):
    api_name: str
    operation: str
    params: dict[str, JsonValue]


ToolArguments = (
    WebSearchArguments
    | DocumentRetrievalArguments
    | CalculatorArguments
    | PythonExecutionArguments
    | FilesystemSandboxArguments
    | FutureApiArguments
)


@dataclass(frozen=True)
class ToolError:
    code: ErrorCode
    message: str
    retryable: bool = False

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class Provenance:
    executor_id: str
    adapter_id: str
    request_sha256: str
    source_refs: tuple[str, ...] = ()
    output_sha256: str | None = None
    observed_output_bytes: int = 0
    content_class: str = "tool_observation"
    training_eligible: bool = False
    weight_update_eligible: bool = False

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "executor_id": self.executor_id,
            "adapter_id": self.adapter_id,
            "request_sha256": self.request_sha256,
            "source_refs": list(self.source_refs),
            "output_sha256": self.output_sha256,
            "observed_output_bytes": self.observed_output_bytes,
            "content_class": self.content_class,
            "training_eligible": self.training_eligible,
            "weight_update_eligible": self.weight_update_eligible,
        }


@dataclass(frozen=True)
class ToolRequest:
    request_id: str
    tool_name: ToolName
    arguments: ToolArguments
    timeout_ms: int
    max_output_bytes: int
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "tool_name": self.tool_name.value,
            "arguments": self.arguments,
            "timeout_ms": self.timeout_ms,
            "max_output_bytes": self.max_output_bytes,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class ToolResult:
    request_id: str
    tool_name: str
    ok: bool
    output: JsonValue | None
    error: ToolError | None
    provenance: Provenance
    protocol_version: int = PROTOCOL_VERSION
    phase: Phase = Phase.TOOL_EXECUTION

    def __post_init__(self) -> None:
        if self.ok == (self.error is not None):
            raise ValueError(
                "successful results must not carry an error; failed results must carry one"
            )
        if self.phase is not Phase.TOOL_EXECUTION:
            raise ValueError("ToolResult phase must be tool_execution")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "protocol_version": self.protocol_version,
            "phase": self.phase.value,
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "output": self.output,
            "error": None if self.error is None else self.error.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True)
class ModelGeneration:
    text: str
    requested_tools: tuple[dict[str, JsonValue], ...] = ()
    phase: Phase = Phase.MODEL_GENERATION

    def __post_init__(self) -> None:
        if self.phase is not Phase.MODEL_GENERATION:
            raise ValueError("ModelGeneration phase must be model_generation")


@dataclass(frozen=True)
class ToolObservation:
    result: ToolResult
    phase: Phase = Phase.TOOL_OBSERVATION
    trusted_as_instruction: bool = False
    training_eligible: bool = False
    weight_update_eligible: bool = False

    def __post_init__(self) -> None:
        if self.phase is not Phase.TOOL_OBSERVATION:
            raise ValueError("ToolObservation phase must be tool_observation")

    @property
    def observation_id(self) -> str:
        return hashlib.sha256(self.result.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class FinalAnswer:
    text: str
    observation_ids: tuple[str, ...] = ()
    phase: Phase = Phase.FINAL_ANSWER

    def __post_init__(self) -> None:
        if self.phase is not Phase.FINAL_ANSWER:
            raise ValueError("FinalAnswer phase must be final_answer")


@dataclass(frozen=True)
class ToolUseCycle:
    generation: ModelGeneration
    executions: tuple[ToolResult, ...]
    observations: tuple[ToolObservation, ...]
    final_answer: FinalAnswer

    def __post_init__(self) -> None:
        execution_ids = {result.request_id for result in self.executions}
        observation_ids = {obs.result.request_id for obs in self.observations}
        if execution_ids != observation_ids:
            raise ValueError("every tool execution must have exactly one observation")
        known_observation_ids = {obs.observation_id for obs in self.observations}
        if not set(self.final_answer.observation_ids).issubset(known_observation_ids):
            raise ValueError("final answer references an unknown observation")


@dataclass(frozen=True)
class MockExecution:
    output: JsonValue
    cost_ms: int = 0
    source_refs: tuple[str, ...] = ()


class ProtocolViolation(ValueError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _validate_json_value(value: Any, *, path: str = "$", depth: int = 0) -> JsonValue:
    if depth > 32:
        raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, f"JSON nesting too deep at {path}")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, f"non-finite float at {path}")
        return value
    if isinstance(value, list):
        return [
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return [
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        output: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolViolation(
                    ErrorCode.INVALID_ARGUMENTS, f"non-string object key at {path}"
                )
            output[key] = _validate_json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return output
    raise ProtocolViolation(
        ErrorCode.INVALID_ARGUMENTS,
        f"unsupported JSON value {type(value).__name__} at {path}",
    )


def canonical_json_bytes(value: Any) -> bytes:
    validated = _validate_json_value(value)
    return json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_keys(
    arguments: Mapping[str, JsonValue], required: set[str], optional: set[str] | None = None
) -> None:
    if optional is None:
        optional = set()
    actual = set(arguments)
    missing = required - actual
    extra = actual - required - optional
    if missing:
        raise ProtocolViolation(
            ErrorCode.INVALID_ARGUMENTS, f"missing argument(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise ProtocolViolation(
            ErrorCode.INVALID_ARGUMENTS, f"unexpected argument(s): {', '.join(sorted(extra))}"
        )


def _bounded_text(value: JsonValue, name: str, *, limit: int) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, f"{name} must be a non-empty string")
    if len(value.encode("utf-8")) > limit:
        raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, f"{name} exceeds {limit} bytes")
    return value


def _bounded_int(value: JsonValue, name: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ProtocolViolation(
            ErrorCode.INVALID_ARGUMENTS, f"{name} must be in [{minimum}, {maximum}]"
        )
    return value


def _validate_relative_path(value: JsonValue) -> str:
    raw = _bounded_text(value, "path", limit=1024)
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise ProtocolViolation(ErrorCode.POLICY_DENIED, "filesystem path escapes sandbox")
    normalized = str(path)
    if normalized in {".", ""}:
        raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, "path must identify a sandbox entry")
    return normalized


_ALLOWED_PYTHON_IMPORT_ROOTS = {
    "collections",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
}
_FORBIDDEN_PYTHON_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}


def _validate_python_source(source: str) -> None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ProtocolViolation(
            ErrorCode.INVALID_ARGUMENTS, "python source is not valid syntax"
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in _ALLOWED_PYTHON_IMPORT_ROOTS:
                    raise ProtocolViolation(
                        ErrorCode.POLICY_DENIED, f"python import denied: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_PYTHON_IMPORT_ROOTS:
                raise ProtocolViolation(ErrorCode.POLICY_DENIED, f"python import denied: {root}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_PYTHON_CALLS
        ):
            raise ProtocolViolation(
                ErrorCode.POLICY_DENIED, f"python call denied: {node.func.id}"
            )


def _validate_arguments(
    tool_name: ToolName, arguments: Mapping[str, JsonValue]
) -> ToolArguments:
    args = dict(arguments)
    if tool_name is ToolName.WEB_SEARCH:
        _strict_keys(args, {"query"}, {"top_k"})
        query = _bounded_text(args["query"], "query", limit=4096)
        top_k = _bounded_int(args.get("top_k", 5), "top_k", minimum=1, maximum=20)
        return {"query": query, "top_k": top_k}
    if tool_name is ToolName.DOCUMENT_RETRIEVAL:
        _strict_keys(args, {"document_id"}, {"query", "max_chunks"})
        document_id = _bounded_text(args["document_id"], "document_id", limit=512)
        query_value = args.get("query")
        query = None if query_value is None else _bounded_text(query_value, "query", limit=4096)
        max_chunks = _bounded_int(
            args.get("max_chunks", 5), "max_chunks", minimum=1, maximum=20
        )
        return {"document_id": document_id, "query": query, "max_chunks": max_chunks}
    if tool_name is ToolName.CALCULATOR:
        _strict_keys(args, {"expression"})
        return {"expression": _bounded_text(args["expression"], "expression", limit=1024)}
    if tool_name is ToolName.PYTHON_EXECUTION:
        _strict_keys(args, {"code"}, {"inputs"})
        code = _bounded_text(args["code"], "code", limit=20_000)
        _validate_python_source(code)
        inputs = args.get("inputs", {})
        if not isinstance(inputs, Mapping):
            raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, "inputs must be an object")
        normalized_inputs = _validate_json_value(inputs, path="$.arguments.inputs")
        assert isinstance(normalized_inputs, dict)
        return {"code": code, "inputs": normalized_inputs}
    if tool_name is ToolName.FILESYSTEM_SANDBOX:
        _strict_keys(args, {"operation", "path"}, {"content"})
        operation = _bounded_text(args["operation"], "operation", limit=32)
        if operation not in {"read", "write", "list"}:
            raise ProtocolViolation(
                ErrorCode.INVALID_ARGUMENTS, "filesystem operation must be read, write, or list"
            )
        path = _validate_relative_path(args["path"])
        content = args.get("content")
        if operation == "write":
            content = _bounded_text(content, "content", limit=32_768)
        elif content is not None:
            raise ProtocolViolation(
                ErrorCode.INVALID_ARGUMENTS, "content is valid only for filesystem write"
            )
        return {"operation": operation, "path": path, "content": content}
    if tool_name is ToolName.FUTURE_API:
        _strict_keys(args, {"api_name", "operation"}, {"params"})
        api_name = _bounded_text(args["api_name"], "api_name", limit=128)
        operation = _bounded_text(args["operation"], "operation", limit=128)
        if not _IDENTIFIER_RE.fullmatch(api_name) or not _IDENTIFIER_RE.fullmatch(operation):
            raise ProtocolViolation(
                ErrorCode.INVALID_ARGUMENTS, "api_name and operation must be stable identifiers"
            )
        params = args.get("params", {})
        if not isinstance(params, Mapping):
            raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, "params must be an object")
        normalized_params = _validate_json_value(params, path="$.arguments.params")
        assert isinstance(normalized_params, dict)
        return {"api_name": api_name, "operation": operation, "params": normalized_params}
    raise ProtocolViolation(ErrorCode.UNKNOWN_TOOL, f"unsupported tool {tool_name.value}")


def parse_tool_request(payload: object) -> ToolRequest:
    if not isinstance(payload, Mapping):
        raise ProtocolViolation(ErrorCode.MALFORMED_REQUEST, "tool request must be an object")
    required = {"request_id", "tool_name", "arguments", "timeout_ms", "max_output_bytes"}
    optional = {"protocol_version"}
    raw = dict(payload)
    actual = set(raw)
    missing = required - actual
    extra = actual - required - optional
    if missing:
        raise ProtocolViolation(
            ErrorCode.MALFORMED_REQUEST,
            f"missing top-level field(s): {', '.join(sorted(missing))}",
        )
    if extra:
        raise ProtocolViolation(
            ErrorCode.MALFORMED_REQUEST,
            f"unexpected top-level field(s): {', '.join(sorted(extra))}",
        )
    protocol_version = raw.get("protocol_version", PROTOCOL_VERSION)
    if protocol_version != PROTOCOL_VERSION:
        raise ProtocolViolation(ErrorCode.MALFORMED_REQUEST, "unsupported protocol_version")
    request_id = raw["request_id"]
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
        raise ProtocolViolation(ErrorCode.MALFORMED_REQUEST, "invalid request_id")
    tool_raw = raw["tool_name"]
    if not isinstance(tool_raw, str) or not tool_raw:
        raise ProtocolViolation(ErrorCode.MALFORMED_REQUEST, "tool_name must be explicit")
    try:
        tool_name = ToolName(tool_raw)
    except ValueError as exc:
        raise ProtocolViolation(ErrorCode.UNKNOWN_TOOL, f"unknown tool: {tool_raw}") from exc
    arguments_raw = raw["arguments"]
    if not isinstance(arguments_raw, Mapping):
        raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, "arguments must be an object")
    normalized_arguments = _validate_json_value(arguments_raw, path="$.arguments")
    assert isinstance(normalized_arguments, dict)
    if len(canonical_json_bytes(normalized_arguments)) > MAX_ARGUMENT_BYTES:
        raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, "arguments exceed protocol limit")
    timeout_ms = _bounded_int(
        raw["timeout_ms"], "timeout_ms", minimum=1, maximum=MAX_TIMEOUT_MS
    )
    max_output_bytes = _bounded_int(
        raw["max_output_bytes"], "max_output_bytes", minimum=1, maximum=MAX_OUTPUT_BYTES
    )
    validated_arguments = _validate_arguments(tool_name, normalized_arguments)
    return ToolRequest(
        request_id=request_id,
        tool_name=tool_name,
        arguments=validated_arguments,
        timeout_ms=timeout_ms,
        max_output_bytes=max_output_bytes,
        protocol_version=PROTOCOL_VERSION,
    )


def _safe_payload_identity(payload: object) -> str:
    try:
        body = canonical_json_bytes(payload)
    except ProtocolViolation:
        body = canonical_json_bytes({"invalid_payload_type": type(payload).__name__})
    return hashlib.sha256(body).hexdigest()


def _calculator(expression: str) -> int | float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ProtocolViolation(
            ErrorCode.INVALID_ARGUMENTS, "calculator expression is not valid syntax"
        ) from exc

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool) or not math.isfinite(float(node.value)):
                raise ProtocolViolation(ErrorCode.INVALID_ARGUMENTS, "invalid numeric literal")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.FloorDiv):
                result = left // right
            elif isinstance(node.op, ast.Mod):
                result = left % right
            else:
                if abs(float(right)) > 64:
                    raise ProtocolViolation(
                        ErrorCode.POLICY_DENIED, "calculator exponent too large"
                    )
                result = left**right
            if isinstance(result, complex) or not math.isfinite(float(result)):
                raise ProtocolViolation(
                    ErrorCode.EXECUTION_ERROR, "calculator result is non-finite"
                )
            if abs(float(result)) > 1e100:
                raise ProtocolViolation(
                    ErrorCode.POLICY_DENIED, "calculator magnitude limit exceeded"
                )
            return result
        raise ProtocolViolation(
            ErrorCode.POLICY_DENIED, "calculator expression is not arithmetic-only"
        )

    try:
        return evaluate(tree)
    except ZeroDivisionError as exc:
        raise ProtocolViolation(ErrorCode.EXECUTION_ERROR, "division by zero") from exc


Adapter = Callable[[ToolRequest], MockExecution]


@dataclass
class MockExecutor:
    web_index: Mapping[str, Sequence[Mapping[str, JsonValue]]] = field(default_factory=dict)
    document_store: Mapping[str, Sequence[str]] = field(default_factory=dict)
    api_fixtures: Mapping[tuple[str, str], JsonValue] = field(default_factory=dict)
    filesystem: dict[str, str] = field(default_factory=dict)
    executor_id: str = "postbase254-mock-executor-v1"
    adapter_cost_ms: Mapping[str, int] = field(default_factory=dict)

    def execute_model_request(self, payload: object) -> ToolResult:
        try:
            request = parse_tool_request(payload)
        except ProtocolViolation as exc:
            payload_hash = _safe_payload_identity(payload)
            request_id = "malformed-" + payload_hash[:16]
            tool_name = "__invalid__"
            if isinstance(payload, Mapping):
                candidate_id = payload.get("request_id")
                candidate_tool = payload.get("tool_name")
                if isinstance(candidate_id, str) and _REQUEST_ID_RE.fullmatch(candidate_id):
                    request_id = candidate_id
                if isinstance(candidate_tool, str) and candidate_tool:
                    tool_name = candidate_tool[:128]
            provenance = Provenance(
                executor_id=self.executor_id,
                adapter_id="request-parser-v1",
                request_sha256=payload_hash,
            )
            return ToolResult(
                request_id=request_id,
                tool_name=tool_name,
                ok=False,
                output=None,
                error=ToolError(exc.code, str(exc), retryable=False),
                provenance=provenance,
            )
        return self.execute(request)

    def execute(self, request: ToolRequest) -> ToolResult:
        request_hash = request.sha256
        adapter_id = f"mock:{request.tool_name.value}:v1"
        try:
            execution = self._dispatch(request)
            declared_cost = max(
                execution.cost_ms, self.adapter_cost_ms.get(request.tool_name.value, 0)
            )
            if declared_cost > request.timeout_ms:
                return self._failure(
                    request,
                    adapter_id,
                    ErrorCode.TIMEOUT,
                    f"tool exceeded timeout budget of {request.timeout_ms} ms",
                    request_hash=request_hash,
                    source_refs=execution.source_refs,
                )
            normalized_output = _validate_json_value(execution.output, path="$.tool_output")
            output_bytes = canonical_json_bytes(normalized_output)
            output_hash = hashlib.sha256(output_bytes).hexdigest()
            if len(output_bytes) > request.max_output_bytes:
                return self._failure(
                    request,
                    adapter_id,
                    ErrorCode.OUTPUT_LIMIT,
                    "tool output exceeded declared max_output_bytes",
                    request_hash=request_hash,
                    source_refs=execution.source_refs,
                    output_hash=output_hash,
                    observed_output_bytes=len(output_bytes),
                )
            provenance = Provenance(
                executor_id=self.executor_id,
                adapter_id=adapter_id,
                request_sha256=request_hash,
                source_refs=execution.source_refs,
                output_sha256=output_hash,
                observed_output_bytes=len(output_bytes),
            )
            return ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name.value,
                ok=True,
                output=normalized_output,
                error=None,
                provenance=provenance,
            )
        except ProtocolViolation as exc:
            return self._failure(
                request,
                adapter_id,
                exc.code,
                str(exc),
                request_hash=request_hash,
            )
        except Exception:  # noqa: BLE001
            return self._failure(
                request,
                adapter_id,
                ErrorCode.EXECUTION_ERROR,
                "tool adapter failed without exposing internal exception data",
                request_hash=request_hash,
            )

    def observe(self, result: ToolResult) -> ToolObservation:
        return ToolObservation(result=result)

    def _failure(
        self,
        request: ToolRequest,
        adapter_id: str,
        code: ErrorCode,
        message: str,
        *,
        request_hash: str,
        source_refs: tuple[str, ...] = (),
        output_hash: str | None = None,
        observed_output_bytes: int = 0,
    ) -> ToolResult:
        return ToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name.value,
            ok=False,
            output=None,
            error=ToolError(code, message, retryable=code is ErrorCode.TIMEOUT),
            provenance=Provenance(
                executor_id=self.executor_id,
                adapter_id=adapter_id,
                request_sha256=request_hash,
                source_refs=source_refs,
                output_sha256=output_hash,
                observed_output_bytes=observed_output_bytes,
            ),
        )

    def _dispatch(self, request: ToolRequest) -> MockExecution:
        if request.tool_name is ToolName.WEB_SEARCH:
            return self._web_search(request)
        if request.tool_name is ToolName.DOCUMENT_RETRIEVAL:
            return self._document_retrieve(request)
        if request.tool_name is ToolName.CALCULATOR:
            expression = str(request.arguments["expression"])
            return MockExecution(
                output={"expression": expression, "value": _calculator(expression)},
                source_refs=("local:calculator",),
            )
        if request.tool_name is ToolName.PYTHON_EXECUTION:
            code = str(request.arguments["code"])
            return MockExecution(
                output={
                    "mode": "mock_only",
                    "executed": False,
                    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "note": "protocol mock never executes model-supplied Python",
                },
                source_refs=("local:python-mock",),
            )
        if request.tool_name is ToolName.FILESYSTEM_SANDBOX:
            return self._filesystem(request)
        if request.tool_name is ToolName.FUTURE_API:
            return self._future_api(request)
        raise ProtocolViolation(ErrorCode.UNKNOWN_TOOL, "tool is not registered")

    def _web_search(self, request: ToolRequest) -> MockExecution:
        query = str(request.arguments["query"])
        top_k = int(request.arguments["top_k"])
        records = list(self.web_index.get(query, ()))[:top_k]
        normalized = [_validate_json_value(record, path="$.web_result") for record in records]
        refs = tuple(
            str(record.get("ref", f"mock-search:{index}"))
            for index, record in enumerate(records)
            if isinstance(record, Mapping)
        )
        return MockExecution(output={"query": query, "results": normalized}, source_refs=refs)

    def _document_retrieve(self, request: ToolRequest) -> MockExecution:
        document_id = str(request.arguments["document_id"])
        chunks = self.document_store.get(document_id)
        if chunks is None:
            raise ProtocolViolation(ErrorCode.NOT_FOUND, "document not found")
        query = request.arguments.get("query")
        selected = list(chunks)
        if isinstance(query, str) and query:
            lowered = query.casefold()
            selected = [chunk for chunk in selected if lowered in chunk.casefold()]
        max_chunks = int(request.arguments["max_chunks"])
        selected = selected[:max_chunks]
        return MockExecution(
            output={"document_id": document_id, "chunks": selected},
            source_refs=(f"document:{document_id}",),
        )

    def _filesystem(self, request: ToolRequest) -> MockExecution:
        operation = str(request.arguments["operation"])
        path = str(request.arguments["path"])
        source = f"sandbox:{path}"
        if operation == "write":
            content = str(request.arguments["content"])
            self.filesystem[path] = content
            return MockExecution(
                output={"path": path, "bytes_written": len(content.encode())},
                source_refs=(source,),
            )
        if operation == "read":
            if path not in self.filesystem:
                raise ProtocolViolation(ErrorCode.NOT_FOUND, "sandbox file not found")
            return MockExecution(
                output={"path": path, "content": self.filesystem[path]}, source_refs=(source,)
            )
        prefix = path.rstrip("/") + "/"
        entries = sorted(
            name for name in self.filesystem if name == path or name.startswith(prefix)
        )
        return MockExecution(
            output={"path": path, "entries": entries}, source_refs=(source,)
        )

    def _future_api(self, request: ToolRequest) -> MockExecution:
        api_name = str(request.arguments["api_name"])
        operation = str(request.arguments["operation"])
        key = (api_name, operation)
        if key not in self.api_fixtures:
            raise ProtocolViolation(ErrorCode.NOT_FOUND, "mock API fixture not found")
        return MockExecution(
            output={
                "api_name": api_name,
                "operation": operation,
                "result": self.api_fixtures[key],
            },
            source_refs=(f"api:{api_name}:{operation}",),
        )
