"""Model-independent typed tool boundary (Plan 5 / Section 7).

Discovery is not an execution grant. This module never imports, creates or
invokes an external tool. TaskStore is authoritative for pending effects and
receipts; the host must independently attest grants, invocations and results.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

from .task_state import TaskSnapshot, TaskStore, _digest, _id, _json, _pairs, _uint

SCHEMA = "12-6.agent-typed-tools.v1"
DESCRIPTOR_VERSION = "12-6.tool-descriptor.v1"
EffectClass = Literal["none", "read", "write", "external"]
Outcome = Literal["success", "failure"]


class ToolBoundaryError(ValueError):
    """Unsupported contract, unauthorized tool, or forged tool evidence."""


def _bounded_json(value: object) -> str:
    try:
        raw = _json(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ToolBoundaryError("invalid canonical JSON") from exc
    if len(raw.encode("utf-8")) > 65_536:
        raise ToolBoundaryError("tool envelope exceeds 64 KiB")
    return raw


def _spec(spec: object, *, depth: int = 0) -> None:
    if depth > 5 or not isinstance(spec, dict):
        raise ToolBoundaryError("invalid or deeply nested tool schema")
    kind = spec.get("type")
    if kind == "object":
        if set(spec) != {"type", "properties", "required", "additionalProperties"}:
            raise ToolBoundaryError("unsupported object schema")
        properties, required = spec["properties"], spec["required"]
        if not isinstance(properties, dict) or len(properties) > 64:
            raise ToolBoundaryError("invalid bounded tool properties")
        if (not isinstance(required, list) or len(required) > len(properties)
                or not all(isinstance(x, str) for x in required)
                or len(set(required)) != len(required)
                or not set(required).issubset(properties)
                or spec["additionalProperties"] is not False):
            raise ToolBoundaryError("open or malformed object schema")
        for key, child in properties.items():
            _id(key, "schema property")
            _spec(child, depth=depth + 1)
    elif kind == "array":
        if set(spec) != {"type", "items", "maxItems"}:
            raise ToolBoundaryError("unbounded or unsupported array schema")
        _uint(spec["maxItems"], "maxItems")
        if spec["maxItems"] > 128:
            raise ToolBoundaryError("array schema maximum too large")
        _spec(spec["items"], depth=depth + 1)
    elif kind == "string":
        if set(spec) != {"type", "maxLength"}:
            raise ToolBoundaryError("unbounded string schema")
        _uint(spec["maxLength"], "maxLength")
        if spec["maxLength"] > 65_536:
            raise ToolBoundaryError("string schema maximum too large")
    elif kind in ("integer", "number", "boolean"):
        if set(spec) != {"type"}:
            raise ToolBoundaryError("unsupported primitive constraints")
    else:
        raise ToolBoundaryError("unsupported schema primitive")


def _value(spec: dict[str, object], value: object) -> None:
    kind = spec["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ToolBoundaryError("expected object")
        if any(not isinstance(k, str) for k in value):
            raise ToolBoundaryError("object has a non-string key")
        if not set(spec["required"]).issubset(value):
            raise ToolBoundaryError("missing required property")
        if not set(value).issubset(spec["properties"]):
            raise ToolBoundaryError("undeclared property")
        for name, item in value.items():
            _value(spec["properties"][name], item)
    elif kind == "array":
        if not isinstance(value, list) or len(value) > spec["maxItems"]:
            raise ToolBoundaryError("array type or bound violation")
        for item in value:
            _value(spec["items"], item)
    elif kind == "string":
        if not isinstance(value, str) or len(value) > spec["maxLength"]:
            raise ToolBoundaryError("string type or bound violation")
    elif kind == "integer":
        if type(value) is not int:
            raise ToolBoundaryError("integer type required")
    elif kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ToolBoundaryError("finite numeric type required")
    elif kind == "boolean" and type(value) is not bool:
        raise ToolBoundaryError("boolean type required")


def _parse_canonical(raw: str) -> object:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65_536:
        raise ToolBoundaryError("invalid envelope size")
    try:
        value = json.loads(
            raw, object_pairs_hook=_pairs,
            parse_constant=lambda val: (_ for _ in ()).throw(ToolBoundaryError(val)),
        )
        if _bounded_json(value) != raw:
            raise ToolBoundaryError("noncanonical JSON encoding")
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ToolBoundaryError("invalid canonical JSON") from exc


@dataclass(frozen=True)
class ToolDescriptor:
    tool_id: str
    version: str
    capabilities: tuple[str, ...]
    permissions: tuple[str, ...]
    side_effect: EffectClass
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    available: bool

    def validate(self) -> None:
        _id(self.tool_id, "tool_id")
        if self.version != DESCRIPTOR_VERSION:
            raise ToolBoundaryError("unsupported descriptor version")
        if self.side_effect not in ("none", "read", "write", "external"):
            raise ToolBoundaryError("invalid effect class")
        if type(self.available) is not bool:
            raise ToolBoundaryError("invalid availability marker")
        for items in (self.capabilities, self.permissions):
            if (not isinstance(items, tuple) or len(items) > 64
                    or items != tuple(sorted(set(items)))):
                raise ToolBoundaryError("permissions/capabilities must be unique and sorted")
            for item in items:
                _id(item, "capability/permission")
        if self.side_effect in ("write", "external") and not self.permissions:
            raise ToolBoundaryError("effectful tool needs a permission")
        _spec(self.input_schema)
        _spec(self.output_schema)
        if self.input_schema.get("type") != "object" or (
            self.output_schema.get("type") != "object"
        ):
            raise ToolBoundaryError("top-level tool schemas must be closed objects")
        _bounded_json(asdict(self))


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    task_id: str
    effect_id: str
    tool_id: str
    descriptor_digest: str
    request_json: str
    grant_evidence_id: str
    permissions: tuple[str, ...]
    control_epoch: int
    task_revision: int


@dataclass(frozen=True)
class ToolResult:
    result_id: str
    call_id: str
    task_id: str
    effect_id: str
    tool_id: str
    receipt_id: str
    evidence_id: str
    outcome: Outcome
    output_json: str


def _identity(payload: object) -> str:
    return _digest(_bounded_json(payload))


def make_result(
    call: ToolCall, *, receipt_id: str, evidence_id: str, outcome: Outcome,
    output: dict[str, object],
) -> ToolResult:
    """Format an untrusted result; only host-verified acceptance is authoritative."""
    if not isinstance(call, ToolCall):
        raise ToolBoundaryError("invalid call envelope")
    _id(receipt_id, "receipt_id")
    _id(evidence_id, "evidence_id")
    if outcome not in ("success", "failure") or not isinstance(output, dict):
        raise ToolBoundaryError("invalid tool result")
    output_json = _bounded_json(output)
    body = {
        "call_id": call.call_id, "task_id": call.task_id, "effect_id": call.effect_id,
        "tool_id": call.tool_id, "receipt_id": receipt_id,
        "evidence_id": evidence_id, "outcome": outcome, "output_json": output_json,
    }
    return ToolResult(result_id=_identity(body), **body)


class ToolRegistry:
    """Immutable source-verified descriptors and evidence-bound call preparation."""

    def __init__(self, tasks: TaskStore) -> None:
        if not isinstance(tasks, TaskStore):
            raise ToolBoundaryError("TaskStore required")
        self.tasks = tasks
        self._descriptors: dict[str, str] = {}

    def register(
        self, descriptor: ToolDescriptor, *,
        verify_descriptor: object,
    ) -> str:
        if not isinstance(descriptor, ToolDescriptor):
            raise ToolBoundaryError("typed descriptor required")
        descriptor.validate()
        if (not callable(verify_descriptor)
                or verify_descriptor(descriptor) is not True):
            raise ToolBoundaryError("unverified tool descriptor")
        if descriptor.tool_id in self._descriptors:
            raise ToolBoundaryError("duplicate tool descriptor")
        raw = _bounded_json(asdict(descriptor))
        self._descriptors[descriptor.tool_id] = raw
        return _digest(raw)

    def get(self, tool_id: str) -> tuple[ToolDescriptor, str]:
        _id(tool_id, "tool_id")
        raw = self._descriptors.get(tool_id)
        if raw is None:
            raise ToolBoundaryError("unknown/unavailable tool")
        value = _parse_canonical(raw)
        descriptor = ToolDescriptor(
            tool_id=value["tool_id"], version=value["version"],
            capabilities=tuple(value["capabilities"]),
            permissions=tuple(value["permissions"]),
            side_effect=value["side_effect"],
            input_schema=value["input_schema"], output_schema=value["output_schema"],
            available=value["available"],
        )
        descriptor.validate()
        if not descriptor.available:
            raise ToolBoundaryError("unavailable tool")
        return descriptor, _digest(raw)

    def discover(self) -> tuple[ToolDescriptor, ...]:
        """Read-only descriptors. No permission, consent, execution or discovery lease."""
        return tuple(self.get(name)[0] for name in sorted(self._descriptors)
                     if json.loads(self._descriptors[name])["available"])

    def prepare(
        self, *, task_id: str, effect_id: str, tool_id: str, request: dict[str, object],
        permissions: tuple[str, ...], grant_evidence_id: str,
        expected_epoch: int, expected_revision: int,
        authorize: object,
    ) -> ToolCall:
        descriptor, digest = self.get(tool_id)
        _id(task_id, "task_id")
        _id(effect_id, "effect_id")
        _id(grant_evidence_id, "grant_evidence_id")
        _uint(expected_epoch, "expected_epoch")
        _uint(expected_revision, "expected_revision")
        if not isinstance(permissions, tuple) or permissions != descriptor.permissions:
            raise ToolBoundaryError("permissions differ from least-privilege contract")
        if not isinstance(request, dict):
            raise ToolBoundaryError("tool arguments must be an object")
        request_json = _bounded_json(request)
        _value(descriptor.input_schema, request)
        state = self.tasks.load(task_id)
        if state.control_epoch != expected_epoch or state.revision != expected_revision:
            raise ToolBoundaryError("stale task control epoch/revision")
        if not any(e.effect_id == effect_id and e.status == "pending"
                   for e in state.pending_effects):
            raise ToolBoundaryError("unreserved, issued or unknown task effect")
        if (not callable(authorize) or
                authorize(descriptor, state, request, grant_evidence_id) is not True):
            raise ToolBoundaryError("host refused tool grant")
        body = {
            "task_id": task_id, "effect_id": effect_id, "tool_id": tool_id,
            "descriptor_digest": digest, "request_json": request_json,
            "grant_evidence_id": grant_evidence_id, "permissions": permissions,
            "control_epoch": expected_epoch, "task_revision": expected_revision,
        }
        return ToolCall(call_id=_identity(body), **body)

    def accept_result(
        self, call: ToolCall, result: ToolResult, *, verify_call: object,
        verify_result: object,
    ) -> TaskSnapshot:
        if not isinstance(call, ToolCall) or not isinstance(result, ToolResult):
            raise ToolBoundaryError("typed call/result envelopes required")
        descriptor, digest = self.get(call.tool_id)
        body = asdict(call)
        call_id = body.pop("call_id")
        if (call_id != _identity(body) or digest != call.descriptor_digest
                or call.permissions != descriptor.permissions):
            raise ToolBoundaryError("call/descriptor integrity mismatch")
        _value(descriptor.input_schema, _parse_canonical(call.request_json))
        result_body = asdict(result)
        result_id = result_body.pop("result_id")
        if result_id != _identity(result_body):
            raise ToolBoundaryError("tool result envelope integrity mismatch")
        if (result.call_id != call.call_id or result.task_id != call.task_id
                or result.effect_id != call.effect_id or result.tool_id != call.tool_id):
            raise ToolBoundaryError("tool result cross-binding mismatch")
        _id(result.receipt_id, "receipt_id")
        _id(result.evidence_id, "evidence_id")
        if result.outcome not in ("success", "failure"):
            raise ToolBoundaryError("invalid tool outcome")
        _value(descriptor.output_schema, _parse_canonical(result.output_json))
        if (not callable(verify_call) or verify_call(call) is not True
                or not callable(verify_result) or verify_result(call, result) is not True):
            raise ToolBoundaryError("missing independently verified effect evidence")
        state = self.tasks.load(call.task_id)
        if state.control_epoch != call.control_epoch or (
            not any(e.effect_id == call.effect_id and e.status == "unknown"
                    for e in state.pending_effects)
        ):
            raise ToolBoundaryError("stale epoch or unissued/unreconciled effect")
        return self.tasks.resolve_effect(
            call.task_id, call.effect_id, result.receipt_id,
            expected_epoch=state.control_epoch, expected_revision=state.revision,
        )
