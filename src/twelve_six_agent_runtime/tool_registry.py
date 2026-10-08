"""Plan 5 S7: provider-neutral typed tool boundary; TaskStore owns all effects.

The trusted embedding host, not the model, supplies grants, confirmations, the
executor and evidence. Discovery is never an execution capability.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .task_state import StateError, TaskStore, _id

SCHEMA = "12-6.agent-tools.v1"
SideEffect = Literal["read", "write", "irreversible"]


class ToolError(ValueError):
    """Unknown/unauthorized tool, malformed schema or ambiguous outcome."""


def canonical(value: Any) -> str:
    def walk(node: Any, depth: int = 0) -> None:
        if depth > 12:
            raise ToolError("JSON nesting limit")
        if node is None or type(node) in (str, bool, int):
            return
        if type(node) is float and math.isfinite(node):
            return
        if type(node) is list:
            for child in node:
                walk(child, depth + 1)
            return
        if type(node) is dict and all(type(k) is str for k in node):
            for child in node.values():
                walk(child, depth + 1)
            return
        raise ToolError("unsupported or nonfinite JSON value")

    walk(value)
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ToolError("noncanonical tool data") from exc
    if len(payload.encode("utf-8")) > 65536:
        raise ToolError("tool data exceeds 64 KiB")
    return payload


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _schema(schema: Any, depth: int = 0) -> None:
    if depth > 8 or type(schema) is not dict:
        raise ToolError("unsupported schema depth or type")
    kind = schema.get("type")
    fields = {"type", "properties", "required", "additionalProperties", "items",
              "maxLength", "minLength", "maximum", "minimum", "maxItems"}
    if set(schema) - fields or kind not in ("object", "array", "string", "integer", "boolean"):
        raise ToolError("unsupported schema type or keyword")
    if kind == "object":
        props = schema.get("properties")
        required = schema.get("required", [])
        if (type(props) is not dict or schema.get("additionalProperties") is not False
                or type(required) is not list or len(set(required)) != len(required)
                or any(type(k) is not str or not k for k in props)
                or any(type(k) is not str or k not in props for k in required)):
            raise ToolError("unsafe object schema")
        if set(schema) - {"type", "properties", "required", "additionalProperties"}:
            raise ToolError("wrong object schema keywords")
        for child in props.values():
            _schema(child, depth + 1)
    elif kind == "array":
        if set(schema) - {"type", "items", "maxItems"}:
            raise ToolError("wrong array schema keywords")
        if type(schema.get("maxItems")) is not int or not 0 <= schema["maxItems"] <= 1024:
            raise ToolError("bounded array required")
        _schema(schema.get("items"), depth + 1)
    elif kind == "string":
        if set(schema) - {"type", "maxLength", "minLength"}:
            raise ToolError("wrong string schema keywords")
        lo, hi = schema.get("minLength", 0), schema.get("maxLength")
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= 65536:
            raise ToolError("bounded string required")
    elif kind == "integer":
        if set(schema) - {"type", "minimum", "maximum"}:
            raise ToolError("wrong integer schema keywords")
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if type(lo) is not int or type(hi) is not int or lo > hi:
            raise ToolError("bounded integer required")
    elif set(schema) != {"type"}:
        raise ToolError("wrong boolean schema keywords")


def _validate(value: Any, schema: Mapping[str, Any], depth: int = 0) -> None:
    if depth > 8:
        raise ToolError("tool payload too deeply nested")
    kind = schema["type"]
    if kind == "object":
        if type(value) is not dict:
            raise ToolError("expected object")
        if set(value) - set(schema["properties"]) or set(schema.get("required", [])) - set(value):
            raise ToolError("unknown or missing object fields")
        for key, child in value.items():
            _validate(child, schema["properties"][key], depth + 1)
    elif kind == "array":
        if type(value) is not list or len(value) > schema["maxItems"]:
            raise ToolError("invalid array")
        for child in value:
            _validate(child, schema["items"], depth + 1)
    elif kind == "string":
        if type(value) is not str or not schema.get("minLength", 0) <= len(value) <= schema["maxLength"]:
            raise ToolError("invalid string")
    elif kind == "integer":
        if type(value) is not int or not schema["minimum"] <= value <= schema["maximum"]:
            raise ToolError("invalid integer")
    elif type(value) is not bool:
        raise ToolError("invalid boolean")
    canonical(value)


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    version: int
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    capabilities: tuple[str, ...]
    permissions: tuple[str, ...]
    effect: SideEffect
    available: bool = True

    def validate(self) -> None:
        _id(self.name, "tool name")
        if type(self.version) is not int or self.version < 1:
            raise ToolError("invalid tool version")
        for label, names in (("capabilities", self.capabilities), ("permissions", self.permissions)):
            if (type(names) is not tuple or any(type(n) is not str or not n
                    or len(n) > 128 for n in names) or tuple(sorted(set(names))) != names):
                raise ToolError("invalid " + label)
        if self.effect not in ("read", "write", "irreversible") or type(self.available) is not bool:
            raise ToolError("invalid tool effect or availability")
        _schema(self.input_schema)
        _schema(self.output_schema)
        if self.input_schema["type"] != "object" or self.output_schema["type"] != "object":
            raise ToolError("tool boundaries require object schemas")
        canonical(self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "input_schema": self.input_schema, "output_schema": self.output_schema,
                "capabilities": self.capabilities, "permissions": self.permissions,
                "effect": self.effect, "available": self.available}


@dataclass(frozen=True)
class Invocation:
    task_id: str
    effect_id: str
    control_epoch: int
    task_revision: int
    tool: str
    tool_identity: str
    arguments: dict[str, Any]
    call_identity: str


@dataclass(frozen=True)
class Receipt:
    call_identity: str
    receipt_id: str
    output: dict[str, Any]
    evidence: tuple[str, ...]


class ToolRegistry:
    """Immutable descriptors; trusted permissions and effect authority remain external."""

    def __init__(self, descriptors: tuple[ToolDescriptor, ...]) -> None:
        if type(descriptors) is not tuple:
            raise ToolError("descriptor tuple required")
        frozen: dict[str, tuple[ToolDescriptor, str]] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, ToolDescriptor):
                raise ToolError("invalid descriptor")
            descriptor.validate()
            if descriptor.name in frozen:
                raise ToolError("duplicate tool name")
            frozen[descriptor.name] = (descriptor, identity(descriptor.snapshot()))
        self._tools = frozen

    def discover(self) -> tuple[dict[str, Any], ...]:
        """Descriptors only. Never returns an execution token."""
        return tuple(json.loads(canonical(self._tools[name][0].snapshot()))
                     for name in sorted(self._tools))

    def prepare(self, tasks: TaskStore, *, task_id: str, effect_id: str,
                epoch: int, revision: int, tool: str, arguments: dict[str, Any],
                host_permissions: frozenset[str], confirmed: Callable[[str], bool] | None = None
                ) -> Invocation:
        item = self._tools.get(tool)
        if item is None or not item[0].available:
            raise ToolError("unknown or unavailable tool")
        desc, digest = item
        if digest != identity(desc.snapshot()):
            raise ToolError("descriptor drift after registry creation")
        if type(host_permissions) is not frozenset or not set(desc.permissions) <= host_permissions:
            raise ToolError("trusted host permission denied")
        _validate(arguments, desc.input_schema)
        state = tasks.load(task_id)
        if state.control_epoch != epoch or state.revision != revision:
            raise ToolError("stale task epoch or revision")
        pending = [e for e in state.pending_effects if e.effect_id == effect_id]
        if len(pending) != 1 or pending[0].status != "pending":
            raise ToolError("effect must be reserved as pending in TaskStore")
        call_identity = identity({"schema": SCHEMA, "task": task_id, "effect": effect_id,
                                  "epoch": epoch, "revision": revision, "tool": digest,
                                  "arguments": arguments})
        if desc.effect == "irreversible":
            if not callable(confirmed) or confirmed(call_identity) is not True:
                raise ToolError("trusted host confirmation missing")
        return Invocation(task_id, effect_id, epoch, revision, tool, digest,
                          json.loads(canonical(arguments)), call_identity)

    def invoke(self, tasks: TaskStore, call: Invocation, *,
               executor: Callable[[str, dict[str, Any]], Receipt]) -> Receipt:
        """Issue BEFORE execution; unknown on failure. Never retry automatically."""
        if not isinstance(call, Invocation):
            raise ToolError("typed invocation required")
        item = self._tools.get(call.tool)
        if item is None or not item[0].available or identity(item[0].snapshot()) != call.tool_identity:
            raise ToolError("unavailable or drifted tool")
        desc = item[0]
        if identity({"schema": SCHEMA, "task": call.task_id, "effect": call.effect_id,
                     "epoch": call.control_epoch, "revision": call.task_revision,
                     "tool": call.tool_identity, "arguments": call.arguments}) != call.call_identity:
            raise ToolError("invocation identity mismatch")
        _validate(call.arguments, desc.input_schema)
        try:
            issued = tasks.issue_effect(
                call.task_id, call.effect_id, expected_epoch=call.control_epoch,
                expected_revision=call.task_revision,
            )
        except StateError as exc:
            raise ToolError("stale or already issued effect; reconcile, never replay") from exc
        # Even if executor throws or output is malformed, TaskStore keeps UNKNOWN.
        receipt = executor(call.tool, json.loads(canonical(call.arguments)))
        if not isinstance(receipt, Receipt) or receipt.call_identity != call.call_identity:
            raise ToolError("invalid or unbound receipt")
        _id(receipt.receipt_id, "receipt_id")
        if (type(receipt.evidence) is not tuple or not receipt.evidence
                or any(type(e) is not str or not e.strip() for e in receipt.evidence)):
            raise ToolError("missing trusted execution evidence")
        _validate(receipt.output, desc.output_schema)
        tasks.resolve_effect(call.task_id, call.effect_id, receipt.receipt_id,
                             expected_epoch=issued.control_epoch,
                             expected_revision=issued.revision)
        return receipt
