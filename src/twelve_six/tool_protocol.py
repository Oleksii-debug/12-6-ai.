from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

MCP_PROTOCOL_VERSION = "2026-07-28"
MCP_UPSTREAM_REPOSITORY = "modelcontextprotocol/modelcontextprotocol"
MCP_UPSTREAM_TAG = "2026-07-28"
MCP_UPSTREAM_COMMIT = "5f5440bb26a62e2cf3440b92da5a667efa03b267"
PROJECT_META_KEY = "io.twelve-six/toolContract"
_SCHEMA_PROFILE = "TWELVE_SIX_JSON_SCHEMA_SAFE_SUBSET_V1"
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_PERMISSION_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_SCHEMA_KEYS = {
    "$schema", "type", "properties", "required", "additionalProperties", "items", "enum",
    "minimum", "maximum", "minLength", "maxLength", "description", "title",
}


class ToolContractError(ValueError):
    """Raised when tool-contract data fails closed."""


class SideEffect(StrEnum):
    READ_ONLY = "READ_ONLY"
    REVERSIBLE_WRITE = "REVERSIBLE_WRITE"
    IRREVERSIBLE_WRITE = "IRREVERSIBLE_WRITE"


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ToolContractError(f"value is not canonical JSON: {exc}") from exc


def stable_identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolContractError(f"{path}: non-finite number is forbidden")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ToolContractError(f"{path}: object keys must be strings")
            _json_value(item, f"{path}.{key}")
        return
    raise ToolContractError(f"{path}: unsupported JSON value type {type(value).__name__}")


def _schema(schema: Any, path: str = "inputSchema") -> None:
    if not isinstance(schema, dict):
        raise ToolContractError(f"{path}: schema must be an object")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        raise ToolContractError(f"{path}: unsupported schema keywords: {sorted(unknown)!r}")
    schema_type = schema.get("type")
    allowed_types = {"object", "array", "string", "integer", "number", "boolean", "null"}
    if schema_type not in allowed_types:
        raise ToolContractError(f"{path}: unsupported or missing type")

    for key in ("description", "title"):
        if key in schema and (not isinstance(schema[key], str) or not schema[key].strip()):
            raise ToolContractError(f"{path}.{key}: must be a non-empty string")
    if schema_type != "object" and {"properties", "required", "additionalProperties"} & set(schema):
        raise ToolContractError(f"{path}: object-only keywords used on {schema_type}")
    if schema_type != "array" and "items" in schema:
        raise ToolContractError(f"{path}: array-only keywords used on {schema_type}")
    if schema_type not in {"integer", "number"} and {"minimum", "maximum"} & set(schema):
        raise ToolContractError(f"{path}: numeric-only keywords used on {schema_type}")
    if schema_type != "string" and {"minLength", "maxLength"} & set(schema):
        raise ToolContractError(f"{path}: string-only keywords used on {schema_type}")

    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise ToolContractError(f"{path}.enum: must be a non-empty array")
        _json_value(enum, f"{path}.enum")

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict):
            raise ToolContractError(f"{path}.properties: must be an object")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ToolContractError(f"{path}.required: must be an array of strings")
        if len(set(required)) != len(required):
            raise ToolContractError(f"{path}.required: duplicate entries are forbidden")
        missing = set(required) - set(properties)
        if missing:
            raise ToolContractError(f"{path}.required: unknown properties {sorted(missing)!r}")
        if schema.get("additionalProperties") is not False:
            raise ToolContractError(f"{path}: executable project profile requires additionalProperties=false")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                raise ToolContractError(f"{path}.properties: names must be non-empty strings")
            _schema(child, f"{path}.properties.{name}")
    elif schema_type == "array":
        if "items" not in schema:
            raise ToolContractError(f"{path}: array schemas require items")
        _schema(schema["items"], f"{path}.items")

    for key in ("minimum", "maximum"):
        if key in schema:
            bound = schema[key]
            if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound):
                raise ToolContractError(f"{path}.{key}: must be finite numeric")
    for key in ("minLength", "maxLength"):
        if key in schema:
            bound = schema[key]
            if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
                raise ToolContractError(f"{path}.{key}: must be a non-negative integer")


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, {"string": str, "array": list, "object": dict}[expected])


def _instance(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    expected = schema["type"]
    if not _type_ok(value, expected):
        raise ToolContractError(f"{path}: expected {expected}, got {type(value).__name__}")
    _json_value(value, path)
    if "enum" in schema and value not in schema["enum"]:
        raise ToolContractError(f"{path}: value is outside declared enum")
    if expected in {"integer", "number"}:
        number = float(value)
        if "minimum" in schema and number < float(schema["minimum"]):
            raise ToolContractError(f"{path}: value is below minimum")
        if "maximum" in schema and number > float(schema["maximum"]):
            raise ToolContractError(f"{path}: value exceeds maximum")
    elif expected == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolContractError(f"{path}: string shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolContractError(f"{path}: string longer than maxLength")
    elif expected == "array":
        for index, item in enumerate(value):
            _instance(item, schema["items"], f"{path}[{index}]")
    elif expected == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        extra = set(value) - set(properties)
        if missing:
            raise ToolContractError(f"{path}: missing required properties {sorted(missing)!r}")
        if extra:
            raise ToolContractError(f"{path}: undeclared properties {sorted(extra)!r}")
        for name, item in value.items():
            _instance(item, properties[name], f"{path}.{name}")


def _permissions(values: tuple[str, ...]) -> None:
    if tuple(sorted(set(values))) != values:
        raise ToolContractError("permissions must be sorted and unique")
    if any(_PERMISSION_RE.fullmatch(item) is None for item in values):
        raise ToolContractError("invalid permission scope")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    required_permissions: tuple[str, ...]
    side_effect: SideEffect
    requires_confirmation: bool
    provider: str
    provider_identity: str
    title: str | None = None
    output_schema: dict[str, Any] | None = None
    idempotent: bool = False
    open_world: bool = False

    def __post_init__(self) -> None:
        if _TOOL_NAME_RE.fullmatch(self.name) is None:
            raise ToolContractError(f"invalid MCP tool name: {self.name!r}")
        if not self.description.strip() or not self.provider.strip() or not self.provider_identity.strip():
            raise ToolContractError("description/provider/provider_identity must be non-empty")
        _permissions(self.required_permissions)
        _schema(self.input_schema)
        if self.output_schema is not None:
            _schema(self.output_schema, "outputSchema")
        if self.side_effect is SideEffect.IRREVERSIBLE_WRITE and not self.requires_confirmation:
            raise ToolContractError("irreversible tools require confirmation")

    def contract_payload(self) -> dict[str, Any]:
        return {
            "name": self.name, "title": self.title, "description": self.description,
            "input_schema": self.input_schema, "output_schema": self.output_schema,
            "required_permissions": list(self.required_permissions), "side_effect": self.side_effect.value,
            "requires_confirmation": self.requires_confirmation, "provider": self.provider,
            "provider_identity": self.provider_identity, "idempotent": self.idempotent,
            "open_world": self.open_world, "schema_profile": _SCHEMA_PROFILE,
        }

    @property
    def identity(self) -> str:
        return stable_identity(self.contract_payload())

    def to_mcp_tool(self) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "readOnlyHint": self.side_effect is SideEffect.READ_ONLY,
                "destructiveHint": self.side_effect is SideEffect.IRREVERSIBLE_WRITE,
                "idempotentHint": self.idempotent,
                "openWorldHint": self.open_world,
            },
            "_meta": {PROJECT_META_KEY: {
                "contractIdentity": self.identity,
                "requiredPermissions": list(self.required_permissions),
                "sideEffect": self.side_effect.value,
                "requiresConfirmation": self.requires_confirmation,
                "provider": self.provider,
                "providerIdentity": self.provider_identity,
                "schemaProfile": _SCHEMA_PROFILE,
            }},
        }
        if self.title is not None:
            tool["title"] = self.title
        if self.output_schema is not None:
            tool["outputSchema"] = self.output_schema
        return tool


@dataclass(frozen=True)
class AuthorizationContext:
    caller_id: str
    granted_permissions: tuple[str, ...]
    confirmation_ref: str | None = None
    confirmation_request_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.caller_id.strip():
            raise ToolContractError("caller_id must be non-empty")
        _permissions(self.granted_permissions)
        if (self.confirmation_ref is None) != (self.confirmation_request_identity is None):
            raise ToolContractError("confirmation reference and request identity must appear together")


@dataclass(frozen=True)
class ValidatedToolCall:
    request_id: str | int
    tool_name: str
    arguments: dict[str, Any]
    caller_id: str
    permissions: tuple[str, ...]
    tool_contract_identity: str
    confirmation_ref: str | None
    client_info: dict[str, Any]

    @property
    def identity(self) -> str:
        return stable_identity({
            "request_id": self.request_id, "tool_name": self.tool_name, "arguments": self.arguments,
            "caller_id": self.caller_id, "permissions": list(self.permissions),
            "tool_contract_identity": self.tool_contract_identity, "client_info": self.client_info,
        })


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    authority: str
    identity: str

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.authority.strip() or not self.identity.strip():
            raise ToolContractError("evidence kind, authority and identity must be non-empty")

    def payload(self) -> dict[str, str]:
        return {"kind": self.kind, "authority": self.authority, "identity": self.identity}


@dataclass(frozen=True)
class ToolExecutionResult:
    call_identity: str
    tool_name: str
    ok: bool
    structured_content: dict[str, Any] | None
    evidence: tuple[EvidenceRef, ...]
    error_code: str | None = None
    message: str | None = None


class ToolRegistry:
    def __init__(self, specs: tuple[ToolSpec, ...]) -> None:
        if len({spec.name for spec in specs}) != len(specs):
            raise ToolContractError("tool names must be unique")
        self._specs = {spec.name: spec for spec in specs}

    @property
    def identity(self) -> str:
        return stable_identity([self._specs[name].contract_payload() for name in sorted(self._specs)])

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise ToolContractError(f"unknown tool: {name!r}")
        return self._specs[name]

    def mcp_tools(self) -> list[dict[str, Any]]:
        return [self._specs[name].to_mcp_tool() for name in sorted(self._specs)]


def _meta(params: dict[str, Any]) -> dict[str, Any]:
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        raise ToolContractError("MCP request params._meta is required")
    if meta.get("io.modelcontextprotocol/protocolVersion") != MCP_PROTOCOL_VERSION:
        raise ToolContractError("unsupported or missing MCP protocolVersion")
    info = meta.get("io.modelcontextprotocol/clientInfo")
    if not isinstance(info, dict) or not isinstance(info.get("name"), str) or not info["name"].strip():
        raise ToolContractError("MCP clientInfo.name is required")
    if not isinstance(info.get("version"), str) or not info["version"].strip():
        raise ToolContractError("MCP clientInfo.version is required")
    if not isinstance(meta.get("io.modelcontextprotocol/clientCapabilities"), dict):
        raise ToolContractError("MCP clientCapabilities is required")
    return info


def _envelope(request: Any, method: str) -> tuple[str | int, dict[str, Any]]:
    if not isinstance(request, dict):
        raise ToolContractError("MCP request must be an object")
    extra = set(request) - {"jsonrpc", "id", "method", "params"}
    if extra:
        raise ToolContractError(f"MCP request has undeclared top-level fields: {sorted(extra)!r}")
    if request.get("jsonrpc") != "2.0" or request.get("method") != method:
        raise ToolContractError(f"expected JSON-RPC 2.0 method {method!r}")
    request_id = request.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise ToolContractError("MCP request id must be a string or integer")
    params = request.get("params")
    if not isinstance(params, dict):
        raise ToolContractError("MCP request params must be an object")
    return request_id, params


def mcp_tools_list_response(registry: ToolRegistry, request: Any) -> dict[str, Any]:
    request_id, params = _envelope(request, "tools/list")
    if set(params) - {"_meta", "cursor"}:
        raise ToolContractError("tools/list has unsupported params")
    _meta(params)
    if params.get("cursor") not in (None, ""):
        raise ToolContractError("pagination cursor is not supported by project profile v1")
    return {"jsonrpc": "2.0", "id": request_id, "result": {
        "resultType": "complete", "tools": registry.mcp_tools(), "ttlMs": 0,
        "cacheScope": "private", "_meta": {PROJECT_META_KEY: {"registryIdentity": registry.identity}},
    }}


def _call_base(
    registry: ToolRegistry, request: Any, context: AuthorizationContext
) -> tuple[ToolSpec, ValidatedToolCall]:
    request_id, params = _envelope(request, "tools/call")
    extra = set(params) - {"_meta", "name", "arguments"}
    if extra:
        raise ToolContractError(
            "project profile v1 does not authorize MRTR/inputResponses/requestState or other "
            f"tools/call extensions: {sorted(extra)!r}"
        )
    client_info = _meta(params)
    name = params.get("name")
    if not isinstance(name, str):
        raise ToolContractError("tools/call name must be a string")
    spec = registry.get(name)
    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ToolContractError("tools/call arguments must be an object")
    _instance(arguments, spec.input_schema)
    if context.granted_permissions != spec.required_permissions:
        raise ToolContractError("granted permissions must exactly match the tool's declared least-privilege set")
    return spec, ValidatedToolCall(
        request_id, name, arguments, context.caller_id, context.granted_permissions,
        spec.identity, context.confirmation_ref, client_info,
    )


def preview_mcp_tool_call_identity(
    registry: ToolRegistry,
    request: Any,
    *,
    caller_id: str,
    granted_permissions: tuple[str, ...],
) -> str:
    """Return a validation-only structural digest a trusted host may ask a user to confirm."""
    _, call = _call_base(
        registry, request, AuthorizationContext(caller_id, granted_permissions)
    )
    return call.identity


def authorize_mcp_tool_call(
    registry: ToolRegistry, request: Any, context: AuthorizationContext
) -> ValidatedToolCall:
    spec, call = _call_base(registry, request, context)
    if spec.requires_confirmation:
        if context.confirmation_ref is None:
            raise ToolContractError("tool requires host-verified confirmation evidence")
        if context.confirmation_request_identity != call.identity:
            raise ToolContractError("confirmation evidence is not bound to this exact call")
    elif context.confirmation_ref is not None:
        raise ToolContractError("unexpected confirmation evidence for a non-confirming tool")
    return call


def validate_tool_result(
    registry: ToolRegistry, call: ValidatedToolCall, result: ToolExecutionResult
) -> None:
    spec = registry.get(call.tool_name)
    if call.tool_contract_identity != spec.identity:
        raise ToolContractError("tool contract changed after authorization")
    if result.call_identity != call.identity or result.tool_name != call.tool_name:
        raise ToolContractError("tool result is not bound to the authorized call")
    if result.ok:
        if result.error_code is not None or not result.evidence or result.structured_content is None:
            raise ToolContractError("successful result requires provenance/evidence and structured_content")
        _json_value(result.structured_content, "structured_content")
        if spec.output_schema is not None:
            _instance(result.structured_content, spec.output_schema, "structured_content")
    elif result.error_code is None or not result.error_code.strip():
        raise ToolContractError("failed result requires error_code")


def mcp_tool_call_response(
    registry: ToolRegistry, call: ValidatedToolCall, result: ToolExecutionResult
) -> dict[str, Any]:
    validate_tool_result(registry, call, result)
    payload: dict[str, Any] = {
        "resultType": "complete",
        "content": [{"type": "text", "text": result.message or "tool completed"}],
        "isError": not result.ok,
        "_meta": {PROJECT_META_KEY: {
            "callIdentity": call.identity,
            "toolContractIdentity": call.tool_contract_identity,
            "evidence": [item.payload() for item in result.evidence],
            "errorCode": result.error_code,
        }},
    }
    if result.structured_content is not None:
        payload["structuredContent"] = result.structured_content
    return {"jsonrpc": "2.0", "id": call.request_id, "result": payload}


def validate_contract_manifest(manifest: Any) -> str:
    if not isinstance(manifest, dict):
        raise ToolContractError("manifest must be an object")
    required = {
        "schema_version", "status", "protocol", "protocol_version", "upstream",
        "project_profile", "security_boundaries", "truth_boundaries",
    }
    missing = required - set(manifest)
    if missing:
        raise ToolContractError(f"manifest missing fields: {sorted(missing)!r}")
    if manifest["schema_version"] != 1 or manifest["status"] != "CANDIDATE":
        raise ToolContractError("manifest must be schema v1 CANDIDATE")
    if manifest["protocol"] != "MCP" or manifest["protocol_version"] != MCP_PROTOCOL_VERSION:
        raise ToolContractError("manifest protocol identity mismatch")
    upstream = manifest["upstream"]
    expected = {
        "repository": MCP_UPSTREAM_REPOSITORY,
        "tag": MCP_UPSTREAM_TAG,
        "commit": MCP_UPSTREAM_COMMIT,
    }
    if not isinstance(upstream, dict) or any(upstream.get(k) != v for k, v in expected.items()):
        raise ToolContractError("manifest upstream identity mismatch")
    if upstream.get("license_state") != "APACHE-2.0_MIT_TRANSITION;DOCS_CC-BY-4.0":
        raise ToolContractError("manifest must preserve upstream license-transition truth")
    profile = manifest["project_profile"]
    if not isinstance(profile, dict) or profile.get("schema_profile") != _SCHEMA_PROFILE:
        raise ToolContractError("project schema profile mismatch")
    if profile.get("supported_methods") != ["tools/list", "tools/call"]:
        raise ToolContractError("supported MCP methods must be explicit and stable")
    if profile.get("full_mcp_conformance_claimed") is not False:
        raise ToolContractError("project profile must not claim full MCP conformance")
    if not isinstance(manifest["security_boundaries"], list) or not manifest["security_boundaries"]:
        raise ToolContractError("security_boundaries must be a non-empty array")
    if not isinstance(manifest["truth_boundaries"], list) or not manifest["truth_boundaries"]:
        raise ToolContractError("truth_boundaries must be a non-empty array")
    _json_value(manifest)
    return stable_identity(manifest)
