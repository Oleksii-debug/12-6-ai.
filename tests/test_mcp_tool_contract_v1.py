from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.tool_protocol import (
    MCP_PROTOCOL_VERSION,
    AuthorizationContext,
    EvidenceRef,
    SideEffect,
    ToolContractError,
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
    authorize_mcp_tool_call,
    mcp_tool_call_response,
    mcp_tools_list_response,
    preview_mcp_tool_call_identity,
    stable_identity,
    validate_contract_manifest,
)


def _meta() -> dict[str, object]:
    return {
        "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": {"name": "pytest-host", "version": "1.0"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }


def _read_tool() -> ToolSpec:
    return ToolSpec(
        name="repo.read_status",
        title="Read repository status",
        description="Return a bounded repository status snapshot.",
        input_schema={
            "type": "object",
            "properties": {"ref": {"type": "string", "minLength": 1, "maxLength": 80}},
            "required": ["ref"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"sha": {"type": "string", "minLength": 40, "maxLength": 40}},
            "required": ["sha"],
            "additionalProperties": False,
        },
        required_permissions=("repo.read",),
        side_effect=SideEffect.READ_ONLY,
        requires_confirmation=False,
        provider="github-adapter",
        provider_identity="adapter-v1",
    )


def _delete_tool() -> ToolSpec:
    return ToolSpec(
        name="repo.delete_branch",
        description="Delete an explicitly named temporary branch.",
        input_schema={
            "type": "object",
            "properties": {"branch": {"type": "string", "minLength": 1, "maxLength": 120}},
            "required": ["branch"],
            "additionalProperties": False,
        },
        required_permissions=("repo.write",),
        side_effect=SideEffect.IRREVERSIBLE_WRITE,
        requires_confirmation=True,
        provider="github-adapter",
        provider_identity="adapter-v1",
    )


def _call(name: str, arguments: dict[str, object], request_id: int = 7) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"_meta": _meta(), "name": name, "arguments": arguments},
    }


def test_registry_and_tools_list_are_deterministic_and_project_authoritative() -> None:
    registry = ToolRegistry((_delete_tool(), _read_tool()))
    request = {
        "jsonrpc": "2.0",
        "id": "list-1",
        "method": "tools/list",
        "params": {"_meta": _meta()},
    }
    first = mcp_tools_list_response(registry, request)
    second = mcp_tools_list_response(registry, request)
    assert first == second
    names = [tool["name"] for tool in first["result"]["tools"]]
    assert names == sorted(names)
    delete_meta = first["result"]["tools"][0]["_meta"]["io.twelve-six/toolContract"]
    assert delete_meta["requiresConfirmation"] is True
    assert delete_meta["requiredPermissions"] == ["repo.write"]
    assert first["result"]["_meta"]["io.twelve-six/toolContract"]["registryIdentity"]


def test_read_call_authorizes_only_exact_least_privilege_and_valid_schema() -> None:
    registry = ToolRegistry((_read_tool(),))
    call = authorize_mcp_tool_call(
        registry,
        _call("repo.read_status", {"ref": "main"}),
        AuthorizationContext(caller_id="operator", granted_permissions=("repo.read",)),
    )
    assert call.tool_name == "repo.read_status"
    assert len(call.identity) == 64

    with pytest.raises(ToolContractError, match="least-privilege"):
        authorize_mcp_tool_call(
            registry,
            _call("repo.read_status", {"ref": "main"}),
            AuthorizationContext(
                caller_id="operator",
                granted_permissions=("repo.read", "repo.write"),
            ),
        )
    with pytest.raises(ToolContractError, match="undeclared properties"):
        authorize_mcp_tool_call(
            registry,
            _call("repo.read_status", {"ref": "main", "force": True}),
            AuthorizationContext(caller_id="operator", granted_permissions=("repo.read",)),
        )


def test_unknown_tool_and_unsupported_mrtr_fields_fail_closed() -> None:
    registry = ToolRegistry((_read_tool(),))
    with pytest.raises(ToolContractError, match="unknown tool"):
        authorize_mcp_tool_call(
            registry,
            _call("repo.unknown", {}),
            AuthorizationContext(caller_id="operator", granted_permissions=()),
        )
    request = _call("repo.read_status", {"ref": "main"})
    request["params"]["requestState"] = "opaque"
    with pytest.raises(ToolContractError, match="does not authorize MRTR"):
        authorize_mcp_tool_call(
            registry,
            request,
            AuthorizationContext(caller_id="operator", granted_permissions=("repo.read",)),
        )


def test_protocol_metadata_is_mandatory_and_version_bound() -> None:
    registry = ToolRegistry((_read_tool(),))
    request = _call("repo.read_status", {"ref": "main"})
    del request["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"]
    with pytest.raises(ToolContractError, match="clientCapabilities"):
        authorize_mcp_tool_call(
            registry,
            request,
            AuthorizationContext(caller_id="operator", granted_permissions=("repo.read",)),
        )
    request = _call("repo.read_status", {"ref": "main"})
    request["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = "2025-11-25"
    with pytest.raises(ToolContractError, match="protocolVersion"):
        authorize_mcp_tool_call(
            registry,
            request,
            AuthorizationContext(caller_id="operator", granted_permissions=("repo.read",)),
        )


def test_irreversible_call_needs_host_verified_confirmation_bound_to_exact_call() -> None:
    registry = ToolRegistry((_delete_tool(),))
    request = _call("repo.delete_branch", {"branch": "tmp/test"})
    with pytest.raises(ToolContractError, match="confirmation"):
        authorize_mcp_tool_call(
            registry,
            request,
            AuthorizationContext(caller_id="operator", granted_permissions=("repo.write",)),
        )

    unconfirmed_call = authorize_mcp_tool_call
    provisional_context = AuthorizationContext(
        caller_id="operator",
        granted_permissions=("repo.write",),
        confirmation_ref="ui-confirmation:123",
        confirmation_request_identity="wrong",
    )
    with pytest.raises(ToolContractError, match="not bound"):
        unconfirmed_call(registry, request, provisional_context)

    # The preview is validation-only and returns the exact structural digest a trusted host can
    # present for confirmation without executing or authorizing the tool.
    expected_identity = preview_mcp_tool_call_identity(
        registry,
        request,
        caller_id="operator",
        granted_permissions=("repo.write",),
    )
    call = authorize_mcp_tool_call(
        registry,
        request,
        AuthorizationContext(
            caller_id="operator",
            granted_permissions=("repo.write",),
            confirmation_ref="ui-confirmation:123",
            confirmation_request_identity=expected_identity,
        ),
    )
    assert call.confirmation_ref == "ui-confirmation:123"


def test_success_result_requires_exact_binding_schema_and_evidence() -> None:
    registry = ToolRegistry((_read_tool(),))
    call = authorize_mcp_tool_call(
        registry,
        _call("repo.read_status", {"ref": "main"}),
        AuthorizationContext(caller_id="operator", granted_permissions=("repo.read",)),
    )
    result = ToolExecutionResult(
        call_identity=call.identity,
        tool_name=call.tool_name,
        ok=True,
        structured_content={"sha": "a" * 40},
        evidence=(EvidenceRef(kind="git", authority="github", identity="a" * 40),),
        message="status read",
    )
    response = mcp_tool_call_response(registry, call, result)
    assert response["result"]["isError"] is False
    assert response["result"]["structuredContent"] == {"sha": "a" * 40}

    no_evidence = ToolExecutionResult(
        call_identity=call.identity,
        tool_name=call.tool_name,
        ok=True,
        structured_content={"sha": "a" * 40},
        evidence=(),
    )
    with pytest.raises(ToolContractError, match="provenance/evidence"):
        mcp_tool_call_response(registry, call, no_evidence)

    wrong_schema = ToolExecutionResult(
        call_identity=call.identity,
        tool_name=call.tool_name,
        ok=True,
        structured_content={"sha": "short"},
        evidence=(EvidenceRef(kind="git", authority="github", identity="a" * 40),),
    )
    with pytest.raises(ToolContractError, match="minLength"):
        mcp_tool_call_response(registry, call, wrong_schema)


def test_contract_identity_changes_on_material_permission_drift() -> None:
    original = _read_tool()
    drifted = ToolSpec(
        name=original.name,
        title=original.title,
        description=original.description,
        input_schema=original.input_schema,
        output_schema=original.output_schema,
        required_permissions=("repo.admin",),
        side_effect=original.side_effect,
        requires_confirmation=original.requires_confirmation,
        provider=original.provider,
        provider_identity=original.provider_identity,
    )
    assert original.identity != drifted.identity
    assert stable_identity(original.contract_payload()) == original.identity


def test_invalid_tool_contracts_fail_closed() -> None:
    with pytest.raises(ToolContractError, match="irreversible"):
        ToolSpec(
            name="dangerous",
            description="bad",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            required_permissions=("repo.write",),
            side_effect=SideEffect.IRREVERSIBLE_WRITE,
            requires_confirmation=False,
            provider="x",
            provider_identity="y",
        )
    with pytest.raises(ToolContractError, match="additionalProperties=false"):
        ToolSpec(
            name="loose",
            description="bad schema",
            input_schema={"type": "object", "properties": {}},
            required_permissions=(),
            side_effect=SideEffect.READ_ONLY,
            requires_confirmation=False,
            provider="x",
            provider_identity="y",
        )


def test_schema_keyword_types_and_optional_sensitive_read_confirmation_are_explicit() -> None:
    with pytest.raises(ToolContractError, match="numeric-only"):
        ToolSpec(
            name="bad.schema",
            description="bad keyword placement",
            input_schema={
                "type": "object",
                "properties": {"ref": {"type": "string", "minimum": 1}},
                "additionalProperties": False,
            },
            required_permissions=(),
            side_effect=SideEffect.READ_ONLY,
            requires_confirmation=False,
            provider="x",
            provider_identity="y",
        )

    sensitive = ToolSpec(
        name="repo.read_secret_metadata",
        description="Read bounded sensitive metadata only after explicit host confirmation.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        required_permissions=("repo.read",),
        side_effect=SideEffect.READ_ONLY,
        requires_confirmation=True,
        provider="x",
        provider_identity="y",
    )
    assert sensitive.requires_confirmation is True


def test_machine_manifest_preserves_upstream_and_truth_boundaries() -> None:
    manifest_path = Path(__file__).parents[1] / "configs/postbase/mcp_tool_contract_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = validate_contract_manifest(manifest)
    assert len(identity) == 64
    assert manifest["upstream"]["commit"] == "5f5440bb26a62e2cf3440b92da5a667efa03b267"
    assert manifest["project_profile"]["full_mcp_conformance_claimed"] is False
