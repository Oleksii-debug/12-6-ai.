# MCP-compatible Tool Contract V1

Status: `CANDIDATE`  
Swarm owner: `SWARM-753`  
Lane key: `D09|TOOL-PROTOCOL|OPEN-SOURCE-REUSE-RESEARCH|MCP-COMPAT-V1`

## Purpose

This package establishes a project-owned, fail-closed contract between future 12-6 agent logic and external tool adapters. It deliberately does **not** make tool behavior part of canonical Base weights and does not authorize model training, a tool backend, paid compute, or production deployment.

The compatibility target is the stable MCP specification `2026-07-28`, upstream repository `modelcontextprotocol/modelcontextprotocol`, stable tag `2026-07-28`, exact release commit `5f5440bb26a62e2cf3440b92da5a667efa03b267`.

The exact tagged upstream `LICENSE` records a licensing transition: new code/spec contributions are Apache-2.0, historical contributions without relicensing consent remain MIT, and non-spec documentation is CC-BY-4.0. This project reimplements a small compatibility boundary rather than vendoring upstream code.

## Why a project-owned contract exists

MCP standardizes discovery and invocation, but protocol metadata is not an authorization system. The 2026-07-28 specification explicitly treats tool annotations as untrusted unless they come from a trusted server and recommends human control over invocations. The 12-6 boundary is therefore stricter than the wire protocol:

1. Trusted project code defines every executable `ToolSpec`.
2. A spec has an immutable identity over its schema, provider identity, permissions and side-effect semantics.
3. The host supplies authorization context separately from the MCP request. Permission scopes are never accepted from tool arguments or untrusted wire metadata.
4. The granted permission set must exactly equal the spec's sorted least-privilege set. Over-granting fails closed rather than silently widening authority.
5. Irreversible tools require host-verified confirmation evidence bound to the exact call identity. The MCP request cannot self-assert that confirmation.
6. A validated call is still only an authorization artifact; it is not evidence that an action ran.
7. A successful result must bind the exact call and carry non-empty provenance/evidence before it can be serialized as MCP success.

## Supported MCP profile

V1 intentionally implements a bounded compatibility profile, not full MCP conformance.

Supported:
- JSON-RPC 2.0 `tools/list`;
- JSON-RPC 2.0 `tools/call`;
- mandatory 2026-07-28 per-request `_meta` protocol version, client info and client capabilities;
- deterministic tool ordering;
- MCP tool schemas and standard annotations;
- project-owned namespaced `_meta` for immutable contract identity and permission/provenance semantics;
- `resultType=complete` responses and private zero-TTL tool-list caching semantics.

Explicitly not supported in V1:
- list pagination beyond a complete single page;
- Multi Round-Trip Request fields such as `inputResponses` and `requestState`;
- subscriptions/list-changed transport behavior;
- resources, prompts, elicitation, tasks or extensions;
- network transport implementation;
- backend execution;
- full JSON Schema 2020-12 execution semantics.

The project executable schema profile is a deterministic safe subset: explicit scalar/object/array types, nested properties/items, required fields, closed objects (`additionalProperties=false`), enum and bounded numeric/string constraints. Unsupported schema keywords fail closed. An adapter that needs broader JSON Schema must extend this profile with tests rather than silently accepting semantics the project cannot validate.

## MCP 2026-07-28 compatibility notes

The target release is stateless: the older initialization/session model was removed and each request carries protocol/client metadata. This implementation therefore rejects stale protocol versions and does not introduce a hidden connection-level authority cache.

`ToolSpec.to_mcp_tool()` emits standard MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) for interoperability. Those hints do not grant permission. The authoritative project semantics live in the trusted `ToolSpec` and are mirrored under `io.twelve-six/toolContract` only for evidence/debugging.

## Confirmation boundary

`AuthorizationContext.confirmation_ref` is not a token validator and is not derived from MCP wire input. It represents confirmation that a trusted host has already verified through its own UI/policy layer. The paired `confirmation_request_identity` must equal the exact call identity. A trusted host can obtain that structural fingerprint through `preview_mcp_tool_call_identity(...)`; the helper validates request structure, schema and least privilege, but it is not authorization and cannot execute a backend. A future UI adapter must own authenticity, expiry, replay prevention and actor verification; this contract only enforces structural binding and refuses to fabricate those guarantees.

## Provenance boundary

A backend success must provide at least one `EvidenceRef(kind, authority, identity)`. The contract does not decide whether an evidence authority is scientifically or operationally sufficient for every domain. Domain adapters remain responsible for stronger evidence rules. The generic boundary only prevents an unbound or evidence-free success from being represented as an accepted tool result.

## Validation

Local/free validation commands:

```text
PYTHONPATH=src python tools/validate_mcp_tool_contract_v1.py
PYTHONPATH=src pytest -q tests/test_mcp_tool_contract_v1.py
ruff check src/twelve_six/tool_protocol.py tests/test_mcp_tool_contract_v1.py
```

The focused tests cover deterministic listing, exact least privilege, unknown tools, closed argument schemas, MCP metadata/version binding, unsupported MRTR fields, irreversible-operation confirmation, result/provenance binding, output-schema failures, contract-identity drift, invalid tool contracts and machine-manifest truth boundaries.

## Promotion and rollback

This package is `CANDIDATE`, not `PARITY_PROVEN` or `ADOPTED`. Promotion requires at minimum: integration with one project-owned adapter, independent negative/security review, real MCP conformance testing for the claimed subset, explicit host confirmation implementation, and evidence that adapter permission semantics cannot be bypassed. Any broader MCP feature must be added explicitly and tested.

Rollback is additive: remove the Tool Contract module/config/tests/docs and keep canonical Base/data/training artifacts untouched. No checkpoint or data migration is required.

## Truth boundary

No external MCP server was contacted by this implementation package. No tool backend ran. No Base weights, tokenizer, dataset, checkpoint, optimizer, training recipe, evaluation payload or CI workflow was changed. No model training, optimizer update, GPU evidence, final-test access or paid compute occurred. Compatibility-shape validation is not a security certification or production-readiness claim.
