# Plan 5 / Section 7 — Typed Tool Registry v1

**Owner:** Plan 5 / agent-facing contracts. Conflict key: `tool-contracts`.
This package does **not** implement a Nika product UI, OS/browser adapter,
privileged executor, cloud invocation, paid compute or ModelGateway.

## Authoritative boundary

`src/twelve_six_agent_runtime/tools.py` defines the stable
`12-6.agent-typed-tools.v1` reference boundary with
`12-6.tool-descriptor.v1` tool manifests. This is compatible with the
independent `TaskStore` from Plan 5 Section 2.

1. **Register:** only a trusted host can verify a typed, versioned
   `ToolDescriptor`. The registry stores an immutable canonical JSON
   snapshot, not the caller's mutable dictionary. Duplicate IDs, unsupported
   versions, malformed or unbounded schemas are rejected.
2. **Discover:** returns read-only descriptions: declared capabilities,
   permissions, input/output schemas and side-effect class.
   Discovery does **not** grant permissions, reserve effects, execute tools,
   expose credentials, or authorize real-world operations.
3. **Prepare:** requires exact `TaskStore` task ID, control epoch, revision
   and a previously reserved `pending` effect ID; a current available
   descriptor; closed input schema; exact least-privilege permission set;
   and host-verified grant evidence bound to the descriptor, task,
   **specific input arguments** and grant identity. The resulting immutable
   `ToolCall` includes those bindings and a canonical SHA-256 call ID.
   It is an intent envelope, **not** an executor command.
4. **External effect:** the trusted host must atomically use
   `TaskStore.issue_effect` before calling the actual adapter, then preserve
   the evidence. Unknown outcomes cannot be repeated blindly on restart.
   This module cannot dispatch the effect or access credentials.
5. **Accept result:** a typed `ToolResult` binds the exact call, task,
   effect, tool, receipt, outcome and evidence. The result must conform to
   the exact registered output schema and have **independently host-verified
   call and result proofs**. A stale epoch, mismatched descriptor, unissued
   effect, changed envelope, duplicate completion or forged identity fails
   closed. Only `TaskStore.resolve_effect` can durably resolve the effect.

The supported JSON schema subset is deliberate and fail-closed:
closed `object` with explicit `properties` and `required`;
bounded `string`, `array`, `integer`, finite `number`, and
strict `boolean`. Unknown JSON Schema extensions are **rejected**,
not silently ignored. The reference boundary is deterministic and
provider-neutral.

## Recovery, evidence and trust limitations

- Task/control/receipt durability belongs to existing `TaskStore`.
- Output evidence can be retained by a trusted host as the canonical
  `ToolResult` envelope; the SHA-256 ID is an integrity checksum,
  **not** an independent signature. The host verification callbacks must
  validate provenance, grant and the effect's external execution receipt.
- The registry has no arbitrary Python plugin loader, no MCP transport
  implementation, and no pass-through API key or secret handling.
- Tests use fake host verifiers and filesystem-free fixture tools; this is
  component-level proof only. Production connector/browser/computer
  integration is separately owned by Plan 5 Section 8 and Plan 10.

## LOCAL_FREE acceptance checks

`PYTHONPATH=src python -m compileall -q src/twelve_six_agent_runtime`

`PYTHONPATH=src pytest -q tests/test_agent_tools_section7.py
tests/test_agent_task_state_section2.py`

Coverage includes strict schema admission, no discovery grant, unknown and
unavailable tools, duplicate registration, mutable descriptor freeze,
least-privilege verification, forged result/receipt/call bindings,
wrong output types, stale control epoch, duplicate outcome, unknown-effect
no-blind-retry, and durable process-restart task state.

No real external effects, paid compute, model training or whole-product
integration are authorized by this contract.
