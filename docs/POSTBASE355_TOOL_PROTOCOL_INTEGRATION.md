# POSTBASE-355 tool-protocol integration

Worker: `POSTBASE-355-TOOL-PROTOCOL-INTEGRATION`

Convergence authority: `NEXT100-086-TOOL-PROTOCOL-TERMINAL`

Status: terminal mechanics candidate. This scope proves deterministic protocol mechanics only. It does not train or modify Base/post-Base weights and makes no claim that a learned model has acquired tool-use or reasoning quality.

## Integration boundary

`src/twelve_six/postbase/model_tool_integration.py` connects the maintained first-party `InferenceBackend` generation path to the accepted POSTBASE-254 protocol without adding a second inference runtime.

The successful one-tool trace is explicit and ordered:

1. `model_request` — first-party generation emits a strict JSON model envelope containing raw candidate requests. No tool is executed here.
2. `validation` — every candidate is passed through POSTBASE-254 `parse_tool_request`. The complete batch must validate before any execution begins.
3. `tool_execution` — validated requests are handed only to POSTBASE-254 `MockExecutor`.
4. `tool_observation` — each `ToolResult` becomes a `ToolObservation` with `trusted_as_instruction=false`, `training_eligible=false`, and `weight_update_eligible=false`.
5. `final_response` — a second, separate first-party generation turn receives observations as a canonical untrusted-data bundle and returns final response text. The resulting `FinalAnswer` references exact observation IDs.

`IntegrationRun` preserves these stages in an ordinal trace and exposes a deterministic SHA-256 identity over the canonical run representation.

## Model request and schema validation

`decode_model_generation` validates only the strict model wire envelope and canonical JSON types. It deliberately does not grant authority to a tool name or validate tool-specific arguments.

The separate validation phase calls POSTBASE-254 `parse_tool_request`, which validates the request envelope, registered tool name, strict argument keys/types/bounds, timeout, output budget, filesystem path policy, and Python policy. An unknown or policy-denied request therefore cannot become executable merely because it was emitted by the model.

## Atomic batch validation

`ToolProtocolIntegration._validate_all` validates the complete candidate batch before its execution loop begins. The adversarial fixture places an otherwise valid in-memory filesystem write first and an unregistered `shell.exec` request second. Validation fails on the second request while the mock filesystem remains empty and the final-response turn is never entered.

This establishes all-or-nothing validation for a generated request batch with respect to tool side effects.

## Mock-only execution and shell exclusion

POSTBASE-355 accepts only POSTBASE-254 `MockExecutor` for tool execution. Its registered tools remain fixture-backed or local deterministic mocks. In particular:

- there is no `shell.exec` registration;
- model-supplied Python is policy-checked and hashed but never executed by the mock;
- filesystem access is an in-memory dictionary sandbox only;
- web/document/API adapters are fixture-backed only;
- no network-backed tool adapter is added;
- no external LLM adapter is accepted.

The terminal qualification does not invoke a real shell tool, mutate a real filesystem through the protocol, contact a real web endpoint, or call a real external API.

## Observation and final-response isolation

The final model prompt marks the observation bundle as untrusted data. `ToolObservation` and result provenance independently carry `training_eligible=false` and `weight_update_eligible=false`.

Hostile instruction-shaped observation text is canonically JSON-escaped and remains data inside the observation bundle. It is never reparsed as a new request. A dedicated fixture also injects fields that falsely claim `content_class=base_training_evidence`, `trusted_as_instruction=true`, `training_eligible=true`, and `weight_update_eligible=true`. Those attacker-controlled fields remain nested tool output only; the authoritative observation and provenance metadata remain non-training, non-weight-updating tool-observation state.

The final response is generated only after tool execution/observation completes and is stored as `FinalAnswer`; the final text is not dispatched as another tool request.

## Adversarial terminal matrix

`tests/test_postbase355_tool_protocol_integration.py` covers:

- explicit model-request → validation → mock-execution → observation → final-response ordering;
- strict model wire JSON validation;
- tool argument schema validation before execution;
- atomic multi-request validation before any mock side effect;
- explicit `shell.exec` rejection;
- arbitrary unknown-tool rejection;
- filesystem path-traversal rejection;
- hostile observation text remaining untrusted data;
- attempted training-evidence/weight-update injection remaining nested untrusted output;
- deterministic repeated-run identity;
- deterministic greedy-only mechanics;
- external-LLM adapter rejection;
- explicit Base/post-Base inference-only lineage compatibility.

## Exact-head qualification procedure

The convergence authority must perform the final qualification after all repository writes:

1. refresh the live branch head;
2. fetch the protocol, integration, inference contracts/generation/sampling, and terminal test files from that exact head;
3. reconstruct the local test slice and verify each file byte-for-byte against its Git blob SHA;
4. run `PYTHONPATH=src pytest -q tests/test_postbase355_tool_protocol_integration.py` locally;
5. grant PASS only if the live head has not changed and the test suite is green.

The exact head SHA is intentionally not embedded in this document because committing such a value would itself create a new head. The convergence authority reports the final immutable SHA after the last live refresh.

Execution profile: `LOCAL_FREE`.
