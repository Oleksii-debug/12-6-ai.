# POSTBASE-355 tool-protocol integration

Worker: `POSTBASE-355-TOOL-PROTOCOL-INTEGRATION`

Status: deterministic post-Base mechanics only. This worker does not train or modify Base/post-Base weights and makes no claim that a learned model has acquired tool-use capability.

## Integration boundary

`src/twelve_six/postbase/model_tool_integration.py` connects the maintained first-party `InferenceBackend` generation path to the accepted POSTBASE-254 protocol without adding a second inference runtime.

The successful one-tool trace is explicit and ordered:

1. `model_request` — first-party generation emits a strict JSON model envelope containing raw candidate requests. No tool is executed here.
2. `validation` — every candidate is passed through POSTBASE-254 `parse_tool_request`. The complete batch must validate before any execution begins.
3. `tool_execution` — validated requests are handed only to POSTBASE-254 `MockExecutor`.
4. `tool_observation` — each `ToolResult` becomes a `ToolObservation` with `trusted_as_instruction=false`, `training_eligible=false`, and `weight_update_eligible=false`.
5. `final_response` — a second, separate first-party generation turn receives the observations as a canonical untrusted-data bundle and returns final response text. The resulting `FinalAnswer` references exact observation IDs.

`IntegrationRun` preserves these stages in an ordinal trace and exposes a deterministic SHA-256 identity over the canonical run representation.

## First-party Base/post-Base adapter

`FirstPartyBasePostBaseModelAdapter` consumes the existing maintained `InferenceBackend` interface used by first-party checkpoint inference. It accepts explicit `BASE` or `POST_BASE` lineage and performs inference only. The adapter does not write checkpoint files, mutate parameters, alter tokenizer identity, add a Base chat template, or move evidence between namespaces.

The request and final turns use greedy generation (`sample=false`) for this mechanics qualification. A model request is decoded as a wire envelope first, but tool-specific fields are deliberately not validated until the separate POSTBASE-254 validation step.

The dedicated POSTBASE-351 worker branch was not available while this integration was authored. Binding at the maintained `InferenceBackend` contract avoids inventing a private dependency and keeps the bridge directly compatible with the existing first-party runtime boundary.

## Mock-only execution and shell exclusion

POSTBASE-355 accepts only POSTBASE-254 `MockExecutor` for tool execution. Its registered tools remain fixture-backed or local deterministic mocks. In particular:

- there is no `shell.exec` registration;
- model-supplied Python is policy-checked and hashed but never executed by the mock;
- filesystem access remains the in-memory sandbox;
- web/document/API adapters remain fixture-backed;
- no network-backed tool adapter is added;
- no external LLM adapter is accepted.

The integration validates the entire request batch before executing any request. Therefore a later malformed or shell-like request cannot leave a side effect from an earlier otherwise-valid mock filesystem write.

## Observation handling

The final model prompt marks the observation bundle as untrusted data and serializes the POSTBASE-254 result/provenance records canonically. Tool-result text is never reparsed as a new tool request. Hostile instruction-shaped text inside a mock search result remains observation content; the controller does not recursively dispatch it.

## Tests

`tests/test_postbase355_tool_protocol_integration.py` provides project-authored deterministic fixtures for:

- exact request → validation → execution → observation → final-response stage ordering;
- calculator result propagation and observation-ID binding;
- all-requests-valid-before-any-side-effect behavior;
- explicit rejection of `shell.exec`;
- hostile instruction-shaped tool output remaining data only;
- deterministic repeated-run identity;
- shared inference-only mechanics for explicit Base and post-Base lineage;
- strict model-wire JSON handling, including non-finite rejection;
- rejection of sampling in this deterministic mechanics path;
- rejection of an adapter declaring external-LLM use.

The source and test files were syntax-compiled locally while authored. No unrestricted shell, external LLM, paid compute, or real external tool was used by this worker. Repository CI was not invoked as evidence in this worker, so no broader runtime or learned-capability claim is made here.

Execution profile: `LOCAL_FREE`.
