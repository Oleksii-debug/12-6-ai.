# POSTBASE-254 Tool-Use Execution Protocol

Worker: `POSTBASE-254-TOOL-USE-PROTOCOL`

Status: post-Base research infrastructure. This protocol is deliberately outside canonical Base model semantics and does not modify model weights, tokenizer identity, checkpoint format, training data, or generation math.

Base ancestry: `integrate222/learned-execution-spine-20260826` at `6afaf5889f9898037b53e8b0bc2b731d77782111`.

## Execution boundary

A tool-use cycle has four explicit, non-interchangeable phases:

1. `model_generation` — the model emits text and zero or more candidate tool requests.
2. `tool_execution` — a registered executor validates and executes one request and returns a bounded `ToolResult`.
3. `tool_observation` — the result is wrapped as untrusted observation content. It is not an instruction, training example, or weight-update input.
4. `final_answer` — the model may produce an answer that explicitly references observation identities.

The protocol never treats tool output as model parameters or training data. `Provenance` and `ToolObservation` hard-code `training_eligible=false` and `weight_update_eligible=false`.

## Request contract

Every request contains:

- protocol version;
- explicit request ID;
- explicit registered tool name;
- validated tool-specific arguments;
- timeout in milliseconds, bounded to 1–60,000 ms;
- maximum serialized output bytes, bounded to 1–65,536 bytes.

Top-level missing or unexpected fields fail as `malformed_request`. Unknown tools fail as `unknown_tool`. Tool arguments are a typed discriminated union selected by `ToolName` and revalidated at runtime.

Registered protocol names are:

| Tool | Arguments | Mock behavior |
| --- | --- | --- |
| `web.search` | query, top_k | fixture-backed search only |
| `document.retrieve` | document_id, optional query/max_chunks | fixture-backed retrieval only |
| `calculator` | arithmetic expression | local arithmetic AST evaluator |
| `python.execute` | code, optional JSON inputs | validates policy, records hash, never executes code |
| `filesystem.sandbox` | read/write/list + relative path | in-memory sandbox only |
| `api.call` | api_name, operation, JSON params | fixture-backed future API adapter |

There is no shell tool. `shell.exec`, subprocess imports, host-file access, and path traversal are denied by the protocol or unavailable in the mock executor.

## Result and error contract

Every `ToolResult` carries:

- request ID and tool name;
- explicit success/failure state;
- either output or a typed error, never both;
- executor/adapter provenance;
- request SHA-256;
- source references where available;
- output SHA-256 and observed serialized byte count where available.

Timeout and output-limit failures return no partial tool content. Unexpected adapter exceptions are converted into sanitized `execution_error` results without exposing internal exception data.

## Deterministic serialization

Protocol identities use canonical UTF-8 JSON with sorted object keys, compact separators, preserved Unicode, and non-finite floats rejected. Request hashes and observation IDs are SHA-256 over those canonical bytes. Equivalent argument objects therefore serialize identically regardless of input key order.

## Python/code safety boundary

The mock executor does not run model-supplied Python. It performs syntax/policy admission checks, rejects disallowed import roots and dangerous direct builtins, and emits only a code hash plus `executed=false`.

A future real Python adapter must use a separately reviewed process/container sandbox with explicit CPU, memory, filesystem, network, and wall-time limits. The AST policy in this protocol is defense in depth, not a substitute for an OS sandbox. Unrestricted shell execution is out of scope.

## Adversarial gates

The focused tests cover malformed requests, unknown shell-like tools, non-finite JSON, calculator injection, Python subprocess attempts, filesystem traversal, deterministic serialization, provenance, timeout behavior, output truncation policy, hostile tool-result instruction text, and phase/reference integrity.

Local source-equivalent validation: `15 passed`; Python compileall: PASS. Exact-head repository CI uses the universal `runtime,tests,lint` bootstrap.

## Research truth boundary

This work supplies infrastructure for later reasoning/tool-use research only. It makes no claim that Base has learned tool use, no external LLM calls are made, and no tool observation is silently admitted to pretraining, fine-tuning, checkpoint state, or model weights.

LOCAL_FREE only.
