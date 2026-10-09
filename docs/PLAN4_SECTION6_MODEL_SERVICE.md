# 12-6 AI — Plan 4 / Section 6: stable local and loopback model service v1

## One model/service authority

`tools/plan4_model_service.py` is the **single** Plan4 Section6 service-dispatch adapter over the already accepted `tools/inference_runtime.py` `ReferenceInference` and `TwelveSixDecoder.generate`. It does not fork model logic, deserialize weight files, bypass checkpoint/HF exporter verification, import vendor-specific backends, promote quantized candidates, or authorize paid compute. Only trusted in-process host code may register a previously validated `ReferenceInference` under an immutable alias. Client calls never contain model file paths, execution commands, Python objects, secrets or serialized weights.

## Public JSON protocol

Every call takes `{"schema":"12-6.plan4-model-service.v1","op":"...","args":{...}}` and returns `{"schema":...,"ok":true,"result":{...}}` or a typed, sanitized `{"schema":...,"ok":false,"error":{"code":...,"message":...}}`. Unknown/extra/missing keys, invalid schema, unknown model alias and unsupported operations fail closed. Canonical `GenerationConfig` validates all generation settings.

Supported operations:

- `health {}`: explicit `UNLOADED`, `READY` or `FAULTED`, readiness Boolean, model alias, epoch, session counts and public backend capability.
- `identity {}`: canonical ModelSpec hash, physical model-weight SHA, tokenizer identity, epoch, backend; only while ready.
- `load {"model":"alias"}`, `unload {}`, `swap {"model":"alias"}`: trusted pre-registered adapters only. Switching/unloading with unfinished streams is denied. A drifted model may not silently swap to an alternative. Changes increment epoch and expire old sessions. `load` requires `UNLOADED`, `swap` requires `READY`.
- `restart {}`: explicit local process-restart analog, invalidating all session IDs and returning to unready `UNLOADED`; clients must explicitly load again. No process-persistence or fake resumed-token claim.
- `generate {"prompt":"text","config":{"max_new_tokens":2}}`: synchronous canonical reference decode with exact model/prompt/result SHA identities.
- `stream.start` with the same `prompt` and `config`: creates bounded opaque session token and initial cursor zero.
- `stream.next {"session_id":"...","cursor":0}`: one canonical `TOKEN` or terminal `COMPLETED`/`CANCELLED` event. Cursor (n) advances to (n+1); repeating the last cursor idempotently returns exactly the last event; stale/future requests fail closed. Single-use reference iterator is not silently restarted.
- `stream.cancel {"session_id":"..."}`: request cancellation, then drain the terminal `CANCELLED` event through `stream.next`; never label it `COMPLETED`.
- `stream.close {"session_id":"..."}`: release a completed/cancelled session. Bounded session slots prevent unbounded accumulation.

Both `LocalClient.call` and `HttpClient.call` use the **same canonical JSON normalization and dispatch**. In-process tuple fields are normalized into JSON arrays before either transport observes them. The HTTP adapter binds strictly to `127.0.0.1`, exposes only `POST /v1/dispatch`, requires a caller-provided unpredictable bearer token, rejects unauthenticated, oversized, duplicate-key, nonfinite or malformed JSON messages and does not log prompts or credentials. The caller explicitly controls the HTTP server lifecycle; this is a LOCAL_FREE reference service, **not an internet-facing production deployment**.

## Failure / recovery / compatibility

All requests run under one lock to preserve ordering, state transitions, and idempotent event replay. Physical reference weights and tokenizer identity are rechecked on admission. Drift transitions to `FAULTED`, invalidates sessions and rejects new generations until an explicit `unload` and trusted fixed-model reload (or `restart`). No silent fallback to another model/backend. Errors are typed and sanitized; unexpected runtime failures close the service state rather than exposing internals. Session IDs are per-epoch cryptographic random identifiers. The protocol is transport neutral: future optimized adapters may only enter after their own explicit backend qualification; clients do not parse SafeTensors, GGUF, model paths or backend tensor formats.

## Local qualification and limits

Scoped automated fixtures exercise lifecycle, deterministic generation and sample/cancel semantics inherited from the canonical inference adapter; session replay, close, stale cursor, active stream pressure, restart, swap, tampered model, unsupported commands/configs, concurrent calls, actual loopback HTTP parity, authentication/JSON negatives, and full readiness reset. Hosted CI is independent and may remain queued; never call it green without a completed exact-head result. The LOCAL_FREE protocol proves a reference service with tiny random-init model, **not** real trained model quality, external provider production readiness, cross-process persistence, unrestricted network serving, real GPU costs or whole-product launch qualification.