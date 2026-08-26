# D05/D07 local raw-Base server lifecycle and backpressure

This extends the incumbent raw Base local server from PR #86 and the privacy/socket-timeout
hardening already owned by PR #120. It does not create another HTTP evidence wrapper and does
not change checkpoint bytes, model/tokenizer logic, generation behavior, sampling behavior, or the
default loopback-only security posture.

## Architecture

HTTP transport and model execution are now separate concurrency domains.

`ThreadingHTTPServer` may handle independent local clients concurrently, so one slow socket or
long completion no longer owns the listener. Model execution is deliberately different:
`ServingRuntime` has exactly one execution lane and bounded admission. Current first-party
backend concurrency, mutable KV-session concurrency, and shared model execution safety have
not been proved, so this server does not manufacture parallel model execution by putting the
same PyTorch model behind multiple request threads.

The execution scheduler is a replaceable seam. A later batching backend or maintained serving
runtime can own execution policy without reimplementing request parsing, local security,
readiness, structured errors, or model identity.

## Lifecycle and probes

`/healthz` is listener liveness and retains the historical response shape. It does not imply
that a model is usable.

`/readyz` is model readiness. It returns 200 only in `ready` state and 503 while the runtime is
`loading`, `draining`, or otherwise unavailable.

`/statusz` exposes privacy-safe scheduler state and counters: queue depth/high-water mark,
single-lane activity, accepted/completed/failed/rejected counts, timeout/cancellation counts,
and cumulative model-execution wall time. It never stores request prompts or generated text.

`/v1/models` retains the existing model entry and adds allowlisted runtime/checkpoint metadata
when the backend provides diagnostics. Arbitrary backend diagnostic keys are not reflected.

The canonical CLI still verifies and loads the checkpoint before binding the socket. This keeps
the stronger existing fail-closed startup property. `make_loading_server()` and
`install_backend()` provide an explicit liveness-before-readiness lifecycle for embedding hosts
that intentionally load out of band. Installation is allowed only once from `loading`; live
checkpoint replacement is deliberately not implemented. Today checkpoint replacement requires
process restart rather than an unsafe hot-swap.

## Bounded work and timeouts

`--max-queue-depth` bounds requests waiting behind the one reserved model execution lane.
The default is 8. When capacity is exhausted, a new completion receives HTTP 503 with stable
error code `queue_full`.

`--completion-timeout-seconds` bounds how long an HTTP request waits for an accepted completion.
The default is 120 seconds. If work has not started, timeout cancellation removes it before
model execution. If model execution already started, the caller receives HTTP 504 but the
runtime does not claim to have preempted PyTorch execution. The active call finishes in the
single execution lane and is accounted separately. Cooperative mid-generation cancellation is
a future backend/runtime contract, not simulated here.

The existing `--request-timeout-seconds` remains an independent socket I/O bound. Request body
size remains bounded by `--max-request-bytes`.

## Shutdown

Readiness can be withdrawn by entering `draining`. `server_close()` stops new runtime
admissions, drains already accepted model work, shuts down the execution lane, and then closes
HTTP client threads. No public remote shutdown endpoint is added.

## Structured failures

Server-owned failures now carry stable codes, including `model_not_ready`, `queue_full`,
`completion_timeout`, `internal_error`, request-body/media/JSON failures, unknown endpoints,
model mismatch, and explicit chat rejection. Existing raw completion validation remains
delegated to the canonical completion/generation layer.

## Privacy and local security retained

The prior PR #120 logging fix remains: logs contain only client address, normalized method,
known endpoint category, status/size, and generic protocol/internal error classes. Raw request
paths, query strings, headers, prompts, generated text, and arbitrary exception text are not
emitted by the new observability surfaces.

Binding remains loopback-only unless `--allow-non-loopback` is explicitly supplied. This flag
is not public-serving authorization. There is still no TLS, authentication, tenant isolation,
rate limiting, or public multi-user security design.

## Intentionally not claimed

- no parallel execution of the same first-party model;
- no batching yet; the scheduler is only the integration seam;
- no streaming;
- no client-disconnect cancellation and no force-preemption of active model work;
- no live checkpoint hot-swap;
- no KV-cache throughput claim;
- no vLLM/Transformers serving claim;
- no latency, throughput, capacity, or SLA benchmark claim.

The separate KV-cache work in PR #138 can remain a first-party execution optimization without
being treated as a proof of concurrent-session safety. Retained checkpoint and inference
acceptance work (#112/#119/#126) remains evidence input rather than being duplicated here.

Canonical Base remains random-init and pretraining-only. No foreign pretrained weights,
instruction/chat/alignment/refusal/personality/domain behavior, paid compute, audit verdict,
CANDIDATE, or STABLE claim is introduced.
