# Plan 4 / Section 9 — Serving observability, capacity and canary v1

## Scope and invariants

The canonical serving plane remains Plan-4 Section 8 MultiModelServing, which in
turn invokes the accepted Section-7 ModelGateway and Section-6 model service.
`tools/plan4_serving_observability.py` wraps that public serving contract.
It does NOT create model weights, inference, agent scheduler, provider routing,
training/evaluation, physical loader or permission authority.

- ServingObserver.dispatch passes admitted requests to S8 exactly once.
  Ambiguous backend effects are never retried; failures carry S8's unchanged
  provenance and model-attempt receipts.
- Thread-safe bounded in-flight capacity, bounded waiting queue and monotonic
  deadline implement explicit OVERLOADED / QUEUE_TIMEOUT results, not silent
  starvation or unbounded process memory. Rejections do not call S8.
- Metrics expose completed requests, errors by deterministic bounded code,
  cancellations, wall-clock throughput, latency mean/max, queuing and active
  pressure. No prompt, completion, secret, or raw backend error text is logged.
- Declared S8 reservations are explicitly labeled declared bytes; they are
  NOT measured resident RAM or VRAM. Physical memory is UNKNOWN until a trusted
  host supplies a nonnegative integer byte probe. GPU memory is always UNKNOWN
  without a GPU-specific measurement authority.
- Host-only record_model_load times a caller-owned existing load action,
  including failed actions; this observer cannot itself load/unload a model.
- Host-only canary calls one S8 admitted generation with a fixed one-token
  greedy configuration and failover=false. It does not return generated text.
  It returns readiness/identity metadata with evaluation=false; it is NEVER
  a quality benchmark, full evaluation score, promotion or health guarantee.
  External provider canaries are denied by default absent explicit host consent.
- Telemetry is process-local and resets after a process restart. S8's host
  re-admission, model identity and stream restoration rules remain authoritative.

## Capacity and error taxonomy

Host settings: max_inflight 1..128, max_queue 0..1024,
queue_timeout_s 0..60. No user dispatch can alter these settings.
Overload and expiration error envelopes retain the S8 schema; the bounded
enumeration of known S8 error codes maps unfamiliar backend codes to OTHER.

Memory probe is optional; exceptions, booleans, nonintegers, negatives and
nonfinite values become UNAVAILABLE, never zero or a fabricated metric.
Metrics snapshots read S8 catalog only; telemetry does not mutate provider
identity, readiness, routing, gateway state or physical GPU memory.

## Qualification boundary

Tests: `tests/test_plan4_serving_observability.py` plus incumbent
`tests/test_plan4_multimodel_serving.py`,
`tests/test_plan4_model_gateway.py`, and
`tests/test_plan4_model_service.py`.
Covers capacity, queue timeout, cancellation, no retry, exact model identity,
canary isolation and external denial, memory-probe failure, restarts and
invalid host policy. Hosted workflow is not PASS while queued.

LOCAL_FREE fixture component only. No paid GPU inference, learned champion,
physical real-GPU capacity audit, benchmark/final eval or whole-product
qualification claim. Terminal DONE requires available tests, integration,
live-main exact Git blob readback and status records.
