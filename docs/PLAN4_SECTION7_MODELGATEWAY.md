# Plan 4 Section 7 — ModelGateway v1 (component-level contract)

## Canonical authority and scope

`tools/plan4_model_gateway.py` is the single Plan-4 cognitive-core routing contract
for Nika, agents, teaching components and tools. It wraps the already integrated,
versioned Plan-4 Section-6 `ModelService`/local and authenticated loopback HTTP
clients. It never implements a second decoder, loads serialized weights, authorizes
a tool call or decides which provider should be used. Only trusted host code can
register an immutable provider/model route or explicitly allow an external adapter.

All client requests use `12-6.plan4-model-gateway.v1` with exact `schema`,
`op` and `args` fields. Supported operations are `describe`, `generate`,
`stream.start`, `stream.next`, `stream.cancel`, `stream.close`.
Provider/model must be supplied explicitly for descriptions and new requests;
a stream uses an opaque, exact gateway session identity pinned to its route.

## Semantics and replacement

Registration pins the complete observed Section-6 model identity: alias, epoch,
ModelSpec hash, immutable weights SHA-256, tokenizer SHA-256 and concrete backend.
Every operation rechecks this identity before and after delegating the call.
Transport type is explicitly `local`, `server` or `external`; an external
adapter requires separate trusted host authorization and must implement the
exact versioned service protocol. Host authorization does not imply permission
to spend money, call external providers or invoke tools.

Successful responses return provider, model, transport, exact identity,
declared capabilities, usage when available, and unmodified underlying result.
No hidden fallback, automatic provider switch, model identity substitution
or tool permission amplification is possible through the client protocol.
A replacement brain is a *new explicit route*, not a silent mutation of
another registered route.

Unknown/malformed operations, unsupported capabilities, omitted provider/model
identity, stale epoch/restarted session, model drift, foreign transport and
failed backends are typed, non-success results. Remote failures are not
reinterpreted as zero usage or successful inference. A backend error is
propagated as a typed rejection with sanitized details; unexpected failures
do not expose credentials or backend internals.

## Qualification and limits

Dedicated checked-in tests cover actual tiny Torch reference generation,
exact usage/model identity, alternate provider routing, host-only registration,
read-only feature discovery, cancellation and idempotent stream event replay,
close, capacity denial, model drift, service restart epoch fencing, malformed
requests and a real authenticated loopback HTTP adapter. The gateway does not
replace Plan-5 agent/task authority or Plan-10 Nika product integration.

This is a LOCAL_FREE fixture-level component contract, not proof of a production
external provider, paid inference entitlement, champion accuracy, browser/tool
authorization, internet-facing hardened service, or complete product readiness.
