# Plan 4 / Section 8 — Multi-model serving and orchestration v1

## Authority and scope

tools/plan4_multimodel_serving.py is a serving/routing plane, not an
agent scheduler, policy engine, checkpoint loader, model decoder or tool
permission authority. It consumes the accepted Plan-4 S7 ModelGateway for
all inference/streaming and its exact frozen model identity. S6 remains the
only service load/unload/swap authority.

Client operations use 12-6.plan4-multimodel-serving.v1 and exact
schema/op/args fields. Operations: catalog, generate, stream.start,
stream.next, stream.cancel, stream.close. Host-only operations:
publish, evict, replace; none is exposed over dispatch.

## Resource/role admission

- Trusted host declares independent endpoint slot ID, role, provider, model,
  transport, priority, reserved bytes and capabilities.
- Total declared reservations must not exceed the explicit serving budget.
  Reservations are host-supplied estimates, not measured RAM/VRAM.
  Physical load/unload belongs to underlying S6 service. Evict unpublishes
  a route; it must not be represented as freeing memory without S6 receipt.
- Roles isolate cognitive usage. Deterministic selection is highest
  priority then lexicographic slot ID.
- External admission requires trusted host authorization. A client request
  cannot change routes, budgets, roles or privileges.
- Streams pin original S7 gateway, role, slot, immutable model identity and
  request ID; active streams block route eviction/replacement.

## Failover and replacement

- Failover is not implicit: every request supplies an explicit boolean,
  and trusted host must independently permit failover for that role.
- Retry only backend failure/rejection/invalid backend/identity drift.
  Permission, malformed client and capacity errors cannot force rerouting.
  Results keep the ordered route identity and typed attempt outcomes.
- Replace validates successor identity and capacity before atomically
  retiring an idle incumbent; failure preserves old published slot.
  Successor must have a new slot ID and new explicitly pinned identity.
- After process restart the host must re-admit S6-ready adapters; no old
  stream cursor is silently resumed.
- Every request result carries provenance for downstream durable storage.
  In-memory retired catalog is only process-local diagnostic history,
  not a durable journal.

## Qualification scope

Deterministic fixture tests against the accepted S7 gateway cover multiple
models/roles, resource admission, failover, identity drift, stream cancellation,
eviction/reload, adversarial malformed requests, capacity/permission failures
and transactional rollback. LOCAL_FREE only: no paid compute, external model
calls, champion, agent scheduler, or whole-product Nika claim.

Component terminal DONE needs available tests, integration and exact readback;
queued hosted CI is never described as PASS.
