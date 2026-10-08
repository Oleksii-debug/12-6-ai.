# Plan 5 / Section 6 — bounded autonomous reference scheduler

This is the 12-6-owned, provider-neutral scheduler contract; it is **not**
Nika Core, a browser automation engine, an executable tool dispatcher, or an
authorization to spend money.

- `TaskStore` remains the sole source of task epochs, snapshots, effects,
  receipts and replay correctness. `SchedulerStore` stores only immutable
  scheduling specs, admission reservations, attempt budgets and lease states.
- A trusted host creates the task in `TaskStore` before `register`.
  Admission is atomic under SQLite `BEGIN IMMEDIATE`, FULL sync, persisted
  counters and a policy fingerprint. Multiple workers cannot claim the same
  ready job or overbook CPU, RAM, global attempts or concurrency.
- Deterministic selection: descending priority, ascending deadline, work
  class, then task ID. Deadlines and capacities fail closed, as does unknown
  or unverified resource evidence. Only a host-verified, recent reading can
  trigger `offer`. High pressure latches a global pause; a separately
  verified normal reading is required to resume admission.
- `DispatchLease` is an **offer**, not a side effect or capability grant.
  Hosts must use the existing TaskStore effect reserve/issue/receipt contract.
  Scheduler cannot execute anything itself. Unknown/pending effects cannot be
  completed or retried. A running lease remains running across restart;
  no blind resubmission is possible. After interrupted execution, a trusted
  external verifier must attest that execution stopped and effects are
  reconciled before `reconcile_interrupted` can release the lease to a
  checkpoint. A checkpoint additionally requires trusted evidence before
  `retry_checkpoint`. Stale lease/epoch/revision is rejected.
- A candidate returning `None` means NO ADMISSION, not success or failure.
  Attempt budgets remain spent even if a run fails or is checkpointed.
  `done` is irreversible by this component. There is no wall-clock sleep,
  job runner, background thread, provider lock-in, secret store, model call,
  network call or training dispatch.

SQLite digests detect accidental corruption; they are *not* digital
signatures or a defense against an attacker who can rewrite the database
and recompute checksums. The host must provide filesystem access controls
and independent verifier authority. The reference's budget is **LOCAL_FREE
simulation/fixture only**; an external worker must separately authorize any
real expense or effects.

Acceptance targets: positive deterministic priority/parallelism; crash and
machine restart; duplicate claims; no effect retry after unknown outcome;
pressure pause/checkpoint/recovery; stale epochs; expired deadlines; corrupt
rows; forged evidence; exhausted attempts; concurrent claims. Production
end-to-end/Nika integration is Plan 10, not evidence for Plan 5 S6.
