# Plan 5 / Section 6 — Reference scheduler contract (v1)

This is a 12-6-owned, provider-neutral reference scheduler, **not** Nika Core, not a tool executor, and not an authorization to run paid compute or external effects. The calling trusted host authenticates WorkSpec and resource readings; model outputs and retrieved text cannot confer scheduling authority.

## Admission and ordering

- Every registered task has stable task/plan/goal IDs, priority 0–100, monotonic deadline tick, positive CPU and memory envelopes, a bounded step budget, and foreground/background classification.
- Admission fails closed for invalid types, zero budgets/envelopes, duplicates or foreign/preexisting task state. A TaskStore initial snapshot created before a crash is adopted only after exact identity/plan/initial-state checks.
- Ready, in-deadline tasks with sufficient resources are selected by descending priority, earliest deadline, then task ID. A single active lease is allowed. Background work throttles at 500,000 ppm resource pressure; all dispatch pauses at 900,000 ppm. Deadlines cannot be overridden by a self-model.
- The caller must not treat a returned Lease as permission to execute tools. The separate trusted authorization boundary still governs all real effects.

## State and recovery

- `twelve_six_agent_runtime.task_state.TaskStore` (Plan 5 Section 2) remains the only authority for control epoch, task/step/checkpoint identity, and pending/unknown/resolved effect reconciliation. The scheduler never creates a second task/effect ledger.
- Scheduler metadata is independent `12-6.agent-scheduler.v1` SQLite FULL-sync versioned queue plus history, with canonical JSON and per-record SHA-256. Queue mutations use immediate transactions and lease revision tokens.
- A lease calls `TaskStore.resume` and records its exact resulting control epoch and revision. Every checkpoint and terminal completion rechecks **both** the queue lease and canonical TaskStore snapshot. Stale or conflicting tokens fail closed.
- Checkpoint saves the exact TaskStore step/checkpoint/effect tuple and counts spent steps. It pauses under severe pressure or exhausted budget; exhausted budget cannot be reset by recovery.
- If a crash occurs between the two distinct SQLite commits, the projection may be temporarily behind TaskStore. The implementation intentionally does not claim two-database atomicity. Uncertain active leases are not automatically retried; `reconcile` requires a trusted caller verifier and verified resolved-effect receipts before a new control epoch is issued.
- Unknown/pending effect states are ineligible for new scheduling even if a high priority job is waiting. External receipt reconciliation uses incumbent TaskStore logic. The verifier is supplied by the trusted embedding host, not a model-generated claim.
- This component does not promise concurrent distributed scheduling, paid compute, Nika UI integration, actual provider calls, OS resource reservation or whole-product readiness. Those are outside Section 6.

## Local-free acceptance

`PYTHONPATH=src python -m pytest -q tests/test_agent_scheduler_section6.py` qualifies deterministic scheduling, deadlines/resources/budgets, foreground/background throttling, crash/restart, stale epochs, uncertain external effects, receipt reconciliation, duplicate identity and concurrency, invalid admission, store corruption and version refusal. Exact TaskStore regression is also exercised by the scoped GitHub workflow.
