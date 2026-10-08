# Plan 5 / Section 2 — Persistent task state

## Owned durable reference contract

- Entry point: `twelve_six_agent_runtime.task_state.TaskStore` (independent agent-facing reference package, not model weights or Nika-specific behavior).
- Versioned state JSON `12-6.agent-task-state.v1`; SQLite `user_version=1`; unknown storage/snapshot versions fail closed.
- Immutable `task_id` / `plan_id`; durable `step_id`, `checkpoint_id`, monotonically increasing revision, and `control_epoch`; bounded explicit `PendingEffect(effect_id,description,status,receipt_id)`.
- Every transition is a single `BEGIN IMMEDIATE` SQLite transaction with `synchronous=FULL` and rollback-journal durability; current snapshot and append-only revision history commit together; SHA-256 checksums and history/current parity detect accidental storage corruption.
- `checkpoint()` is compare-and-swap on **both** expected epoch and revision. Invalid identifiers/versions/duplicates, malformed effects, stale writer, stale async epoch and same checkpoint ID are rejected before publication.
- `resume()` is a durable lease change: increments control epoch and revision without changing task/plan/step/checkpoint/pending-effects. A delayed response from an earlier epoch cannot mutate the resumed task even when its payload is otherwise valid.
- Side effects: `issue_effect` durably changes `pending` -> `unknown` **before** external execution; after restart `unknown` cannot be reissued blindly. Only a verifiable external `receipt_id` can drive `unknown` -> `resolved`. The store itself never performs a tool or model call; it neither guarantees external exactly-once execution nor retries ambiguous effects.
- `load_revision()` recovers historical signed-by-checksum state content without rolling back the current authority. Checksums are corruption indicators, not signatures or OS security boundaries.
- Restart evidence: close and reopen the SQLite store in a new `TaskStore`, and copy the quiescent SQLite file to a new path to simulate a cold machine move; the exact task, plan, step, pending effects, revision lineage and checkpoint ID survive. No model identity, paid compute, network or external weights required.

## Existing-work audit / integration

The unmerged historical tool-executor PR #155 defines tool dispatch, verification and execution traces, **not** canonical persistent task storage. Section 2 extends the already integrated independent Plan-5 package from Section 1; it does not create another executor or mutate any of Plans 1–4, 6–10 or the Plan-8 source-inventory baseline.

## Qualification

`pytest -q tests/test_agent_task_state_section2.py` covers process/machine/model-neutral restart, history replay, stale epochs, stale revisions, concurrent CAS writers, issued-effects unknown-on-restart behavior, external receipt reconciliation, duplicate IDs, rollback, corruption and unsupported storage versions.

GitHub Ruff and full repository CI are separate checks; the existing Plan-8 eight-test baseline failure must not be reported as green or silently repaired inside Plan 5.
