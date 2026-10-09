# Plan 10 / Section 2 — staged repository integration evidence (NOT terminal)

This is **not** a whole-product restore qualification or production release candidate.
The live multi-plan registry continues to put Plan 10 Section 1 at
`WAITING_UPSTREAM`. The required production champion and exact integrated
release candidate are absent; in particular Plan 9 Section 1 lacks a released
training corpus, real champion, and authorized training receipts.

## Scoped change (Plan 10 only)

`src/twelve_six/integration/plan10_release_state_binding.py` is an offline
composition layer for seven explicitly provided component-root directories:
configs, manifests, registries, model artifacts, checkpoints, tasks and memory.
It neither replaces Plan 3 checkpoint serialization nor Plan 5 task/memory
databases. SQLite v1 sources are copied with the SQLite backup API, with
component table-shape and integrity checks. Files are sealed under a canonical
SHA-256 manifest whose digest must be pinned by an *independent caller*.
Output publication occurs only after complete verification into a new root.

This is LOCAL_FREE engineering material, **not an authenticated production
signature**. The caller's `writers_stopped=True` promise is not verified by
the tool; it MUST be backed by actual cross-component quiescence in the release
orchestrator. Schema-version changes fail closed; they are **not** silently
migrated. The API intentionally refuses overwrite of existing destinations
and thus is **not** a production in-place update/rollback facility.

## Qualification performed on staged Git-byte-identical files

The local test suite exercises 13 cases: positive seven-component roundtrip,
SQLite version and kind guards, stop-the-writers gate, missing components,
tamper, forged manifest, partial restore, source/snapshot symlinks,
destination no-clobber, duplicate publication, and invalid digest. Test
result: 13 passed / 0 failed; local `compileall` passed. Git blob SHA
source: `0db99c8339ad62acab0148828287d4be7511d112`;
test: `9e8fe2ba1adde48d0bf04c0d88252ef377515ed6`.
No full-repository pytest, CI, Ruff, Nika, production champion, production
checkpoint restore, or physical Windows/cloud qualifications are claimed.

## Required before Section 2 terminal DONE

1. Produce the real exact Plan-9/Plan-10 champion/release candidate and valid
   seven-category binding inventory, not synthetic category fixtures.
2. Bind the composition protocol to real component quiescence and canonical
   task/memory semantic validators; cover actual checkpoints/model weights
   and immutable source identities.
3. Validate real end-to-end backup/cold-restart/restore/migration/rollback,
   corruption and partial-restore tests on one frozen exact release SHA, and
   prove no loss of task/memory continuity.
4. Execute applicable qualification, repair any gating failures, integrate
   into accepted `main`, reread exact merged SHA and verified artifacts.
5. Only then update `MULTI_PLAN_CLOSURE_STATE.md` and Drive Plan 10 Section 2
   to `DONE`; later Sections remain unpromoted.

No other Drive plan is modified. No materially paid compute is requested or run.
