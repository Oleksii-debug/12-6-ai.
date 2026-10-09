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


## Narrow repair follow-up — exact Git-byte LOCAL_FREE qualification (2026-10-09)

The originally qualified 13-test source/test blobs listed above are historical,
not the latest repaired candidate. A security/restore review found that checking
SQLite table names and `PRAGMA integrity_check` alone accepted altered schemas,
including changed column sets or missing canonical unique indexes. A second
reproduced defect used an unescaped SQLite URI: a real directory containing
`#` failed a valid snapshot with an unsupported-schema error.

Repairs on this same Plan-10-S2 PR lineage:
- validate canonical Plan-5 v1 task/memory column signatures and unique key sets,
  without changing the Plan-5 store implementation or its ownership;
- construct the read-only SQLite URI using percent-encoded absolute Path.as_uri();
- add negative extra-column and missing-unique-identity cases, plus a literal
  `#` source-path capture/restore roundtrip.

Reconstructed exact Git blob copies, verified byte identity to the PR branch:
source `8fc32c1024d7d7eb2719738b8e3c17ebb2d499e6`;
tests `9ffdcd95b06b6b85b5c44412f1c39a576780ac4c`.
On these exact code/test bytes, LOCAL_FREE
`PYTHONPATH=src python -m pytest -q tests/test_plan10_section2_state_binding.py`
**17 passed / 0 failed**, with AST syntax parsing PASS. This is a focused Linux
sandbox run, not a full-repository, Windows, hosted-CI, or integrated-release PASS.

No real champion, whole-product candidate, cross-component quiescence, semantic
cold-restart, versioned migration, physical HIL/NVDA or release approval is
inferred. Section 1 WAITING_UPSTREAM; Section 2 remains NOT DONE.


## 2026-10-09 Plan-10-only semantic state audit and live CI readback

**Staging only; Section 2 is NOT terminal DONE.** Audit found that SQLite table
shapes and `integrity_check` alone could accept a malformed task revision ledger
or an invalid memory payload. The existing Plan-5 authorities are now reused at
the release composition boundary rather than copied or replaced:

- `twelve_six_agent_runtime.task_state._decode` validates task digest, canonical
  encoding, task/effect contract and every history entry; composition additionally
  checks contiguous revisions, parent task/plan identity, epoch monotonicity,
  current/latest equality and rejects orphan history rows;
- `twelve_six_agent_runtime.memory.MemoryStore._rows` validates memory record
  digests, revision/sequence, schema, and correction lineage before snapshot
  *and* before restore publication;
- newly added tests cover real Plan-5 task/memory cold restart, forged orphan
  history, tampered history digest, stale memory revision, memory digest
  corruption, and a malicious snapshot whose member and manifest digests were
  recomputed after semantic tampering.

**Exact Git-byte local mirror qualification:** source
`db63e09228625dfb7e2a0b4f52d5fd2aa0bfef46`, tests
`0ec9a3ada759cc41495f933166a90a022c324fde`, matched with
`git hash-object`. `PYTHONPATH=src python -m pytest -q
tests/test_plan10_section2_state_binding.py` yielded **22 passed / 0 failed**.
`python -m compileall` passed. This is scoped LOCAL_FREE testing with the
reused Plan-5 source code, **not** exact integrated product / whole-repo CI.

Independent live GitHub Actions readback: run
https://github.com/Oleksii-debug/12-6-ai./actions/runs/37872588159 on
**previous** PR head `983a065080419872acf02838b6f4fa032d1309f4`
finished **FAILURE**, bootstrap job `113633800276` failed shared
`ruff check src tests` (77 issues in other plans' files); full shared pytest
and SIL were skipped. That run is **not** qualification of this repaired
candidate; do not present it as PASS or as a Plan-10 scoped Ruff failure.
New head requires its own CI readback.

**Remaining mandatory release gates (not simulated):** actual stable champion,
exact approved whole-product release candidate, verified cross-component writer
quiescence, physical model/checkpoint integrity and cold restart, versioned
migration including rollback of entire canonical state, actual Nika/Live Agent
task/memory continuity and same-candidate integration qualification. Upstream
Plan-9 champion remains unavailable. No product release signature, Windows
acceptance, paid compute, or terminal Plan-10 status is claimed.
