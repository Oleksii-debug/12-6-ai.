# Plan 5 Section 3 — source-aware durable reference memory

This is a repository-owned agent-facing reference contract; it is not Nika Core, a scheduler, a model, an internet fact database, or a grant of external permissions.

## Contract and integration

- `twelve_six_agent_runtime.memory` uses the existing Plan 5 `ContextEntry` and Section 2 `task_state` identity/canonical hashing helpers; it does **not** fork those authorities.
- Typed disjoint `episodic`, `semantic`, `procedural`, `project` records with stable IDs, separately explicit `source_id`, `source_kind` (`owner`, `tool`, `retrieval`, `observation`, `model`) and mandatory `evidence_ref`.
- Strict bounded UTF-8 content, immutable finite confidence [0,1], observation and optional expiry timestamps, `corrects` lineage. Strict exact keys, canonical JSON and SHA-256 row checksum; checksum is corruption detection, **not** a cryptographic identity of the source or authorization of the underlying claim.
- SQLite `user_version=1`, FULL-sync transactions, a crash-consistent append-only record ledger plus host-approved acceptance receipts. No rewrite or deletion of records on correction, no duplicate identities, correction forks, kind changes, future/expired acceptance, or unsupported storage schema. `retrieve` excludes pending, expired, future or superseded entries; casefold query, kind filter and deterministic stable ranking. Model-independent copy/cold restart and new-process replay.
- Acceptance means **eligible for recollection**, never authority about an external fact. `as_context_entries` always marks stored memory as `source_kind="retrieval"`, `critical=False`, independent of whether the recorded source was "owner"; no stored memory becomes system/owner instructions or capability permissions.
- `verify_live_external_fact` requires a fresh host-controlled, separately trusted source adapter that verifies the original claim against the live source. Without an explicit adapter TRUE result, it fails closed. Verification returns an ephemeral `LiveFact`; the SQLite store and future retrieval **never** persist or grant `canonical_external_truth`.
- The trusted host, not untrusted model text, supplies source identity, acceptance receipt, live verifier and execution permissions. This module does not call tools or access secrets. Do not pass a model-provided fake verifier as a trusted host adapter.
- ModelGateway remains a fixture/mock integration boundary. No production model, paid GPU/cloud, Nika state, whole-product release or internet browsing is implied.

## Qualified tests

`PYTHONPATH=src python -m pytest -q tests/test_agent_memory_section3.py tests/test_agent_memory_section3_adversarial.py`

Covered: all four memory classes, provenance and ranking, approval gate, live-source verification failure, expiration, correcting prior evidence, duplicate and fork refusal, untrusted context projection, NaN/infinite/type abuse, storage version and symlink refusal, corruption/checksum/metadata tampering, CAS-like concurrent ID collision, file-copy cold machine migration and independent child-process restart/replay. Separate Section 2 task-state tests are run as integration regression with no production model.

Full global CI remains independently affected by legacy Plan 8 source-map failures; do not report it green. Hosted CI queued is not a PASS. This Section is scoped repository-controllable reference engineering only.
