# Research Corpus V1 bulk acquisition controller

Issue: #556  
Worker: `DATA-BULK-ACQ-V1`  
Mode: `LOCAL_FREE`

## Why this exists

The live source-authority vector is far below the frozen Research Corpus V1 capacity targets. Qualifying one small page or one small source repository at a time is useful for provenance research, but it is not a credible primary path for closing a multi-megabyte corpus gap.

This controller adds a planning and intake layer between source discovery and corpus materialization. It deliberately does **not** release a corpus and does **not** authorize training.

## Bound starting vector

The plan binds the current NEXT100-063 source-registry convergence head `5356d60c8c8af46d6fc34debfd3cb36731045338` and exact config blob `d5b640b386219290f69d02a7f2e30a338c883009` without modifying that worker's files.

The CLI independently recomputes the Git blob SHA-1 of the bound parent config and cross-checks the frozen targets, converged capacity vector, remaining gap and safe-result boundary. This prevents a stale duplicated planner vector from silently surviving parent-branch evolution.

Frozen source-capacity targets:

- Ukrainian: 9,000,000 bytes
- English: 7,000,000 bytes
- code: 4,000,000 bytes
- total: 20,000,000 bytes

Converged pre-successor-dedup vector consumed by this planner:

- Ukrainian: 100,856 bytes
- English: 150,643 bytes
- code: 69,133 bytes
- total: 320,632 bytes

Exact remaining planning gap:

- Ukrainian: 8,899,144 bytes
- English: 6,849,357 bytes
- code: 3,930,867 bytes
- total: 19,679,368 bytes

These values are source-capacity bytes. They are not token counts, loss positions, post-dedup corpus capacity, or training authorization.

## Buffered acquisition strategy

The versioned plan uses a `0.60` planning survival floor. This is intentionally marked as **planning-only**, not measured evidence. Its purpose is to avoid planning exactly to the raw gap and then predictably falling below target after rights exclusions, quality filtering, privacy filtering, duplicate removal, and other corpus gates.

The resulting minimum gross acquisition plan is:

- Ukrainian: 14,831,907 bytes
- English: 11,415,595 bytes
- code: 6,551,445 bytes
- total: 32,798,947 bytes

The current work-package allocation is 33,200,000 gross bytes, giving 401,053 bytes of planning headroom above that buffered minimum.

No prospective byte receives source-capacity credit. Only a terminal source authority that passes the exact handoff contract can receive credit.

## Work-package rules

A work package is a source-acquisition pool, not an authority. Each pool declares:

- one stratum: `uk`, `en`, or `code`;
- planned gross acquisition bytes;
- a minimum independent-family budget;
- rights and provenance review state;
- evaluation isolation;
- controls required before terminal admission.

The planner applies a per-stratum package concentration cap so a nominally large acquisition does not silently turn a stratum into one-source monoculture.

Public availability is never accepted as training authority. A candidate can be easy to download and still receive zero capacity credit.

## Terminal authority handoff

A `TERMINAL_ADMIT` package must provide all of the following before its bytes are creditable:

1. exact terminal Git head;
2. terminal-success execution evidence and run id;
3. stable authority identity;
4. explicit training rights decision;
5. provenance review pass and evidence reference;
6. evaluation authorization set to false;
7. exact independent family id;
8. content-addressed object ledger;
9. exact eligible-byte ledger whose sum equals the claimed capacity;
10. capacity-ledger identity.

Missing or inconsistent evidence fails closed.

## Explicit non-claims

Passing this controller means only `PASS_PLANNING_CONTRACT_ONLY` plus `PASS_BASE_AUTHORITY_BINDING`.

It does not claim:

- post-dedup capacity;
- Research Corpus V1 release;
- decontamination closure;
- unique-loss exposure;
- tokenizer readiness or tokenizer fit;
- learned 20M or 100M checkpoint readiness;
- compute authorization.

The next gate is source acquisition plus terminal source authority. Global cross-source dedup, corpus materialization, decontamination, unique-loss accounting, tokenizer fit, and learned 20M training remain downstream gates.

## Validation

Run from repository root:

```bash
python tools/validate_research_corpus_v1_acquisition.py configs/data/research_corpus_v1_acquisition_plan.json
python -m unittest tests.test_research_corpus_v1_acquisition
```

The adversarial suite covers exact arithmetic, deterministic identity, parent-config/blob drift, false prospect capacity credit, malformed nonterminal authority payloads, public-availability promotion, evaluation leakage, paid-compute/training claims, under-buffering, excessive package concentration, fake terminal execution evidence, and premature release claims.
