# Plan 2 / Section 15 — terminal data qualification contract

Status: **OPEN — component audit implemented but real dataset release NOT authorized**.
Plan authority: Google Drive "2. Другий план", Sections 15.1–15.3, and
`MULTI_PLAN_CLOSURE_STATE.md` (GitHub overrides the Drive migration snapshot).

## Reused implementation

The single candidate lineage is the existing Plan-2 S1–S14 source registry,
rights catalog, physical materializer, normalization, G06 privacy, exact
and complete-link near dedup, DATA-232 reserved evaluation firewall, corpus
mixture, cluster-safe split, fitted frozen tokenizer fixture, packer and
ordered exposure ledger. No second ingestion, tokenizer, trainer, dedup,
split, checkpoint or permission authority is created.

`tools/plan2_terminal_qualification_v1.py` runs two independently
materialized LOCAL_FREE builds in fresh directories and demands byte-identical
inventories. It re-verifies the S5 privacy -> S6 exact -> S7 near ->
S8 eval firewall -> S9 mixture chain using the canonical staged manifests
and SHA-256 content identities. It refuses noncanonical, missing, changed,
or symlinked outputs, source-count drift, unexpected split failures,
rights escalation, leaked raw text / final-test access, synthetic-as-production
tokenizer/shard/exposure promotion, incompatible manifests, and immutable
readback tampering. It emits only hash/metadata evidence (not training text).

Current source is the pinned DATA324 Ukrainian Kubernetes documentation,
a single document family. Its Plan-2 rights grant admits
`allowed_uses=["source_candidate"]`, not training or release.
Physical three-way family-safe train/validation/test split correctly refuses
the one-family candidate. S12 tokenizer and S13/S14 packing/exposure results
are expressly **synthetic component fixtures**, not production artifacts.

Other D03 evidence on main may contain additional candidate families,
but is not a lawful Plan-2 release. Candidate source admission, partial
balance, or independent component success is not final training authority.

## Real public-domain source expansion (2026-10-08)

This branch now physically carries **1,265,481 UTF-8 bytes** from three
original public-domain books (Austen, Carroll, Shelley), each pinned to its
exact GITenberg source Git blob and commit in
`configs/data/plan2_public_domain_books_v1.json`. Project Gutenberg
trademark and license envelopes were stripped before storing the bodies.

**One source family, three document clusters:** These three works belong to
the ONE existing canonical
`en.project-gutenberg.public-domain-books` source family under the
accepted NEXT100-107 seal. Individual authors/works have different
`document_family` identities and upstream revisions. They are NOT three
independent source-family capacity credits.

`tools/plan2_public_domain_books_v1.py` reuses the canonical S1 inventory,
S2 source rights/materialization receipts, incumbent G06 privacy authority,
and DATA-232 contamination matching against the pinned synthetic S8 reserve.
It refuses source/rights/snapshot drift, unauthorized training release,
source symlinks, inconsistent source-family assignments, broken privacy
receipt roots, contamination report tampering and changed publications.
The two clean S15 builds reverify the full real-book snapshots and hash-only
evidence. New tests and steps are in
`tests/test_plan2_public_domain_books_v1.py` and the existing Plan-2 CI.

**The new books are NOT a terminal dataset or admitted P9 training input.**
Source-level legal eligibility and G06/DATA232 candidate audits are not
real reserved final-test custody, complete S3–S9 multi-source admission,
balanced train/validation/test splits, frozen physical tokenizer,
production packed shards, ordered exposures or Plan9 launch approval.
The older physical S3/S9 cohort remains one separately pinned Ukrainian
Kubernetes source with allowed use `source_candidate` only. No source
family was invented or silently promoted into that authority.

## Qualification commands

From a complete checkout of the exact PR head with project requirements:

```sh
PYTHONPATH=src:. pytest -q tests/test_plan2_terminal_qualification_v1.py
ruff check tools/plan2_terminal_qualification_v1.py tests/test_plan2_terminal_qualification_v1.py
PYTHONPATH=src:. python tools/plan2_terminal_qualification_v1.py \
  --root . --out-dir /tmp/plan2-s15-audit
PYTHONPATH=src:. python tools/plan2_terminal_qualification_v1.py \
  --root . --out-dir /tmp/plan2-s15-audit
```

The same-directory repeat must preserve all bytes and hash identities.
Run once more with a fresh output directory and compare reports.
The GitHub "Plan 2 Data Component Qualification" workflow includes prior
S1–S14 scoped gates and the S15 audit. A queued or cancelled run is **NOT PASS**.

## Exact terminal acceptance — no promotion by assertion

Only mark Plan-2 Section 15 DONE after all are *actually* true:

1. An accepted multi-family corpus has independently verified source identity,
   explicit training/release rights, member provenance and removal path.
2. Physical rights, privacy, global dedup, evaluation decontamination,
   mixture and family-safe split pass on the **same exact** candidate, without
   seeding test/validation records into tokenizer fit or model training.
3. The tokenizer is fitted/frozen on this accepted train partition, with
   immutable vocab, IDs, normalization and compatibility proof (not S12 fixture).
4. Physical deterministic shards, source/target mapping and stable ordered
   exposures exist; exact identities match the frozen corpus and tokenizer.
5. At least two clean rebuilds produce identical physical dataset/tokenizer/
   shard/exposure hashes; negative and restart/recovery tests qualify the
   release, not merely a staging fixture.
6. The release manifest provides immutable, read-back-verified dataset,
   tokenizer, shard and exposure identities for Plan 9. This is a **data
   handoff only**; Plan 9 must still independently authorize any pretraining.
7. A canonical integrated SHA and evidence are read back from GitHub, then
   both GitHub closure state and Drive plan are updated to terminal DONE.

Until then, the report must keep
`decision=COMPONENT_AUDIT_ONLY_NOT_TERMINAL`,
`terminal_done=false` and `production_release_authorized=false`.
No model training, final-test reading, paid GPU run, foreign pretrained weights,
source-rights promotion or PLAN-9 authority is conveyed.

There is no Section 16 in the second plan; after terminal Section 15 the
numbered plan closes. Any next-plan assignment must be explicit.
