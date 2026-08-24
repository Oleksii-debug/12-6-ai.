# S0 late-wave intake registry — 2026-08-25

This document is the D01 convergence view of the burst that followed exact-green PR #89. It exists because the current risk is no longer lack of S0 Product code; it is accidental composition of queued/red heads or duplicate workers touching the same surface.

## Exact trusted base

Current terminal-green base is PR #89, exact SHA `c631c024e641dac102036fafee6d78ba31c067cd`.

Its exact-source CI, real S0 training, fresh-run repeatability/seed-causality workflow, and strict exact-candidate evaluation are all completed SUCCESS. The same-seed proof is limited to its declared locked Linux x86-64 CPU contract. It does not grant audit or promotion authority.

## Highest-priority S0 intake sequence

The next D01 composition should not be opened until these three surfaces are terminal exact-green:

1. PR #90 — D01 r2 intake of exact-green PR #89 repeatability evidence.
2. PR #91 — D05/D07 single immutable checkpoint snapshot for first-party inference, closing the real verify-then-reopen TOCTOU lineage split.
3. PR #100 — D04 binding of strict S0 evaluation evidence to PR #89 repeatability evidence for audit consumption.

At the snapshot cutoff all three still have at least one queued exact-head workflow. Therefore none is accepted by this registry.

## Duplicate/collision decisions

The repeatability-intake collision is resolved in favor of incumbent PR #90. The later D01 PR #97 was closed unmerged after late-wave discovery of #90.

The checkpoint-to-first-party-inference snapshot collision is resolved in favor of incumbent PR #91. PRs #94, #98, and #102 are duplicate/superseded implementation attempts and must not be wholesale mixed with #91.

HF-style export has competing D05 implementations (#95 and #111). This is not an S0 promotion-critical surface. Select at most one after terminal exact-head CI and semantic comparison; do not compose both implementations of `hf_export.py`.

Several D05/D07 branches (#93, #108, #112, #116, #119) address retained/replayable/trained inference evidence. These may use different file paths but overlap in purpose. D01 should select the smallest green set that adds distinct evidence rather than multiplying near-identical 40-step workflows.

## Useful but nonblocking P1/P2 work

PR #92 strengthens locked-runtime evaluation evidence. PR #101 works around the trailing-period repository name for Windows CLI transport but explicitly does not prove canonical Torch checkpoint loading or live NVDA. PRs #117/#118/#120 harden generation numerics, precision-runtime fail-closed behavior, and local server privacy/timeout behavior respectively. They are useful correctness work, but they should not postpone exact S0 audit handoff if the core candidate is otherwise ready.

PR #106 is an S1 engineering numerical preflight. It is explicitly excluded from S0 composition.

## Governance remains separate

PR #113 is D10-owned release-authority composition on top of current Product evidence. D01 must not absorb or rewrite it. Live `main` remains bootstrap-only and unprotected. Vulnerability/license adjudication and trusted release-root controls are still independent blockers.

AUDIT-A #13 and AUDIT-B #14 retain their last actual `CHANGES_REQUIRED` verdict until each independently retests one final exact candidate SHA. Developer workflow success does not upgrade those verdicts.

## Fail-closed process

The machine snapshot at `evidence/swarm_exp_01/d01_late_wave_intake_snapshot_20260825.json` is validated by `src/twelve_six/integration/late_wave_intake.py`.

The validator enforces:

- PR #89 exact-green workflow evidence as the registry base;
- random-init/pretraining-only Base and no paid/foreign/alignment claims;
- historical audit authority remains unchanged;
- one incumbent per explicit collision group;
- S1 cannot enter S0 composition;
- D10 governance remains separately owned;
- the minimum next S0 composition set is #90/#91/#100;
- queued/red heads cannot be treated as accepted;
- post-composition exact-head workflows must rerun;
- the final exact head must be handed to both independent auditors.

Run:

```text
python tools/validate_late_wave_intake.py evidence/swarm_exp_01/d01_late_wave_intake_snapshot_20260825.json
pytest -q tests/test_late_wave_intake.py
```

This snapshot is deliberately point-in-time. When a required PR becomes terminal green or red, update the snapshot in a fresh commit rather than reinterpreting stale recorded state.
