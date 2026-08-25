# S0 late-wave intake registry — W1 materialized composition 2026-08-25

This document records the W1 convergence state after the late S0 PR forest was pruned and the minimum exact-green successor set was materially composed. It is not promotion authority.

## Exact trusted base

The broadly proven common base remains PR #89 at exact SHA `c631c024e641dac102036fafee6d78ba31c067cd`.

Its exact-source CI, real 40-step S0 CPU training, fresh-process repeatability/seed-causality workflow and strict exact-candidate evaluation are terminal SUCCESS. Base semantics remain random-init and pretraining-only.

## Materialized W1 composition

The former minimum composition set was rechecked immediately before mutation and remained open, mergeable, terminal-green and rooted at exact #89:

1. PR #90 `df13fbc5c218ff42b00b749384e6b02a1bc775c9` — D01 repeatability convergence intake.
2. PR #91 `ee620ff2f25fcba7537c41ef1322124ada82b02c` — D05/D07 single verified checkpoint snapshot.
3. PR #100 `f6e95128c885fa176b9eb7f5f71abd154e0b30b7` — D04 evaluation/repeatability binding.

Their changed-path sets are 2 / 3 / 3 files and have zero mutual overlap. They were composed on the existing W1 incumbent branch, not through a new wrapper.

Multi-parent composition commit:

`425138d7cdad6b2b2c7600c63fcb34dee246f7fb`

Parent order is intentional:

1. terminal-green W1 registry head `d6c054f7ecbc3161b7659ab30fe1b4130089b709`;
2. PR #90 exact head;
3. PR #91 exact head;
4. PR #100 exact head.

The composition tree was built from the terminal-green #132 tree plus the exact source blob identities from the eight disjoint source paths. Source history was not flattened or destroyed.

The machine registry now records #90/#91/#100 as integrated terminal-green sources rather than leaving them classified `COMPOSABLE` or required for another composition. `minimum_required` is therefore empty.

## Exact-head validation boundary

Materializing the parents does not inherit PASS onto a newer exact head.

After the composition branch moves, the resulting exact head must obtain fresh terminal results for:

- broad CI;
- D02 real S0 CPU training;
- D02 fresh-process determinism/repeatability;
- D04 strict exact-candidate evaluation.

Queued or in-progress runs are not PASS. The registry deliberately encodes `MATERIALIZED_REQUIRES_EXACT_HEAD_RERUN`; it does not try to self-reference the final containing commit or freeze transient workflow state.

Only after a final exact head is terminal-green should that exact SHA be handed to both independent auditors. Audit handoff is not an audit verdict.

## Remaining intake map

Optional green semantic-review work remains separate: #92 and #134. Neither is silently added to the minimum composition.

Held incumbents remain held:

- #95 transactional HF export: incumbent + RED + HOLD until its owner repairs/revalidates it;
- #113 D10 release authority: RED + HOLD and separately owned;
- #106 S1 numerical preflight: HOLD for S0.

Closed-unmerged duplicate history remains:

`94, 97, 98, 102, 105, 111, 136`.

The collision owners remain #90, #91, #95 and #134 for their respective same-surface groups.

## Validator v3

`evidence/swarm_exp_01/d01_late_wave_intake_snapshot_20260825.json` uses schema `12-6.s0-late-wave-intake.v3`.

The validator continues to enforce the W1 classification vocabulary:

`INCUMBENT / GREEN / RED / QUEUED / SUPERSEDED / DUPLICATE / COMPOSABLE / HOLD`.

In addition, v3 requires:

- exact integrated source PR/SHA binding;
- globally disjoint integrated changed paths;
- zero-overlap, history-preserving composition metadata;
- integrated sources to remain terminal-green and no longer be marked `COMPOSABLE` or required for another composition;
- exact checkout ancestry from #89, the multi-parent composition commit and all three source heads;
- the exact parent set and parent order of the materialized composition commit;
- fresh post-composition workflow reruns;
- both independent audit handoffs after final exact-head validation;
- promotion, paid compute, foreign pretrained Base weights and behavioral/alignment Base work to remain false.

Run on an exact checkout:

```text
python tools/validate_late_wave_intake.py evidence/swarm_exp_01/d01_late_wave_intake_snapshot_20260825.json
pytest -q tests/test_late_wave_intake.py
```

## Governance boundary

Live `main` remains bootstrap-only at `f2e94c7212888cdb960bb66154d56d210e9b27ab` and unprotected. AUDIT-A #13 and AUDIT-B #14 retain their last actual `CHANGES_REQUIRED` verdict until they independently issue a new verdict on a final exact candidate.

S0 therefore remains **EXPERIMENTAL**. Canonical-main protection, vulnerability/license adjudication and D10 release authority remain separate gates.

## Next W1 action

Do not create another S0 convergence wrapper. Let the refreshed exact composition head finish all required workflows. If any current-head workflow is RED, diagnose and repair only the concrete owner-safe defect on the same incumbent branch. If all required workflows are terminal SUCCESS, bind the exact SHA and workflow IDs into durable handoffs to AUDIT-A and AUDIT-B, then advance W1 to protected-main/release convergence rather than further S0 wrapper proliferation.
