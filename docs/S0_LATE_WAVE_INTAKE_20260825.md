# S0 late-wave intake registry — W1 live refresh 2026-08-25

This is the fail-closed convergence view of the late S0 worker forest. Its purpose is to reduce competing branches to one truthful intake path, not to manufacture another integration wrapper.

## Exact trusted base

The retained broadly proven base is PR #89 at exact SHA `c631c024e641dac102036fafee6d78ba31c067cd`.

Its exact-source CI, real 40-step S0 CPU training, fresh-process repeatability/seed-causality workflow and strict exact-candidate evaluation are terminal SUCCESS. That evidence does not grant independent audit or promotion authority.

## Live minimum composition set

The former pending minimum set has now become terminal-green on the same exact heads:

1. PR #90 `df13fbc5c218ff42b00b749384e6b02a1bc775c9` — D01 repeatability intake incumbent. CI `32779205999`, real S0 training `32779206011`, repeatability `32779205956`: SUCCESS.
2. PR #91 `ee620ff2f25fcba7537c41ef1322124ada82b02c` — D05/D07 immutable first-party checkpoint snapshot incumbent. CI `32779800315`, real training `32779800261`, repeatability `32779800335`, strict evaluation `32779800319`: SUCCESS.
3. PR #100 `f6e95128c885fa176b9eb7f5f71abd154e0b30b7` — D04 evaluation/repeatability binding. CI `32779657381`, real training `32779657117`, repeatability `32779657126`: SUCCESS.

The machine registry therefore classifies these exact heads `COMPOSABLE`. This means only that their recorded exact-head workflow evidence is green and they are eligible for selective composition. It does not mean they are merged, audited, promoted or safe to flatten without ancestry/file-overlap checks.

PR #92 `c910b008d628256fc38ecd58a87e43f462660b54` is also terminal-green and remains optional semantic-review intake rather than part of the minimum set.

## Red and held work

PR #95 remains the transactional HF-export incumbent, but current exact head `9d8deeb5368af8a87215c15f9674cf26cc4651b6` is RED: CI `32779993542`, real training `32779993547`, strict evaluation `32779993531` all failed. It is `INCUMBENT + RED + HOLD`, not a reason to create a competing exporter.

PR #113 is D10-owned release/governance composition and remains separate from W1 Product intake. Its current head `5e9723ba7e15706055a578c9f8b9f40000e088c7` has successful security/history evidence but RED Product workflows. It is `RED + HOLD` and must be repaired by its owner or selectively superseded through D10 governance, not absorbed into this registry implementation.

PR #106 is S1 engineering preflight and is `HOLD` for S0. S1 work must not enter S0 composition merely because its mechanics exist.

## Duplicate pruning decisions

The registry records closed-unmerged duplicate/superseded branches instead of deleting useful history:

- repeatability intake: #90 incumbent; #97 duplicate closed unmerged;
- first-party checkpoint snapshot: #91 incumbent; #94/#98/#102 duplicate attempts closed unmerged;
- HF export transactionality: #95 incumbent; overlapping exporter attempts #105/#111 closed unmerged;
- parity oracle: #134 incumbent; #136 closed unmerged after its distinct non-vacuous/zero-step authority finding was routed back to #134.

Current closed-unmerged duplicate set recorded by the machine snapshot: `94, 97, 98, 102, 105, 111, 136`.

This is intentionally conservative. A closed branch can retain useful evidence or a transferable finding, but it must not become a second implementation owner.

## Audit and governance boundary

Live `main` remains bootstrap-only at `f2e94c7212888cdb960bb66154d56d210e9b27ab` and unprotected. AUDIT-A #13 and AUDIT-B #14 still have actual verdict `CHANGES_REQUIRED`; later retest handoffs are not verdict upgrades. Vulnerability/license adjudication, canonical-main protection and release authority remain independent blockers.

Drive `SWARM_CONTROL` is supporting context only and is stale for the persistent-five scheduler (`ACTIVE_GENERATION: NONE`, `WAITING_FOR_AUDITOR_51`). GitHub exact SHA/PR/CI remains authority.

## Validator v2

`evidence/swarm_exp_01/d01_late_wave_intake_snapshot_20260825.json` now uses `12-6.s0-late-wave-intake.v2` and supports the W1 intake vocabulary:

`INCUMBENT / GREEN / RED / QUEUED / SUPERSEDED / DUPLICATE / COMPOSABLE / HOLD`.

Classifications are orthogonal. For example, an incumbent can be RED without losing ownership, while a required head must be GREEN/COMPOSABLE and cannot simultaneously be RED, QUEUED, HOLD, DUPLICATE or SUPERSEDED.

The validator requires terminal-success workflow evidence for every GREEN/COMPOSABLE claim, terminal failure evidence for RED, pending evidence for QUEUED, keeps S1/D10 held out of S0 Product intake, preserves one collision incumbent, records closed duplicate pruning, keeps promotion false, preserves historical audit verdicts, and requires full exact-head workflows plus both independent audit handoffs after the next real composition.

Run on an exact checkout:

```text
python tools/validate_late_wave_intake.py evidence/swarm_exp_01/d01_late_wave_intake_snapshot_20260825.json
pytest -q tests/test_late_wave_intake.py
```

## Next W1 action after this registry is exact-green

Do not open another wrapper. Recheck #90/#91/#100 heads and changed filenames, then construct the smallest ancestry-preserving composition of those exact green deltas on the selected Product lineage. After any composition head moves, rerun CI, real S0 training, fresh-process repeatability and strict exact-candidate evaluation on that exact SHA. Only then hand that exact head to AUDIT-A and AUDIT-B. Main protection/release adjudication remains a separate governance gate.
