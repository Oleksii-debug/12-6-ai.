# S0 exact-candidate train → checkpoint → eval handoff

Status: **PREPARED / READY_LOCAL_FREE / NOT PROMOTED**

This package is an integration/control follow-on to PR #81. It does not change model,
tokenizer, data, trainer, checkpoint, evaluator, or inference semantics. The Product
target remains exactly:

- repository: `Oleksii-debug/12-6-ai.`
- PR: #81
- branch: `d01/s0-candidate-convergence-20260824-b`
- Product SHA: `1caa729c8efafc84e7a5c4b1f7295eb8dcdb5a8d`
- exact-head CI: `32761570313` = completed success
- exact-head D02 real S0 training: `32761570314` = completed success

The D02 exact-head run already proved a real 40-step CPU pretraining path on the
integrated candidate: train loss 5.5579279558 → 2.7985758683, validation loss
5.5592589744 → 2.7907250028, 10,833 optimized tokens, and 10,140/10,140 parameter
elements changed. This handoff does not reinterpret that as promotion evidence.

## Why v2 exists

PR #59 (`d01/s0-handoff-gate-20260824`) is CI-green at
`19f813e780b544959c0bc0a5a4a523101a534461`, but its machine handoff remains
semantically stale: it holds D06 source `914973...` because that source was red at its
cutoff. PR #81 later selectively accepted exact-green D06 source
`89fa0b2c17ef78ba202d019a8959aa1a19377391` without weakening D05's strict checkpoint
identity contract. Therefore PR #59 is retained as useful history and v1 contract
provenance, but must not be used as the current launch gate.

## Exact execution contract

The resolved C01-style manifest is
`configs/runs/s0_10k.pr81_exact_candidate.local_free.json`. It binds:

- 10,140-parameter random-init Base;
- exact ModelSpec and InitSpec identities;
- deterministic S0 controlled dataset identity;
- byte tokenizer/vocabulary identities;
- packing identity;
- hash-locked environment identity;
- seed 1337, CPU fp32, AdamW, constant schedule, 40 target steps;
- `LOCAL_FREE` only; paid compute remains unauthorized.

The run must retain these outputs under the declared run root:

1. resolved run manifest;
2. D02 training evidence;
3. D05 strict checkpoint manifest plus manifest checksum;
4. D05 interrupted/resumed trajectory evidence;
5. D06 evaluation/stage-gate report;
6. D07 first-party inference report from the reloaded checkpoint.

Any identity mismatch, NaN/Inf, serialization mismatch, evaluation failure, or
authorization mismatch fails closed. Blind retry is prohibited.

## Audit and promotion boundary

`READY_LOCAL_FREE` means only that a local/free execution handoff has complete
composition evidence. It does **not** mean S0 is AUDITED, STABLE, canonical, or
release-approved.

AUDIT-A #13 and AUDIT-B #14 have been asked to retest the exact PR #81 Product SHA.
Until both independent audit records bind an allowed verdict to that same exact SHA,
promotion remains false.

## Collision and intake disposition

Current active surfaces deliberately not edited by this package:

- PR #77: D01 canonical integration state ledger;
- PR #79: secret/history and supply-chain triage;
- PR #80: live promotion-authority verifier;
- PR #62: dependency-security evidence.

Stale/current-history surfaces:

- PR #59: superseded for current handoff semantics, history retained;
- PR #45: older D01 convergence path, superseded for current S0 composition;
- PR #57: useful ancestor, not the current candidate.

Machine-readable authority and supersession evidence is in
`evidence/swarm_exp_01/d01_s0_authority_map_20260824.json`.

## Validation

Run:

```bash
python tools/validate_s0_exact_handoff.py
pytest -q tests/test_s0_exact_handoff.py
pytest -q
ruff check src tests tools
```

The validator cross-checks the handoff, PR #81 composition manifest, and resolved run
manifest. Tests include negative cases for stale/failed CI, component SHA drift,
obsolete held-lane semantics, tampered component evidence, paid compute, foreign
pretrained Base weights, run-identity drift, missing output artifacts, stale audit
metadata, and self-promotion.

The next execution action is to run the resolved manifest against exact Product SHA
`1caa729c...`, retain the complete artifact bundle, and then bind independent audit
verdicts to that exact target.
