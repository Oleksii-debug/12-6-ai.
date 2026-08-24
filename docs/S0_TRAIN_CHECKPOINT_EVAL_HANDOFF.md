# S0 train -> checkpoint -> evaluation handoff

This document describes the integration-owned handoff prepared during
`12-6-AI-SWARM-EXP-01`. It does not implement D06 evaluation semantics and it does
not grant candidate, audit, release, paid-compute, or promotion authority.

## Current authority

The handoff is stacked on dependency-lock PR #58 exact source
`3d5d2332577d1ccb2b6ecbb5197b1d95a4baba6f`, whose CI run `32742220948`
completed successfully. Its convergence parent is PR #57 exact source
`c1e37854829faa96291ee76088f703f5096ea10b`, CI run `32739239590`, which proved
the LOCAL_FREE D03 -> D04 -> D01 -> D02 -> D05 -> D07 path including a real
optimizer step, SafeTensors checkpoint, fresh trainer/model reload, restored
state, and generation.

The environment-lock identity inherited from PR #58 is
`61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac`.

Machine-readable authority is in
`configs/releases/s0_handoff_20260824.prepared.json`. The complete ordered
D01-D08 evidence map has SHA-256
`021768ace9f464dd6301abe4dfe37fde8394f97043dc4134aa8578811b83174b`.

## Current blocker

D06 exact source `914973502ab92a925a5cc29d72e4b3cce0e81c80` is held. Repository CI run
`32646767818` and dedicated D06 evaluation workflow run `32646767856` both
completed with failure before pytest because of a D06-owned Ruff failure. D01
integration must not silently repair or accept that domain-owned source.

Consequently the committed handoff state is `PREPARED_BLOCKED`. Seven required
lanes are accepted with exact-head success evidence; D06 remains held. The
handoff validator will reject `READY_LOCAL_FREE` while any required lane is held
or any explicit execution blocker remains.

## Transition after D06 repair

After D06 publishes a *new* exact source SHA with completed successful CI and its
required evaluation workflow:

1. Selectively intake only that exact D06 source on top of the locked composition.
2. Preserve the D01-D05/D07/D08 ancestry and the dependency-lock identity.
3. Run the full locked repository CI on the new combined source; do not inherit
   PASS from any parent SHA.
4. Update the machine handoff evidence with the new D06 SHA/run IDs, recompute
   `component_map_sha256`, remove the D06 execution blocker, and change
   `handoff_state` to `READY_LOCAL_FREE` only if validation succeeds.
5. Resolve the C01 run manifest from the exact combined candidate rather than
   copying unresolved template values.
6. Execute only the authorized LOCAL_FREE S0 sequence: train -> D05 checkpoint
   -> verify/reload with a fresh D02 Trainer -> D06 evaluation -> D07 generation
   evidence.
7. Retain exact run manifest, metrics, checkpoint manifest/payload hashes, and
   evaluation report at the paths named by the handoff artifact contract.

`READY_LOCAL_FREE` is execution readiness only. `promotion_allowed` remains
false in this handoff contract. AUDIT-A and AUDIT-B must independently retest the
exact future candidate before any promotion decision.

## Validator

Validate the prepared blocked evidence:

```bash
python tools/validate_s0_handoff.py configs/releases/s0_handoff_20260824.prepared.json
```

Require execution readiness (expected to fail while D06 is held):

```bash
python tools/validate_s0_handoff.py configs/releases/s0_handoff_20260824.prepared.json --require-ready
```

The validator also rejects abbreviated Git identities, missing required lanes,
accepted components without exact-head success, tampered component evidence,
paid-compute authorization, behavioral/foreign-pretrained Base inputs, and any
attempt by this handoff to self-authorize promotion.
