# NEXT100-065 V4 object-level successor dedup intake

## Decision

`BLOCKED_UPSTREAM_SOURCE_CONVERGENCE_NONTERMINAL`

The live NEXT100-063 convergence head reports 320,632 bytes / 11 independent families / 22 numeric source objects before successor global dedup. Its late positive-credit authorities contribute 76,662 bytes across 11 exact objects:

- KMu Secretariat: 9,153 bytes / 6 objects / `ua.kmu.portal.secretariat-news`.
- NIST Technical Series: 59,358 bytes / 3 objects / `en.usgov.nist.technical-series`.
- MDN prose-only: 6,492 bytes / 1 object / `en.mdn.webdocs.prose`.
- Ukrainian public-domain Nomis1864 snapshot: 1,659 bytes / 1 object / `ua.verba.public-domain.nomis1864`.

This V4 intake materializes those 11 object-level identities from their exact terminal authorities. Each row binds source/family identity, stable origin/object identity, modality, capacity bytes, raw content identity, and authority-specific comparison-payload SHA-256/byte count.

## Exact upstream binding

- PR: #527
- issue: #521
- head: `5356d60c8c8af46d6fc34debfd3cb36731045338`
- CI run observed: `33005956076`
- state at binding time: queued / nonterminal

Because upstream CI is not yet terminal-success, the V4 snapshot deliberately remains blocked. When that exact run is terminal-success, updating only the bound workflow state is enough for the intake validator to reach `READY_FOR_GLOBAL_DEDUP_OBJECT_COMPARISON`, assuming no source-convergence drift.

## Executable safety contract

`src/twelve_six/data/cross_source_dedup_v4_intake.py` enforces:

1. exact upstream convergence schema/vector binding;
2. terminal-success upstream CI before comparison authorization;
3. explicit binding of every positive-credit late authority, including any newly introduced authority;
4. exact authority identity, family, stratum, object count, and byte total;
5. globally unique source IDs and stable object IDs;
6. raw content identity plus exact authority-specific comparison-payload identity;
7. per-object family/modality consistency;
8. aggregate bytes equal the sum of object rows;
9. CPython docs remain zero-credit until their accepted-chunk eligible-byte ledger exists.

`READY_FOR_GLOBAL_DEDUP_OBJECT_COMPARISON` authorizes only execution of the comparison algorithm. It does not claim post-dedup capacity, balance/diversity PASS, Research Corpus V1, tokenizer readiness, training readiness, or any learned checkpoint.

## Current vector and gap

The source-convergence authority reports:

- Ukrainian: 100,856 bytes / 4 families.
- English: 150,643 bytes / 3 families.
- Code: 69,133 bytes / 4 families.
- Total: 320,632 bytes / 11 families.
- Remaining frozen 20M source-capacity gap: 19,679,368 bytes.

These are source-capacity bytes, not training tokens or unique causal-loss positions.

## Validation

The regression suite proves:

- complete object handoff still blocks on nonterminal upstream CI;
- terminal exact upstream makes only the dedup-comparison gate ready;
- aggregate-only handoff blocks;
- a newly positive but unbound authority blocks;
- duplicate object identity blocks;
- missing content hash blocks;
- unsupported comparison normalization blocks;
- family/count/capacity drift blocks;
- zero-credit CPython poisoning blocks.

Run locally:

```bash
PYTHONPATH=src python -m pytest -q tests/test_next100_065_cross_source_dedup_v4_intake.py
PYTHONPATH=src python tools/validate_next100_065_cross_source_dedup_v4_intake.py --expect-status BLOCKED_UPSTREAM_SOURCE_CONVERGENCE_NONTERMINAL
```

## Safety boundary

LOCAL_FREE only. No model training, tokenizer fit, final-test payload access, paid compute, corpus identity, or learned-checkpoint claim is introduced by this work.
