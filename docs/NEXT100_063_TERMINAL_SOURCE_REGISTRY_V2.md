# NEXT100-063 Terminal Source Registry V2

## Decision

`PASS_FAIL_CLOSED_CANDIDATE_AUTHORITY_VECTOR`

V2 supersedes the V1 source-authority accounting identity `77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526` because V1 credited three sources that were not eligible for numeric capacity at the exact live cutoff.

## Corrections

1. CPython documentation PR #467 remains a useful terminal source authority, but the authority reports 16 chunks with 14 accepted and 2 rejected for `pii_phone`. The full source normalization is 17,901 bytes, while an exact accepted-chunk byte ledger/content materialization is not sealed. V2 therefore gives this source zero numeric capacity and zero family credit until a successor materializes the exact eligible payload.
2. Pydantic PR #465 is zero-credit because dedicated exact-head workflow run `32999061340` (`NEXT100-048 Pydantic Source Admission`) completed with `failure`. A generic DATA-227 success cannot substitute for the source-specific terminal workflow.
3. Rich PR #475 is zero-credit because dedicated exact-head workflow run `32999511493` (`NEXT100-051 Rich Source Admission`) completed with `failure`. A generic DATA-227 success cannot substitute for the source-specific terminal workflow.

## Corrected pre-global-dedup vector

- Ukrainian: 100,856 normalized bytes / 4 independent families.
- English: 150,643 normalized bytes / 3 independent families.
- Code: 14,977 normalized bytes / 3 independent families.
- Total: 266,476 normalized bytes / 10 independent families.
- Frozen minimum family gate: PASS before successor global dedup (`>=2` per stratum).
- Research Corpus V1 20,000,000-byte target gap: 19,733,524 bytes.
- Authorized balanced no-replay loss positions: exactly `0`.

Registry V2 identity: `934933896a4b3b01dd58cd18d13bcc36245913f83412c6b3f697c64dd03e4d4d`.

## Credited exact-head source authorities

V2 credits only KMu Secretariat, bounded Ukrainian Wikisource, Nomis1864, MDN prose, NIST technical series, and Starlette in addition to the five-source DATA-287 base. Each credited late source is bound to its dedicated exact-head workflow name, run ID and `success` conclusion in the machine-readable registry.

## Required next gates

The registry is not a corpus freeze. The executable sequence remains:

1. materialize an exact immutable candidate-record inventory bound to the V2 registry identity;
2. run global exact and near cross-source dedup;
3. rerun balance/diversity on the post-dedup inventory;
4. run evaluation decontamination without consuming final-test payloads;
5. rerun composed quality/privacy checks;
6. create cluster-safe split/shard/pack identities and prove deterministic clean rebuilds;
7. materialize the post-pack unique causal-loss ledger;
8. authorize tokenizer fitting only after the corpus gate is terminal;
9. authorize the learned ~20M campaign only when non-zero unique training exposure is proven.

No tokenizer fit, model training, optimizer update, paid compute, Research Corpus V1 freeze, learned 20M checkpoint or learned 100M checkpoint is claimed by V2.
