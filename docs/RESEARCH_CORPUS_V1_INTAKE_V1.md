# Research Corpus V1 intake V1

## Decision

`SOURCE_AUTHORITY_INTAKE_READY_MATERIALIZATION_REQUIRED`

This package performs the first ordered action from the live ~20M readiness controller. It binds terminal source authorities needed to remove the incumbent 1/1/2 independent-family bottleneck without claiming that a trainable corpus already exists.

## Bound authority vector

The incumbent DATA-287 registry remains fixed at head `b0523ccbc4b957615aac849d476cfa851be87578`, identity `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`, with 5 snapshots, 4 independent families and 183,061 normalized source bytes.

Two successor source authorities are admitted into the intake plan:

- Ukrainian KMu Secretariat: NEXT100-026 head `40950a950b60921fd856af2719e1ae2486d9e892`, dedicated run `32997970539` success, 6 immutable records, 9,153 normalized bytes, family `ua.kmu.portal.secretariat-news`, training allowed under its bound CC BY 4.0 policy.
- English CPython documentation: NEXT100-037 head `5a6a495a24bce449334cbc5126d0114f61a9f57c`, dedicated run `32998356906` success, family `python.cpython.documentation`. The source has 17,901 normalized bytes, but only 14 exact chunk identities are accepted; 2 chunks remain rejected by the incumbent phone-PII predicate.

At source-authority level this yields the minimum independent-family vector `UK=2 / EN=2 / code=2`. This is not a corpus-level diversity PASS until the exact objects are materialized together and global dedup/lineage checks confirm that independence survives composition.

## Capacity boundary

`183,061 + 9,153 + 17,901 = 210,115` bytes is recorded only as the pre-filter normalized-source envelope. It is not training capacity because CPython rejected chunks have not been subtracted into a single materialized candidate, evaluation decontamination has not run, global dedup has not run, split/packing has not run, and the unique causal-loss ledger does not exist.

Therefore:

- exact training-eligible bytes: `null`;
- exact post-pack unique nonignored loss positions: `null`;
- authorized optimized targets: `0`;
- real ~20M long training: `BLOCKED`.

## Required successor sequence

1. Materialize the exact accepted KMu and CPython objects alongside DATA-287 on one successor candidate.
2. Freeze a sorted record inventory and candidate identity.
3. Run exact/normalized/fragment/near-match/lineage evaluation decontamination without reading final-test outcomes.
4. Run global dedup, quality, privacy and family accounting.
5. Build cluster-safe train split and deterministic shards twice from clean roots.
6. Emit the exact post-pack unique nonignored causal-loss ledger.
7. Requalify ~20M checkpoint and optimizer mechanics on the frozen corpus contract.
8. Refresh the ~20M campaign preregistration. Paid compute remains fail-closed until separately authorized.

## Truth boundary

No model training, optimizer update, learned checkpoint, final-test outcome access, paid compute, corpus freeze, representativeness claim or stage promotion is produced by this intake package.
