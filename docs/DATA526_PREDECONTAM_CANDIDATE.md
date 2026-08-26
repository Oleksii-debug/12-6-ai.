# DATA-526 — exact pre-decontamination candidate records

## Result

This package creates a deterministic record-level input for the next evaluation-decontamination step. It does not create Research Corpus V1, training shards, tokenizer-fit authority, or training authorization.

Frozen candidate identity:

`749d1449182abb4d71f90eb3510fb212c5ac8f90d15d8ff60a407b0cebd1baaa`

Record inventory identity:

`3b3f6cda92b248d327861e335ec9ccc4ad6fb6a250ac020c9618bf6f14310f21`

Authority bundle identity:

`4d338d4fb79c37afa501cb64663e6a7ca329b1f643f9ad104d489d90aac01b85`

## Exact candidate

The candidate contains 9 exact normalized records and 243,898 normalized UTF-8 bytes:

- Ukrainian: 90,044 bytes, 2 records, 2 independent families.
- English: 144,151 bytes, 5 records, 2 independent families.
- Code: 9,703 bytes, 2 records, 2 independent families.

The consumed authorities are:

1. DATA-287 incumbent five-record external snapshot registry at `b0523ccbc4b957615aac849d476cfa851be87578`, registry identity `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`.
2. NEXT100-022 bounded Ukrainian Wikisource snapshot at `84c51e42b6daa51796fd20d793b5ef1ff01cc9d2`, authority identity `6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6`.
3. NEXT100-034 terminal NIST technical-series subset at `b7491745b34ac8679baaf69cb96cd609dcbe0a16`, terminal payload identity `3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c`.

Each scoped authority has a completed successful dedicated workflow recorded in the input manifest.

## Why CPython is deliberately not in this exact record set

NEXT100-037 is a valid terminal source authority, but it explicitly permits only 14 of 16 derived chunks; two chunks are rejected for phone-like PII. The authority publishes the 14 accepted normalized hashes but not exact byte counts for each accepted chunk.

The pre-decontamination builder requires every admitted record to have an exact normalized hash and positive byte count. Treating the full 17,901-byte normalized CPython source as one allowed record would silently re-admit the two rejected chunks. DATA-526 therefore fails closed and holds CPython until those 14 accepted chunks are materialized as exact records. NIST supplies the second independent English family without weakening this boundary.

## Concurrency boundary

NEXT100-026 KMu is terminal-success but is not silently composed here because the active NEXT100-063 source-registry convergence lane owns late source aggregation. This DATA-526 package intentionally creates the smallest exact record inventory that satisfies the `>=2` independent-family pre-decontamination gate for Ukrainian, English, and code without replay.

Later terminal source-registry convergence may create a larger successor candidate identity. It must not mutate this frozen identity in place.

## What is still blocked

All 243,898 bytes remain pre-decontamination candidates only. Authorized training exposure is zero until downstream gates produce a terminal final corpus.

Required next order:

1. independently re-verify this candidate identity;
2. run exact-hash, normalized-hash, fragment and near-copy evaluation decontamination against terminal reserved-evaluation authorities;
3. rerun cross-source exact/near dedup on the resulting candidate;
4. rerun post-composition quality and privacy checks;
5. perform cluster-safe split and deterministic packing/sharding;
6. prove two clean builds are byte-identical;
7. build the exact post-pack unique causal-loss ledger;
8. only then reconsider tokenizer fit and the bounded 20.6M-parameter campaign.

## Truth boundary

`LOCAL_FREE` only. No optimizer updates, no long training, no paid compute, no final-test payload access, no replay, no corpus freeze and no learned-20M claim.
