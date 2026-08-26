# D03 DATA-12 scalable deduplication and decontamination

## Scope and incumbent ownership

This package is stacked on the exact open D03 corpus-foundation head `cd259202bd7fd2bdbb4d75a40cb7b67bf0908593` from PR #64. It extends, rather than replaces, the incumbent D03 contracts:

- `SQLiteExactDedupIndex` remains the bounded-memory exact-dedup primitive for LOCAL_FREE mechanics;
- `StreamingShardPlan` remains the deterministic record-to-shard contract;
- D03's `reserved_registry_from_d06_manifest()` remains the bridge from D06 evaluation identities into training exclusions;
- D06's `12-6.benchmark-registry.v1` remains the only benchmark-registry authority.

No second benchmark registry is introduced. No real external source is approved by this package.

## DataTrove execution seam

The previous D03 foundation represented scalable MinHash as a deterministic plan but did not execute DataTrove. DATA-12 adds an executable layer pinned to DataTrove `0.10.0` and to the exact PyPI wheel SHA-256:

`c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d`

The maintained DataTrove 0.10.0 MinHash configuration uses:

- 5-word shingles;
- 14 LSH buckets;
- 8 hashes per bucket;
- seed 1;
- 64-bit hashing.

Therefore the actual signature contains `14 * 8 = 112` hash values. This is made explicit in `DataTroveMinhashExecutionPlan`; the older generic D03 `minhash_signature_size` field is not silently reinterpreted.

The stage topology is also explicit because DataTrove imposes different task counts by stage:

1. reserved benchmark signatures: one local task for the controlled fixture;
2. reserved benchmark index: 14 bucket tasks;
3. candidate signatures: configured candidate-shard task count;
4. candidate bucket matching: 14 bucket tasks;
5. clustering: one task;
6. filtering: the same task count and input partitioning used for candidate signatures.

All local executors use `skip_completed=true`. The experiment deliberately executes only a prefix of signature tasks first, then reruns the full signature stage against the same logging/output identity to prove rank-level restart/skip behavior.

## Benchmark decontamination through the maintained MinHash index

DataTrove 0.10.0 already supports a MinHash reference index through `MinhashDedupBuckets(index_folder=..., create_index_name=...)`. DATA-12 uses that maintained mechanism rather than building another contamination engine.

The controlled D06 held-out samples are converted by the incumbent D03 bridge into the canonical reserved registry. The same samples are then used to build the MinHash reference index. Candidate bucket matching runs with `only_dedup_in_index=false`, so the same stage can find candidate-candidate near duplicates and candidate-to-reserved-benchmark matches. Clustering retains the DataTrove sentinel semantics so the reference side is not emitted into the candidate corpus.

Exact source-ID and normalized-content exclusions still run before MinHash. This preserves cheap deterministic exclusions while MinHash handles lexical near overlap.

## Controlled LOCAL_FREE injection experiment

Canonical configuration: `configs/data/dedup_scale_local_free_v1.json`.

The experiment contains 50,000 deterministic synthetic candidate records and 128 reserved benchmark records. It injects:

- 500 exact duplicates;
- 500 English near duplicates;
- 500 Ukrainian near duplicates;
- 300 copied/edited code samples;
- 500 boilerplate-heavy negative controls;
- 100 exact benchmark-content copies under a non-benchmark source ID;
- 50 records carrying the reserved benchmark source ID;
- 250 English lexical near copies of held-out benchmark samples;
- 250 Ukrainian lexical near copies of held-out benchmark samples;
- 250 cross-language benchmark transformations intentionally constructed as known semantic relations with low lexical overlap.

The last group is not expected to be reliably caught by lexical MinHash. Its purpose is to make the truth boundary measurable: known semantic residuals are counted and keep the resulting synthetic candidate ineligible even if exact and lexical decontamination are strong.

## Measurements and acceptance

The experiment records:

- exact-stage wall time and records/second;
- SQLite index size and exact-stage `tracemalloc` peak;
- process and child peak RSS where the operating system exposes it;
- deterministic shard counts and skew;
- DataTrove wall time and input records/second;
- restart evidence from partial then resumed signature execution;
- exact-removal counts;
- internal near-duplicate family recall and false negatives;
- benchmark lexical near-overlap recall and survivors;
- cross-language semantic-calibration detection rate and survivors;
- boilerplate-negative removals;
- unexpected-removal rate;
- final corpus reduction ratio;
- per-output artifact hashes, metrics hash, input identity, D06/reserved identity and scale-plan identity.

The LOCAL_FREE acceptance thresholds are deliberately narrow mechanics thresholds rather than a corpus policy:

- internal injected near-family recall >= 0.90;
- injected lexical benchmark recall >= 0.95;
- unexpected-removal rate <= 0.005;
- restart verification must pass.

These thresholds do not approve a source or set a universal future dedup threshold.

## Training eligibility

`build_training_eligibility_envelope()` binds the dedup output manifest to three separate classes of gate:

1. source-rights eligibility from the existing D03 source registry;
2. record-policy eligibility from existing D03 quality/language/PII/copyright evidence;
3. dedup/decontamination experiment status plus residual exact, lexical and known-semantic benchmark overlap.

The committed experiment is synthetic mechanics and explicitly sets source-rights and record-policy eligibility to false. It therefore cannot become train-eligible. A future approved corpus can only become eligible when those upstream gates are independently true and registered/known benchmark residual counts are zero.

A zero residual count is never represented as proof of universal semantic cleanliness. The output envelope carries `semantic_universal_cleanliness_claimed=false` by construction.

## Reproduction

The dedicated GitHub Actions workflow installs the exact DataTrove 0.10.0 wheel only after verifying its SHA-256, runs targeted contract tests, then executes the 50,000-record experiment:

```bash
PYTHONPATH=src python tools/run_dedup_scale_experiment.py \
  --config configs/data/dedup_scale_local_free_v1.json \
  --output reports/data12/dedup_scale_report.json
```

The workflow uploads the exact report and `/usr/bin/time -v` output as retained Actions evidence. Concrete observed metrics are recorded only after an exact-head workflow succeeds; configuration or theoretical values are not substituted for runtime evidence.

## Next corpus-scale target

After this 50,000-record mechanics run is green, the next LOCAL_FREE/controlled target is:

- 1,000,000 records;
- at least 1 GiB uncompressed text;
- 64 deterministic shards;
- the same D06 reference-index semantics and recorded plan identity;
- a deliberately interrupted stage followed by resume;
- retained wall-clock throughput and peak-RSS evidence;
- a streaming scorer rather than materializing all outputs for experiment analysis.

No approved-source run should start until the exact source versions have explicit D03 rights approval and immutable snapshot identities. Public GitHub or other public availability is not training permission.
