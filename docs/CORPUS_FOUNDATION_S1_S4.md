# External corpus foundation: S1-S4 readiness

This package extends the exact-green D03 external-source foundation without changing any S0 fixture byte, split assignment, tokenizer, model, trainer, checkpoint, or D06 evaluation implementation. It approves zero external sources by itself.

## Fail-closed corpus eligibility

Source rights and record-level data-engineering metadata are separate gates. An external source version must already satisfy the D03 source registry: immutable snapshot URI, byte size and SHA-256, explicit rights status, non-`NOASSERTION` license state, durable policy/reviewer references, and explicit permission for model training. Public availability is not permission.

Every record then carries metadata-only hook results for `quality`, `language`, `pii`, and `copyright`. These hooks can be `PASS`, `REJECT`, `REVIEW_REQUIRED`, or `NOT_RUN`; only a complete all-`PASS` set can enter a corpus-eligibility manifest. A hook result cannot upgrade or approve source-level rights.

`build_corpus_eligibility_manifest()` streams record metadata, binds it to an exact source-registry identity, and produces a deterministic SHA-256 manifest. It retains only aggregate counts, accepted source-version identities, and one running metadata hash, rather than materializing document text.

## D06 reserved benchmark bridge

D06 owns benchmark/evaluation semantics. D03 owns exclusion from training. `reserved_registry_from_d06()` consumes the exact `12-6.benchmark-registry.v1` manifest emitted by D06, verifies its `manifest_sha256`, requires every imported item to be held out, rejects any training use, and converts the identities into the D03 reserved registry. Optional normalized-content fingerprints remain D03-side decontamination metadata.

`contamination_gate()` fails on either reserved `source_id` overlap or exact normalized-content SHA-256 overlap. Tests inject both classes of contamination. This is exact identity decontamination, not a claim of universal semantic contamination detection.

## Exact and near deduplication

`ExactDedupPlan` defines a deterministic Parquet/fsspec execution seam. Records are partitioned by exact `content_sha256`; each partition can be processed independently with deterministic tie-break fields (`source_id`, `source_version`, `record_id`). The memory contract is one hash partition at a time.

`DataTroveMinhashPlan` defines a four-stage DataTrove seam over immutable Parquet shards:

1. `ParquetReader -> MinhashDedupSignature`
2. `MinhashDedupBuckets`
3. `MinhashDedupCluster`
4. `ParquetReader -> MinhashDedupFilter -> ParquetWriter`

The plan pins the same DataTrove `0.10.0` compatibility target as the parent D03 foundation, binds all intermediate/logging URIs and MinHash parameters into a plan hash, and requires `skip_completed=true`. `validate_datatrove_minhash_runtime()` refuses a different installed DataTrove version or missing required symbols. This package does not claim a DataTrove production run unless exact runtime evidence exists.

## Deterministic streaming and resume

`StreamingShardPlan` assigns a record to a shard from `SHA-256(assignment_salt + NUL + record_id) mod shard_count`. Worker count is deliberately absent from the identity, so changing local concurrency cannot silently reshuffle a frozen corpus. `iter_shard()` is a generator and enforces a maximum normalized record size.

`StreamingResumeManifest` binds completed shard IDs to record counts, output sizes, output SHA-256 values, and the exact shard-plan SHA-256. Resume rejects a different plan or shard count and returns only still-pending shard IDs.

## Evidence and non-claims

The committed S1-S4 readiness backlog is `configs/data/s1_s4_corpus_readiness.json`. Every stage remains `NOT_READY`. Acceptance requires concrete hashes, source-rights decisions, D06 identity binding, decontamination injection evidence, deterministic rebuild/resume, mechanics benchmarks, and an independent AUDIT-B verdict at corpus-freeze time.

This package does not:

- approve a real external source;
- mutate the controlled S0 fixture;
- claim semantic/near-duplicate benchmark cleanliness from exact hashes;
- claim DataTrove production execution without runtime evidence;
- authorize paid compute;
- freeze a corpus or promote any model stage.
