# External source acquisition receipts

This package is a collision-safe adjunct to the D03 external-source registry and the D09/D03 corpus-foundation package. It does not change S0 fixture bytes and does not modify the corpus-foundation files owned by PR #64.

## Boundary between provenance, retrieval, and training eligibility

The external-source registry remains the authority for `source_id`, `source_version`, immutable snapshot URI, expected byte size, expected SHA-256, upstream version, retrieval method, and the current rights-review status.

A `SourceRetrievalPlan` binds all of those immutable expectations to one destination and a fixed chunk size. The plan deliberately records rights as `OBSERVED_ONLY_NOT_APPROVAL` and sets `training_eligibility_evaluated=false`. Byte acquisition can therefore be tested for a source whose rights are still `REVIEW_REQUIRED` without silently converting retrieval success into permission to train.

Corpus construction must continue to use the source-rights and record-policy gates from the parent D03/D09 packages. A verified retrieval receipt is necessary byte-provenance evidence, not sufficient train eligibility.

## Bounded-memory local staging and resume

`verify_and_stage_local_mirror()` reads one configured chunk at a time, writes to a sibling `.partial` path, maintains SHA-256 state, flushes and fsyncs the partial file, and atomically publishes the final destination only after both exact registered size and exact registered SHA-256 match.

Existing partial data is hashed into a `RetrievalCheckpoint`. Checkpoint chunks must be contiguous, bound to the exact plan SHA-256, and end at a full chunk boundary unless the entire expected file is complete. Before appending, every completed partial chunk is compared to the source prefix. A tampered partial therefore fails before it can become a final artifact.

A hash or size mismatch never creates the final destination. The `.partial` artifact is intentionally left for diagnosis or an explicit retry rather than being mislabeled as verified data.

## Deterministic receipts and inventory

`VerifiedRetrievalReceipt` binds:

- exact retrieval-plan SHA-256;
- exact source-registry identity;
- source/version and source/destination URIs;
- expected and verified SHA-256;
- verified byte size;
- number of chunks and chunk-manifest SHA-256;
- observed rights status, explicitly without approval semantics.

`build_retrieval_inventory()` validates every receipt against the same exact external-source registry, rejects duplicate source versions, rejects registry/hash/size drift, and emits a deterministic inventory SHA-256. This inventory is the handoff boundary to later Parquet/dedup/sharding work.

## fsspec and remote acquisition

The module includes a lazy `validate_fsspec_runtime()` seam so remote acquisition can use the maintained fsspec stack without making it a base dependency. This package does not claim a remote/object-store retrieval run. Remote fsspec execution remains `NOT_TESTED` until exact runtime/version, credentials handling, interruption/resume, final object checksum, and storage-cost evidence are captured on an authorized environment.

Stable URI validation rejects embedded credentials, query strings, and fragments from provenance manifests. Secrets belong in runtime credential providers, never durable source metadata.

## Evidence and non-claims

The local synthetic benchmark stages 32 MiB with 1 MiB chunks using LOCAL_FREE disk I/O. It measures mechanics only. It is not a network-throughput benchmark and says nothing about corpus quality, copyright permission, PII clearance, benchmark cleanliness, or model quality.

This package approves zero real sources, authorizes zero paid compute, does not freeze a corpus, does not promote any stage, and does not issue an audit verdict.
