# Corpus provenance foundation and S1-S4 readiness

This package extends the exact-green D03 external-source foundation without changing any S0 fixture byte, split assignment, tokenizer, model, trainer, checkpoint, evaluator or release surface.

## Authority and truth boundary

The physical repository is `Oleksii-debug/12-6-ai.`. The parent D03 provenance head is `ff39bf9285a1c59458a848a4e52d13135d76248b`; CI run `32646727768` completed successfully. The external-source and reserved-fingerprint registries remain intentionally empty. This package approves zero external training sources and does not substitute engineering metadata for legal/owner approval.

Source rights and record-level policy hooks are separate gates. A source must first pass the existing D03 immutable snapshot and rights contract. Record policy metadata then requires explicit PASS evidence from quality, language, PII and copyright hooks. `NOT_RUN`, `REVIEW_REQUIRED` and `REJECT` are fail-closed. A hook PASS is not itself a source-rights decision.

## D06 reserved benchmark bridge

`reserved_registry_from_d06_manifest()` consumes D06's `12-6.benchmark-registry.v1` manifest without modifying D06 code. It recomputes the D06 manifest hash, requires held-out entries, rejects any held-out benchmark that permits training-like uses, and translates exact benchmark source identities plus supplied normalized-content fingerprints into the D03 reserved registry.

The regression suite injects a reserved benchmark fingerprint into a training record and requires the D03 contamination report/gate to reject it. This is exact-content/source identity evidence, not a claim of semantic decontamination.

## Exact and near dedup seam

Local/free exact-dedup mechanics use a SQLite `PRIMARY KEY` fingerprint index so memory does not grow with corpus cardinality. Production near-dedup is deliberately not reimplemented: `DataTroveDedupPlan` binds the exact content fingerprint key, DataTrove MinHash engine, signature/bucket/n-gram configuration, source/reserved registry identities, URIs and task topology into a deterministic plan hash. DataTrove `0.10.0` remains the current compatibility target inherited from the parent D03 foundation and must be revalidated before version changes.

Large-corpus execution should materialize Parquet through the existing DataTrove/fsspec seam. The plan is infrastructure evidence only until an actual maintained DataTrove MinHash run is executed and retained.

## Deterministic sharding and resume

`StreamingShardPlan` assigns records by SHA-256 of `(partition_salt, record_id)` modulo a fixed shard count. It binds source and reserved-registry identities, Parquet/fsspec output, compression, row-group target and an explicit maximum in-memory record batch.

`build_resume_manifest()` records ordered `(shard, part)` artifacts with SHA-256, byte size and record count. Part indexes must be contiguous per shard. The manifest binds the streaming plan and dedup plan hashes and is self-hashed. `validate_resume_manifest()` reconstructs the expected object and rejects any tampering or incompatible plan.

## LOCAL_FREE synthetic mechanics benchmark

Command:

`PYTHONPATH=src python tools/benchmark_corpus_foundation.py --records 25000 --duplicate-every 11`

Recorded report: `reports/d03/corpus_foundation_local_benchmark_20260824.json`.

Observed in the disposable LOCAL_FREE Linux environment on Python 3.13.5: 25,000 synthetic records, 2,272 injected exact duplicates removed, 22,728 unique records, 32 deterministic shards, approximately 37,235 input records/second, 25,635 bytes peak Python allocation measured by `tracemalloc`, and a 1,810,432-byte SQLite exact-dedup index. These measurements are environment-specific mechanics evidence. They do not measure DataTrove MinHash throughput, remote object-store throughput, corpus quality, GPU training or paid compute.

## S1-S4 readiness

Machine-readable acceptance backlog is `configs/data/corpus_readiness_s1_s4.json`. Every stage is `NOT_READY`. S1 requires at least one actually reviewed source plus immutable acquisition and contamination evidence. S2 requires real DataTrove/Parquet/fsspec execution and restart evidence. S3 requires multilingual/policy calibration and near-contamination quality measurements. S4 requires a reproducible corpus-freeze candidate bound into D04/D06/D10 and independently adjudicated by AUDIT-B.
