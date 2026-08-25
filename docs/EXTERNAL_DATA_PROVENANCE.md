# External data provenance and scalable ingestion

This document defines the D03 contract that must be satisfied before an external source can enter canonical 12-6 pretraining.

## Truth boundary

The existing S0 corpus is a controlled project-authored fixture. This package does **not** add an external training source and does not claim any external corpus is license-clean, contamination-clean, or approved for training.

`data/external/external_sources.json` is intentionally empty. A source may be added only with an exact version, immutable snapshot reference and hash, explicit provenance, and a rights decision. `NOASSERTION`, `REVIEW_REQUIRED`, or a source that does not explicitly allow model training fails closed.

`data/external/reserved_fingerprints.json` is also intentionally empty until D06 and D03 register actual benchmark/evaluation/test identities. It exists to bind source IDs and normalized-content SHA-256 fingerprints that canonical training must reject.

## Required external source fields

Every source version must record:

- stable `source_id` and `source_version`;
- provider and non-secret upstream `source_url`;
- source kind and declared purpose;
- explicit synthetic / benchmark / held-out booleans;
- immutable snapshot URI outside Git;
- exact snapshot SHA-256 and byte size;
- retrieval time, upstream version, and retrieval method;
- rights review status;
- license identifier and terms URL;
- explicit model-training / derivatives / redistribution booleans;
- durable policy, reviewer, and review-time references.

A source is train-eligible only if its purpose is training/pretraining, it is not benchmark or held-out material, and rights status is exactly `APPROVED_FOR_TRAINING` with model training explicitly allowed. The registry does not infer permission from a public URL or from the mere presence of a license string.

## Immutable snapshot rule

Upstream network URLs are discovery/provenance references, not canonical training bytes. Canonical bytes must first be copied into controlled immutable storage and bound by byte size + SHA-256. Secret-bearing presigned/query URLs, embedded credentials, and path traversal are rejected from snapshot metadata.

`verify_local_snapshot()` provides a local fail-closed verifier. Remote/object-store acquisition jobs must produce the same size/SHA-256 evidence before the source can be promoted into a corpus manifest.

## Benchmark / held-out contamination

`ReservedSetSpec` records a versioned evaluation/test source identity and zero or more normalized-content SHA-256 fingerprints. `contamination_report()` checks training records for both source-ID collision and exact normalized-content collision and emits a hashed machine-readable report.

This deliberately complements D06's benchmark registry. D06 owns evaluation semantics and stage gates; D03 owns training-source provenance and corpus exclusion. Source-level and fingerprint-level checks are necessary but are not a claim of universal semantic/near-duplicate decontamination.

## Scale seam: DataTrove + Parquet + fsspec

For large structured snapshots, D03 uses a thin optional adapter instead of rewriting distributed I/O. `DataTroveParquetPlan` binds source/version, snapshot hash, source-registry identity, input/output/logging URIs, task count, worker count, and the validated DataTrove version into a deterministic plan hash.

The current compatibility target is DataTrove `0.10.0`. DataTrove supports local/Slurm/Ray-style executors, fsspec-backed local/remote storage, Parquet readers/writers, and task-completion markers. The base package does not install DataTrove; `build_datatrove_executor()` imports it lazily and requires `datatrove[io]==0.10.0` when an actual scalable ingestion run is executed.

The adapter currently performs provenance-preserving structured staging only. Normalization, language ID, quality filtering, PII hooks, dedup, contamination, splitting, and corpus identity remain explicit D03 stages. At scale those stages should be implemented with mature DataTrove blocks/MinHash or equivalent primitives without changing the provenance/manifest contracts.

## No silent mutation

Changing source version, snapshot bytes, rights decision, reserved fingerprint set, or corpus membership changes identity and therefore requires a new registry/corpus version and hashes. Training code must never rewrite a held-out split or silently reinterpret an unresolved source as train-eligible.

## S0 / next validation

This package is infrastructure only. Before the first external source is admitted, D03 must execute acquisition against a real reviewed source, verify immutable snapshot hashes, build Parquet shards through the optional backend, run filtering/dedup/decontamination, and hand the exact registry/corpus identities to D04/D06/D10 and AUDIT-B.
