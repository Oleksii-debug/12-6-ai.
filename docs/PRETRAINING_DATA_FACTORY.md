# Pretraining data factory — D03/D09 scale execution

## Position in the live lineage

This package extends PR #75 on top of exact-green PR #64. It does not modify the six files owned by #64 or the six original acquisition files owned by #75. PR #68 remains superseded. The S0 controlled corpus remains a mechanics fixture and is not promoted into a future canonical corpus or tokenizer.

The factory composes existing authority instead of replacing it:

1. D03 external-source registry and `ExternalSourceSpec.assert_training_eligible()` remain the source-rights gate.
2. D09 source-acquisition receipts prove exact bytes and resume-safe retrieval but explicitly do not grant training permission.
3. The factory re-verifies the exact retrieved snapshot, then streams extraction, NFKC normalization, compatibility LID, quality/PII filtering, benchmark decontamination and disk-backed exact dedup.
4. Production near dedup is delegated to maintained DataTrove 0.10.0 MinHash. The bounded S0 Jaccard path exists only for LOCAL_FREE synthetic equivalence tests and fails above its configured document ceiling.
5. Deterministic split and shard identities are plan-bound. Final JSONL shards feed an explicitly non-canonical tokenizer-training handoff.
6. Maintained DataTrove `ParquetWriter` is the Parquet materialization path; no custom Parquet writer is introduced.

## Factory plan and restart boundary

`FactoryPlan` binds the exact source-registry identity, verified-retrieval inventory, reserved benchmark registry, split seed, shard topology, policy thresholds and DataTrove/MinHash configuration. Its SHA-256 is carried into stage completion markers.

Acquisition keeps the stronger chunk-level resume semantics from PR #75. Factory processing uses content-addressed completed-stage reuse: an existing `COMPLETE.json` is accepted only when its self-hash, plan identity and payload hash still match. An incomplete stage fails closed and requires explicit recovery instead of silently treating partial output as complete.

## Extraction and policy boundary

Factory v1 executes `jsonl` / `jsonl_text_v1` sources whose exact raw snapshot is already in the approved D03 registry and verified by the D09 retrieval inventory. Source formats such as Wikimedia XML require a future explicit maintained extractor adapter before they are executable; the owner-review candidates in this package are therefore not silently ingestible.

Normalization reuses the existing D03 NFKC/whitespace contract. EN/UK language ID is the current project heuristic and is recorded as compatibility-only, not the final multilingual classifier. Email/phone PII patterns and basic quality thresholds are applied before exact dedup. Source-level copyright/rights authority remains the D03 rights decision; record filtering cannot upgrade source rights.

Reserved benchmark source IDs and exact normalized-content fingerprints are rejected before exact dedup so benchmark material cannot be hidden by later deduplication.

## Maintained scale backends

Production near dedup calls the DataTrove 0.10.0 four-stage MinHash pipeline: signature generation, bucket matching, clustering, then filtering. Production Parquet conversion uses DataTrove `JsonlReader` and `ParquetWriter` with bounded batch size.

The disposable LOCAL_FREE runtime used for this package contains `fsspec 2026.4.0` but does not contain DataTrove, `pyarrow`, or `fastparquet`. Therefore production MinHash and Parquet execution are **NOT EXECUTED** here. The code fails closed on a missing or non-0.10.0 DataTrove runtime instead of reporting false evidence. D08 still owns the hash-locked environment.

## LOCAL_FREE evidence

`tools/benchmark_pretraining_factory.py` executes a controlled synthetic chain. Current retained run:

- 1,202 input documents;
- 41 exact duplicates removed;
- 1 exact reserved-benchmark injection rejected;
- 1 email-PII injection rejected;
- 1,159 exact-unique accepted records;
- 28 bounded fixture near-duplicates removed;
- 1,131 final documents;
- deterministic 1,124 train / 7 validation documents;
- 8 deterministic shard IDs per split plus tokenizer-training JSONL;
- completed exact and final stages reused on repeat execution;
- 2.059292 s end-to-end, 583.7 input records/s;
- 7,670,420 bytes peak Python allocation by `tracemalloc`.

The local near-dedup evidence is deliberately not a production scaling claim. PR #64 already supplies a larger disk-backed exact-dedup mechanics run; this package proves the missing end-to-end composition and handoff.

## Token/corpus planning targets

`configs/data/pretraining_token_targets.json` provides 2 tokens/parameter for mechanics gates, 10 for serious ablations, and 20 for a scratch-baseline planning target. These are engineering budgets, not a universal compute-optimality claim.

For approximately 1M / 10M / 100M / 400M parameters, scratch-baseline clean-token targets are 20M / 200M / 2B / 8B. At an explicit 70% retained-token planning yield, raw candidate budgets are approximately 28.6M / 285.7M / 2.86B / 11.43B tokens. The 70% yield is not observed evidence and must be replaced after a real reviewed corpus slice is measured.

All counts are in the future selected experiment tokenizer, not S0 byte tokens. No future tokenizer is frozen by this package.

## Owner-review source candidates

`configs/data/source_candidate_owner_review_20260825.json` records Wikimedia EN/UK dated text dumps and item-level unrestricted Project Gutenberg material as candidates for owner/legal review only. Every entry is `REVIEW_REQUIRED`, `training_eligible=false`, and `auto_ingest=false`. Nothing is added to the live external-source registry.

## Exact next ingestion action

1. Owner/legal reviewer chooses one exact candidate source version, preferably one bounded EN or UK slice first.
2. Record a durable D03 `RightsDecision` for that exact version. If it is not `APPROVED_FOR_TRAINING` with explicit model-training permission, stop.
3. Register immutable snapshot URI, exact byte size and SHA-256 in `data/external/external_sources.json` without weakening the rights gate.
4. Use PR #75 acquisition to produce a verified retrieval receipt/inventory for that exact version.
5. Run the factory exact stage and retain statistics/rejection samples. Do not run production MinHash/Parquet until D08 supplies a hash-locked DataTrove 0.10.0 + Parquet environment.
6. Then execute DataTrove MinHash and Parquet, deterministic split/shard/tokenizer handoff, and replace the 70% planning yield with measured retained-token yield before increasing corpus volume.

No paid compute or external acquisition is authorized by this package.
