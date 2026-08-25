# DATA-30 Near Deduplication

DATA-30 is a calibrated policy and evidence layer over the existing D03/DATA-12 DataTrove MinHash implementation. It does not add a second deduplication engine.

## Incumbent

- DataTrove: `0.10.0`, exact wheel SHA-256 `c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d`.
- DATA-12 exact-green parent: `3cf64203caec61a8424bb8638b599c208ca758f5` (PR #170).
- Natural-text center policy remains the validated DATA-12 `9` word n-grams, `14` buckets, `8` hashes per bucket, seed `1`, 64-bit hashes, number normalization disabled.
- MinHash is lexical near-duplicate detection. DATA-30 makes no semantic-deduplication claim; translations are explicitly calibration examples outside lexical-MinHash authority.

## Calibration

The labeled set at `data/calibration/near_dedup_v1.json` includes true near copies, boilerplate, translations, code forks, and legitimate similar documents. Only three policies per modality are evaluated. The selection gate requires recall at least `0.75` and labeled preserve-pair false-removal risk at most `0.25` on this bounded calibration set.

Natural candidates keep `n_grams=9` and vary only LSH banding around the incumbent. Code candidates use `n_grams=5` to exercise fork-sensitive lexical overlap, while banding is allowed to become stricter only when calibration evidence requires it. The natural DATA-12 incumbent is retained whenever it passes the gate. For code, incumbent `14x8` banding remains preferred unless the strict `10x10` candidate passes the gate and either the incumbent fails it or the strict candidate demonstrates lower labeled false-removal risk.

The `lsh_similarity_at_50pct_detection` values are operational LSH curve descriptors, not semantic thresholds and not guarantees that a pair will or will not be clustered.

## Cluster provenance and representative selection

DATA-30 calls DataTrove `MinhashDedupCluster` with `save_cluster_id=True` and `save_cluster_size=True`. Input records are sorted by stable record ID and the bounded executor uses one deterministic input shard, allowing DataTrove `(file_id, doc_id)` coordinates to be joined back to source and raw identities. DataTrove chooses one connected-component root as the survivor. DATA-30 records that representative and every discarded alias; it does not substitute a second clustering or representative-selection implementation.

A selected-policy run is repeated in the same workspace to exercise `skip_completed`, and repeated in a fresh workspace to verify stable survivor/cluster identity and deterministic representative selection.

## Current corpus truth boundary

At DATA-30 creation time, the composed repository has no real external UK/EN/code source with explicit model-training eligibility. DATA-21/22/23/24/25 branches still have no committed source/corpus payload beyond the DATA-10 parent. The only committed UK/EN/code input is `data/synthetic/data10/uk-en-code-train.txt`, whose authority is `PROJECT_AUTHORED_SYNTHETIC_ONLY` and whose recipe explicitly says it is not representative corpus evidence.

Therefore the workflow executes the complete DataTrove calibration and a bounded pass over every currently committed DATA-10 mechanics record, but records the real-corpus mission state as `BLOCKED_NO_REAL_TRAINING_ELIGIBLE_UK_EN_CODE_CORPUS`. It does not relabel the mechanics file as real corpus evidence. The same runner is the re-entry point after manifested rights-approved shards arrive.

## Machine evidence

The workflow `.github/workflows/data30-near-dedup.yml` retains `reports/data30/near_dedup_report.json` and `/usr/bin/time -v` output as an Actions artifact. The report contains candidate calibration metrics, selected policies, cluster statistics, document/byte reduction ratios, cluster provenance, restart/determinism checks, a false-positive review sample, the surviving mechanics-corpus identity, and the fail-closed real-corpus gate.
