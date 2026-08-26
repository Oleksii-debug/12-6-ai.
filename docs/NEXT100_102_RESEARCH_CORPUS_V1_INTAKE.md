# NEXT100-102 Research Corpus V1 pre-decontamination intake

## Decision

This worker creates the missing successor intake boundary requested by the live ~20M readiness controller. It freezes an exact pre-decontamination record-identity set, but it does **not** claim a terminal Research Corpus V1, shard identity, nonzero optimized-loss capacity, or authorization for long training.

The parent DATA-301 attempt is preserved as terminal-blocked at `8820ba1b255f6bb95c7db0531fd846078a1aae01`. Its five frozen records are inherited by exact content identity. Two terminal source authorities are added only as successor intake candidates:

- UA Wikisource bounded Lesia Ukrainka 1892 page snapshot: one exact record, one independent Ukrainian family, training permission allowed but training selection remains blocked until near-match evaluation decontamination.
- CPython official tutorial documentation: one independent English family represented by the exact 14 accepted normalized chunk hashes; the two privacy-rejected chunks remain excluded.

The resulting structural family vector is UK=2 / EN=2 / code=2. This closes only the old *minimum family-count impossibility*. It does not prove the DATA-295 family-share caps, 45/35/20 mixture capacity, corpus quality, privacy, global deduplication, evaluation decontamination, cluster-safe splits, packing, or unique optimized loss positions.

## Exact identity boundary

`configs/data/next100_102_research_corpus_v1_intake.json` binds:

- live controller PR #519 exact head and evidence identity;
- DATA-301 and DATA-300 exact parent heads;
- EVAL-303 nonempty immutable selection authority without reading final-test outcomes;
- UA Wikisource authority exact head, config blob and authority identity;
- CPython documentation authority exact head, config blob and authority identity;
- all five inherited parent content identities;
- the exact UA snapshot SHA-256;
- all 14 accepted CPython normalized chunk SHA-256 identities.

The candidate inventory has its own canonical SHA-256 identity. The full authority is also self-hashed.

## Why training capacity remains zero

The CPython source authority proves the 14 accepted chunk hashes but does not place those accepted bytes and per-record byte lengths into one immutable successor candidate tree. Therefore this worker refuses to infer exact corpus bytes or optimized causal-loss positions from the whole-source 17,901-byte normalization figure.

Before decontamination, a successor materializer must reproduce the exact accepted CPython chunk hashes, bind the exact UA snapshot bytes and the five DATA-300 parent records, and emit one immutable per-record byte manifest. Only then can exact/near-match decontamination and the remaining corpus gates run.

## Required next sequence

1. Materialize all exact candidate record bytes and byte lengths from the bound immutable authorities.
2. Run exact/normalized/near-copy/fragment evaluation decontamination.
3. Run composite quality, privacy, global deduplication and lineage clustering on the exact materialization.
4. Build cluster-safe splits and enforce 45/35/20 plus family caps without replay or duplication.
5. Build twice from independent clean roots and require byte-identical trees/shards.
6. Build the post-pack unique causal-loss ledger.
7. Requalify 20M checkpoint recovery and bounded optimizer mechanics.
8. Refresh the 20M campaign preregistration.
9. Request paid/material compute authorization only after all data gates are terminal.

`LOCAL_FREE` only. No model training, optimizer update, external LLM, GPU/cloud run, final-test outcome access, or paid compute is performed by this worker.
