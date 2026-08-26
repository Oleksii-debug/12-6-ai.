# TOK-323 BPE vs Byte Final

Worker: `TOK-323-BPE-VS-BYTE-FINAL`

Execution profile: `LOCAL_FREE`

## Terminal result at this authority cutoff

`BLOCKED_NO_TERMINAL_FROZEN_RESEARCH_CORPUS_V1`

No byte/BPE model arm was trained and no tokenizer-family recommendation is scientifically valid at this cutoff. `PRACTICAL_TIE` is **not** emitted: a tie requires paired measurements, not missing authorized measurements.

## Why execution is blocked

TOK-316 already froze the maintained reproducible BPE path but reported `BLOCKED_TOKENIZER_FIT_NOT_YET_PERMITTED`; no candidate in the 320/384/437/512 grid was trained there.

The newer DATA-305 authority does not unblock that state. At DATA-305 head `9300e96aab2f17deb086a2329c8062c4a1bea1fe`, the decontamination evidence is `BLOCKED_NO_EXACT_CORPUS_IDENTITY`: `training_corpus_identity` and `shard_identity` are null, raw/normalized/fragment/near/mirror/code-copy matchers were not executed, and no `PASS_CLEAN` or `PASS_WITH_EXCLUSIONS` verdict exists.

EVAL-303 supplies a useful exact-distinctness proof for a 10-document selection set against the DATA-300 training plan: exact-byte overlap is zero and pinned Git-object overlap is zero. However its own proof explicitly does **not** claim a near-copy/dedup-cluster scan and says Wave-3 G07/G08 remain required. Therefore it cannot substitute for the missing exact materialized corpus/decontamination authority.

The experiment cannot be repaired by relabelling DATA-300 source inventory, DATA-229 snapshots, old DATA-25/DATA-183 corpora, final-test bytes, or locally reconstructed text as Research Corpus V1.

## Frozen paired comparison contract

The comparison remains preregistered as follows:

- canonical byte tokenizer: vocabulary 256;
- reproducible BPE candidates: 320, 384, 437, 512;
- maintained BPE implementation: `src/twelve_six/tokenization/experiments.py::train_hf_tokenizer`;
- runtime: `tokenizers==0.23.1`;
- each BPE vocabulary must reproduce byte-identically across two independent fits before model comparison;
- total trainable parameters per arm: 467,808;
- tokenizer-dependent embedding growth must be offset in non-embedding capacity rather than granting BPE more total parameters;
- optimized loss-token budget per model arm: 16,384;
- paired model seeds: 1337, 7331, 18701;
- immutable selection-validation only may rank candidates;
- final-test may not fit, tune, select, rank, or choose checkpoints.

## Required reported metrics after unblocking

Primary: paired aggregate selection-validation BPB.

Required strata: UA BPB, EN BPB, code BPB, and worst-modality BPB.

Secondary operational evidence: UA/EN/code fertility, worst-modality fertility, tokenizer throughput, end-to-end training throughput, and embedding parameter tax relative to the canonical 256-byte vocabulary.

Embedding tax for extra vocabulary rows remains a reported cost, not a free parameter increase. For model width `d_model`, tied embeddings add `(V-256)*d_model`; untied embeddings add `2*(V-256)*d_model` before matched-capacity rebalancing.

## Current numerical matrix

| Arm | Reproducible tokenizer artifact | Seeds | Aggregate BPB | UA | EN | code | Worst modality | Fertility | Training throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Byte-256 | canonical mechanics available | 0/3 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED_ON_V1 | NOT_MEASURED |
| BPE-320 | NOT_FIT | 0/3 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED_ON_V1 | NOT_MEASURED |
| BPE-384 | NOT_FIT | 0/3 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED_ON_V1 | NOT_MEASURED |
| BPE-437 | NOT_FIT | 0/3 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED_ON_V1 | NOT_MEASURED |
| BPE-512 | NOT_FIT | 0/3 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED_ON_V1 | NOT_MEASURED |

Training started: **false**. Optimizer updates: **0**. Final-test outcomes read: **false**. Selection-validation outcomes read: **false**.

## Exact unblock gate

Numerical execution may begin only when all of the following are true simultaneously:

1. a non-null terminal frozen Research Corpus V1 identity exists;
2. deterministic final training shard identity and exact record inventory are bound to it;
3. DATA-305 or a superseding decontamination run emits `PASS_CLEAN` or `PASS_WITH_EXCLUSIONS` on that exact inventory;
4. selection-validation has full exact/near/cluster-safe exclusion authority, not only exact-object distinctness;
5. TOK-315 or a superseding tokenizer-fit authority explicitly permits fitting on the exact materialized train bytes;
6. the maintained `tokenizers==0.23.1` BPE path is available locally and the four candidates reproduce across two independent fits.

Only after that gate may the paired byte/BPE arms run and emit exactly one of `BYTE`, `BPE`, or `PRACTICAL_TIE`.

## Current family decision

`NO_VALID_FAMILY_RECOMMENDATION`

This is intentionally different from `PRACTICAL_TIE`. Promoting byte by default, promoting BPE from historical experiments, or calling the missing experiment a tie would all violate the requested frozen-corpus/paired-selection contract.

Evidence: `evidence/tok323/authority-gate.json`

Evidence identity: `8705fb217b7bb685149c1414373d473bf0e599cf0f5435af0eb3a5a62ec1855d`
