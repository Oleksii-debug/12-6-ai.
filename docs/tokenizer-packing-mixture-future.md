# Tokenizer / packing / mixture foundation after S0

Status: experimental planning and regression infrastructure. This document does not freeze an S1+ tokenizer and does not change canonical `s0-byte-v1`.

## S0 baseline preserved

Canonical S0 remains raw UTF-8 bytes, IDs `0..255`, vocab `256`, no semantic special tokens and no tokenizer-side normalization. Exact identities:

- config SHA-256: `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`
- ordered vocabulary SHA-256: `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`
- packing config SHA-256: `23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285`

Documents remain isolated by default because S0 has no semantic EOS. One-token window overlap preserves every within-document adjacent causal pair exactly once. Masked fill positions do not count as optimized targets. D02 raw labels are shifted once inside `causal_lm_loss`; D04 aligned `target_ids + loss_mask` use `causal_pair_loss` with no second shift.

## Controlled measurements

Against exact D03 dataset identity `bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89`, all 12 texts round-trip exactly under the byte tokenizer and there is no OOV condition.

| Slice | Documents | Unicode code points | Byte tokens | Tokens / code point |
| --- | ---: | ---: | ---: | ---: |
| English (`en`) | 6 | 811 | 811 | 1.0000 |
| Ukrainian (`uk`) | 6 | 811 | 1,515 | 1.8681 |

The Ukrainian fertility penalty is expected from UTF-8 byte encoding and is a concrete target for learned subword experiments; it is not by itself a model-quality metric. Code and mixed Unicode probes remain strict round-trip gates. Machine-readable values live in `configs/tokenizers/s0_controlled_metrics_v1.json`.

## Future BPE / Unigram harness

`TokenizerTrainingManifest` is content-addressed and non-promoting. It binds algorithm, exact maintained Hugging Face `tokenizers` version, D03 dataset-manifest hash, exact corpus file hashes/byte counts, vocabulary size, algorithm parameters, normalization/pre-tokenizer/decoder policy and deterministic input ordering policy.

A trained experimental artifact binds the complete serialized tokenizer JSON SHA-256 and a separate complete ordered token-ID vocabulary SHA-256. Same vocabulary size or same token strings with permuted IDs is identity drift and must fail compatibility.

The maintained library is imported lazily. D08 currently owns canonical dependency locks, so this vertical deliberately does not modify `pyproject.toml` or lock files. Actual BPE/Unigram training remains `NOT_RUN` until an exact D08-owned experiment runtime is admitted. This keeps S0 reproducible and avoids a dependency-surface collision.

## Recommendation matrix — not a freeze

| Candidate | Coverage | Fertility expectation | Code | Complexity | Recommendation |
| --- | --- | --- | --- | --- | --- |
| S0 raw UTF-8 bytes | Exact UTF-8 coverage | Weak on multi-byte scripts; UK measured 1.868 | Predictable/lossless | Minimal | Keep as S0 canonical and permanent reference baseline |
| ByteLevel BPE / HF Tokenizers | Byte-backed | Expected material reduction; must measure EN/UK separately | Usually strong, but test | Moderate | Primary S1-S4 experiment candidate; not selected |
| ByteLevel Unigram / HF Tokenizers | Byte-backed | Potentially competitive multilingual segmentation | Must test | Moderate | Run head-to-head with BPE; not selected |
| SentencePiece alternative | Mature ecosystem | Unknown until same corpus/probes | Requires separate normalization/byte-fallback audit | Additional ecosystem | Research fallback only if HF comparison shows a need |

Do not choose a tokenizer solely on fertility. Compare strict round-trip, unknowns, code, Unicode edges, vocabulary parameter cost, packing utilization, model loss at controlled compute and deterministic rebuild identity. Tokenizer training data must remain train-only and must not absorb benchmark/test content.

## Vocabulary parameter cost

With tied embeddings, vocabulary parameters are `vocab_size * d_model`; an untied LM head doubles that surface.

- current S0: `256 * 20 = 5,120` vocabulary parameters;
- current D01 S4 engineering candidate (explicitly **not frozen**): `32,768 * 768 = 25,165,824` tied vocabulary parameters, about `25.07%` of its `100,384,512` total;
- same S4 geometry with an untied head: `50,331,648` vocabulary parameters.

Therefore future vocabulary size is an architecture-budget decision and must be reconciled with D01 before any stage freeze.

## S1-S4 deterministic mixture / sharding / restart contract

`MixturePlan` replaces floating weight normalization with positive weight units and binds tokenizer config/vocabulary hashes, packing config hash, exact source-manifest hashes, weights, seed, sharding topology and algorithm versions into one SHA-256 identity. Source choice uses SHA-256-derived bounded integer selection, independent of global PRNG consumption.

`shard_for_record()` hashes immutable record IDs rather than list positions, preventing physical input reordering from silently changing shard ownership.

`RestartCursor` binds the exact plan SHA-256 plus next global sample index, per-source consumed offsets, emitted sequence count and emitted loss-token count. Resume under changed tokenizer vocabulary, packing config, source manifest, weights, seed or shard topology fails because plan identity changes. Advancing with the wrong deterministic source also fails.

This is contract evidence, not a distributed execution claim. Multi-worker DataLoader/Parquet/object-store/DataTrove restarts remain `NOT_TESTED`.

## Freeze gate for the next tokenizer

Do not freeze BPE or Unigram until one exact candidate lineage has all of the following:

1. accepted train-only corpus provenance/rights and contamination policy;
2. exact maintained tokenizer runtime in D08 hash locks;
3. repeated training identity evidence for tokenizer JSON and ordered vocabulary;
4. EN/UK/code/Unicode held-out fertility, zero-unknown and strict round-trip evidence;
5. token-ID permutation regression;
6. D01 vocabulary/model parameter-budget reconciliation;
7. D05/C01 binding of tokenizer config and ordered vocabulary hashes;
8. D04/D02 packing, masked-token accounting and single-shift/aligned-target equivalence regression;
9. restart parity under the same `MixturePlan` and fail-closed rejection after plan drift;
10. normal D10 integration plus independent audit on the exact candidate.

No item here grants CANDIDATE/STABLE promotion by itself.
