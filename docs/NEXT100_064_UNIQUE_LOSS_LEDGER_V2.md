# NEXT100-064 — Unique Loss Ledger V2

Worker: `NEXT100-064-UNIQUE-LOSS-LEDGER-V2`

Execution profile: `LOCAL_FREE`. No model training, tokenizer fitting, optimizer update, final-test payload access, or paid compute is performed by this worker.

## Purpose

This successor generalizes DATA-294 from the earlier text-only snapshot to a corpus-independent contract that supports both natural-language text and code. It counts the exact logical causal target positions that actually carry loss after normalization, evaluation reservation exclusion, document boundaries, deduplication, split assignment and packing.

The counting domain is **logical token targets**, never source-byte cardinality. Source bytes and normalized bytes remain provenance/capacity fields only. A byte tokenizer may create a numerical equality between bytes and tokens for a particular object, but V2 still identifies each loss position by `(document identity, target token index)` and its actual packed loss span.

## Required materialization contract

`12-6.postpack-loss-materialization.v2` binds five immutable stage identities:

1. normalization;
2. evaluation reservations;
3. dedup;
4. split;
5. packing.

Every document binds language, modality (`text` or `code`), canonical family, normalized payload SHA-256, token count, split, dedup cluster, reservation state and exact eligible target ranges. Target index zero is never eligible because it has no same-document causal predecessor.

A terminal packing manifest enumerates compact loss spans with both the logical document target interval and the packed target slots. The builder rejects:

- a reserved target;
- a non-training split;
- a dedup loser;
- two retained training documents in one dedup cluster;
- a replayed logical target in any later pack;
- overlapping target slots inside a pack;
- incomplete one-pass packing when terminal mode is requested;
- any materialization whose self-identity or stage identities drift.

Padding and cross-document transitions contribute exactly zero loss positions.

## Deterministic replay and resume binding

`ExposureReplayGuard` authorizes exact subranges of immutable ledger segments and atomically compares their cardinality with the trainer's observed nonignored loss-mask count. Overlap/replay is rejected before the budget moves.

The durable state is self-hashed and binds:

- ledger identity;
- materialization identity;
- packing identity;
- authorized exposure budget;
- consumed exact segment intervals;
- claim sequence;
- checkpoint generation;
- checkpoint manifest SHA-256;
- optimizer step;
- trainer nonignored-target counter.

Resume on a different ledger, corpus materialization, packing trace, checkpoint manifest, budget or trainer target counter fails closed.

## Current corpus result

At the final source/corpus refresh for this worker, DATA-301 remains `TERMINAL_BLOCKED` with no immutable post-split/post-pack corpus identity. NEXT100-065 preserves a source-level dedup cut but explicitly does not turn source capacity into post-reservation/post-split causal-loss capacity. NEXT100-066 independently reports `BLOCKED_NO_EXACT_CANDIDATE_CORPUS_IDENTITY`.

Several later source workers have terminal source-level success, including KMu Secretariat, the bounded Ukrainian Wikisource snapshot and Starlette. They are consumed in the authority vector, but none is promoted to loss positions because no successor corpus materialization has composed all terminal sources through reservations, global dedup, split and packing.

Therefore:

- exact terminal post-pack one-pass maximum: `NOT_MATERIALIZED`;
- training-authorized exposure at this cutoff: exactly `0`;
- no per-language, per-modality or per-family post-pack counts are fabricated.

The evidence also retains a clearly non-authoritative historical DATA-300/DATA-301 prebuild diagnostic: under S0 byte tokenization and source-object document boundaries only, the five-source candidate has 183,056 document-isolated causal token targets (UK 88,564; EN 84,791; Python/code 9,701). This value predates the current evaluation-reservation/decontamination/split/packing gates and **must not** be used as a training budget.

## Unblock condition

A successor may publish nonzero exact exposure only after it receives one terminal immutable materialization that cryptographically binds the complete final terminal-source vector to normalization, purpose-specific evaluation exclusions, global dedup clusters, train split membership and exact packed loss spans. Rebuilding that materialization twice must yield byte-identical bytes and the V2 ledger must rebuild byte-identically from it.
