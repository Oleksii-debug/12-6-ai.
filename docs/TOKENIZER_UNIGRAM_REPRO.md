# Unigram reproducibility decision

Status: experimental tokenizer research / canonical Unigram rejected for the pinned runtime.

This note covers Hugging Face `tokenizers==0.23.1` only. It does not change canonical
`s0-byte-v1`, choose BPE for a later stage, or claim representative-corpus model quality.

## Locked runtime and incumbent

The experiment runs inside the existing D04 tokenizer workflow on Ubuntu 24.04 with
CPython 3.11.16 and the hash-admitted `tokenizers==0.23.1` wheel. Training consumes the
same manifest-bound D03 train split already used by the real BPE/Unigram comparison;
validation remains held out.

The new `twelve_six.tokenization.unigram_repro` diagnostic calls the existing
`train_hf_tokenizer` path. It does not fork, patch, or reimplement Unigram training.

## Upstream source finding

The exact `tokenizers` v0.23.1 Unigram trainer source is
`tokenizers/src/models/unigram/trainer.rs` (blob
`ff5ca9428ab7c7ca9b96065046f32b42246dc234`). Its training state uses `AHashMap` and
`AHashSet`. `feed()` aggregates pre-tokenized strings into an `AHashMap`, and `train()`
constructs the training sentence vector by iterating that map. The crate declares
`ahash==0.8.11` semantics through its dependency and does not expose a public
`UnigramTrainer` random-seed field.

Consequences:

1. Fixing the caller's document order is necessary for experiment identity but is not
   sufficient for Unigram training identity because the backend aggregates into a
   randomized hash map before training.
2. `PYTHONHASHSEED` cannot control Rust `AHashMap` state.
3. `TOKENIZERS_PARALLELISM=false` plus `RAYON_NUM_THREADS=1` removes parallel reduction as
   a sufficient explanation, but the randomized hash-container ordering remains.
4. The trainer also contains parallel floating-point reductions in E-step/pruning paths,
   so parallel reduction ordering can be an additional source when multithreading is used.
5. The supported Python `UnigramTrainer` interface has no seed argument in this pinned
   release. The project therefore cannot claim a deterministic seed that the maintained
   backend does not expose.

## Executable reproduction

The diagnostic runs separate fresh processes so process-local hash state cannot be hidden
by repeated calls in one Python process. It records:

- three manifest-order serial runs with `PYTHONHASHSEED=0`,
  `TOKENIZERS_PARALLELISM=false`, and `RAYON_NUM_THREADS=1`;
- two reversed-input serial runs under the same controls;
- two manifest-order parallel runs;
- exact tokenizer JSON, ordered token-ID vocabulary, ordered Unigram model vocabulary,
  token and score fingerprints;
- a direct `UnigramTrainer(seed=0)` capability probe;
- repeated `to_str()` and serialize-load-serialize checks to distinguish training drift
  from serialization instability;
- held-out token sequences, fertility, strict UTF-8 round trip, and unknown-token counts.

The report is self-hashed and source-SHA bound. It fails closed if the serial regime no
longer reproduces the known drift, if serialization itself becomes unstable, or if a
supported seed suddenly appears. Such a runtime change requires a new investigation rather
than silently inheriting this decision.

## Semantic-equivalence identity is rejected

The previously retained real comparison already showed that repeated Unigram builds do
not merely serialize the same tokenizer differently: ordered vocabulary identity changes
and held-out token sequences change. Strict text round trip and zero unknown tokens remain
necessary checks, but they are not sufficient for checkpoint identity.

A language-model checkpoint binds embedding and LM-head rows to exact token IDs. If two
Unigram builds assign different IDs or segment the same held-out text into different token
sequences, treating them as one semantic tokenizer identity can reinterpret checkpoint
parameters. Finite probe agreement would not prove global equivalence either. Therefore a
weaker semantic-equivalence identity is not admitted.

## Decision

For Hugging Face `tokenizers==0.23.1`, Unigram remains `FAIL` for canonical use under this
project's exact-artifact identity requirement. The machine decision is
`REJECT_UNIGRAM_CANONICAL_TOKENIZERS_0_23_1`, and semantic-equivalence identity is disabled.

Unigram may be reconsidered only under a separately pinned maintained backend/version that
provides a supported deterministic training contract and then proves repeated byte-identical
artifacts, exact ordered vocabulary identity, held-out encoding identity, strict round trip,
and zero unintended unknowns on the intended corpus. No custom Unigram trainer is warranted
for this decision.
