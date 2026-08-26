# EVAL-131 first-party real-corpus core

## Authority

This package establishes the canonical first-party held-out quality contract for the learned 100K-to-1M ladder. It extends D06 evaluation semantics without replacing D06, and it consumes the current DATA-31 decontamination identity rather than introducing a second contamination engine.

First-party held-out quality is the authority for this task. No external benchmark is required. Training loss is not a quality metric and is not emitted into the scaling-dashboard quality rows.

## Immutable held-out contract

A canonical held-out bundle contains all three required modalities: `ua`, `en`, and `code`. Every admitted row must declare `source_kind=EXTERNAL_REAL`, a pinned source version and source snapshot SHA-256, a source family, provenance, and an explicit evaluation-use authority reference. Project-authored DATA-25 text is therefore useful for mechanics tests but cannot be promoted into this real-corpus authority.

The builder writes one canonical JSONL file per modality plus a self-hashed manifest. Existing bytes are immutable: rebuilding the same identity is idempotent, while any changed inventory or byte is rejected. The manifest includes source-family totals and a reserved registry of record IDs and exact content SHA-256 values.

Before a learned tokenizer is fitted, its candidate fit corpus must produce a zero-overlap exclusion proof against the held-out registry. A fixed tokenizer such as `s0-byte-v1` records an explicit no-fit proof instead. Model training always requires a separate zero-overlap proof before use. These proofs bind the exact held-out identity and candidate identity.

DATA-31 remains the near-match/decontamination authority. EVAL-131 only binds the exact DATA-31 decontamination report SHA-256, D06 registry SHA-256, and frozen reference-bundle SHA-256 into the held-out/evaluation identities. It does not reimplement DataTrove MinHash.

## Evaluation semantics

`evaluate_checkpoint` requires a checkpoint descriptor, live model, live Trainer, tokenizer, immutable held-out manifest and rows, and exact training/split/decontamination/exclusion identities.

The scorer executes all forward passes inside `torch.no_grad()`. It never calls `model.eval()`, `model.train()`, a Trainer training method, optimizer, scheduler, or scaler. Model parameters/buffers plus every module training flag are hashed before and after evaluation. The full checkpoint-safe Trainer state plus `micro_step`, `optimizer_step`, and `tokens_seen` are also hashed before and after. Any mutation is a hard error. The descriptor's `optimized_tokens` must equal the live Trainer `tokens_seen` before scoring and the delta after scoring must be zero.

Documents are isolated. Long documents use contiguous windows with one-token overlap so each next-token target is scored once and no context crosses documents. The first token of each document is unscored because the canonical Base path has no BOS token. BPB is total scored token NLL in bits divided by exact frozen UTF-8 source bytes; this first-token boundary is recorded explicitly in every report.

Metrics are reported for the complete held-out bundle, each modality, and each source family:

- natural-log cross-entropy per scored token;
- perplexity only when the cross-entropy is finite, non-negative and representable, and only as meaningful under the exact tokenizer identity;
- bits per frozen source byte;
- tokenizer tokens per frozen source byte.

Confidence summaries use a deterministic document bootstrap. SplitMix64 sampling and the percentile interpolation rule are fixed by the implementation. The seed is derived from held-out identity, checkpoint identity, and group name. A one-document group gets a deterministic degenerate interval rather than a fabricated uncertainty estimate.

## Random-init comparison and dashboard report

Every learned ModelSpec in a ladder report must have exactly one same-geometry random-init baseline. Cross-architecture 100K-to-1M rows may coexist, but learned-vs-random improvement is computed only within the same exact ModelSpec identity. This prevents a random baseline from one geometry being used to inflate another geometry's result.

The ladder report is self-hashed and contains flat `12-6.scaling-dashboard-heldout-quality-row.v1` rows for aggregate, modality and source-family views. Each row binds checkpoint, ModelSpec, parameter count, optimized-token count, tokenizer, training corpus, training split and held-out identities. Train-loss fields are rejected by the report verifier except for explicit policy flags whose value is `false`.

## Current fail-closed readiness, 2026-08-26

The evaluator contract is executable, but the repository does not yet possess a canonical three-modality real-corpus bundle that can be used to publish quality numbers.

DATA-21/22 head `dcc7dfc39299487bca5bdbfe5e6c70eaa6706278` proves bounded real UA/EN intake candidates, including approved Rada and Standard Ebooks sources, but its own truth boundary says those bytes are sample evidence rather than canonical immutable source snapshots/corpus promotion.

DATA-23 head `5f223f9ef77762a042e966372fdf9f064b3cc9fe` retains three real Python files / 4,998 bytes for mechanical evidence, but all three are blocked by the live D03 rights authority and zero code bytes are training-eligible. EVAL-131 does not reinterpret that state as evaluation authorization.

DATA-25 corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8` remains project-authored. It may validate the evaluator mechanically, but it is not real-corpus quality evidence.

Therefore the current readiness state is `BLOCKED_REAL_CORPUS_INPUTS`. A legitimate first real report requires canonical immutable UA/EN/code source snapshots with evaluation-use authority, the held-out bundle frozen before tokenizer fitting/training, a fresh DATA-31 pass for the exact training candidate, exclusion proofs, and random-init plus learned checkpoint evaluations under one exact identity envelope.

## Local validation

The focused implementation tests are designed to run with local CPU only:

```text
PYTHONPATH=src python -m compileall -q src/twelve_six tools
PYTHONPATH=src python -m pytest -q tests/test_real_corpus_holdout.py tests/test_real_corpus_evaluation.py
```

No paid compute is required or authorized by this package.
