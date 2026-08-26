# D03 Rada bulk normalization V1 semantic lock

## Finding

The Rada bulk normalizer in the parent D03 branch correctly validates exact archive and per-entry identity, but its configuration validation only requires non-empty `hidden_tags` and `block_tags` collections. The materializer then consumes those collections directly.

That means a coordinated or accidental config edit could change the canonical normalization semantics while still allowing the tool to emit `PASS_NORMALIZATION_ONLY`. For example, removing `script` from `hidden_tags` would make script bodies eligible for normalized visible text without changing the schema name.

This is a reproducibility and provenance defect. A normalized-record identity is only meaningful if the transformation policy that produced it is itself exact and immutable.

## Repair

`tools/validate_d03_rada_bulk_normalization_policy_lock.py` defines the exact canonical V1 semantic projection and fails closed on drift in:

- decoder and Unicode normalization policy;
- inline whitespace and blank-line behavior;
- exact hidden-tag and block-tag sets and order;
- record-ID prefix;
- output record fields and deterministic output contract;
- required downstream gate sequence;
- zero-training / zero-promotion truth boundary and safe-result label.

A successful validation returns a SHA-256 over the canonical semantic projection. This identity can be bound by later materialization/corpus evidence without treating incidental JSON formatting as semantic state.

Adversarial tests explicitly cover removal of `script`, block-tag drift, decoder/NFKC drift, record-ID drift, output-hash-field removal, downstream decontamination removal, training authorization mutation, and safe-result promotion.

## Scope boundary

This package is additive audit hardening over the current D03 Rada bulk normalization branch. It does not change source bytes, normalized payloads, rights decisions, capacity accounting, tokenizer state, model weights, optimizer state, or checkpoint state.

It authorizes zero training bytes and no paid compute.

The validator protects the committed canonical V1 configuration and provides a stable policy identity. Full runtime enforcement for callers supplying an arbitrary alternate `--config` remains a parent-integration decision: the materializer should consume this validator before treating any configuration as canonical evidence, or should explicitly mark noncanonical configs as experimental rather than canonical V1.

## Promotion rule

Do not treat this audit package or its parent normalization package as Research Corpus V1 admission. Quality/privacy, global cross-source dedup, evaluation decontamination, balance/family caps, deterministic split/shard/packing, post-pack unique causal-loss accounting, tokenizer authorization, checkpoint integrity, and explicit learned-20M compute authorization remain separate required gates.
