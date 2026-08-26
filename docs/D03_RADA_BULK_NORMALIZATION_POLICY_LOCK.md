# D03 Rada bulk normalization semantic lock

## Finding

The current Rada bulk normalizer already binds the executable decode policy, NFKC behavior, whitespace behavior, record-ID prefix, parent probe identity, archive identity, and zero-training boundary. It still consumes `hidden_tags` and `block_tags` from configuration without requiring their exact canonical V1 values, and it does not bind the exact normalization name, declared output-field vector, downstream-gate vector, or `safe_result` label.

That leaves a provenance seam: a configuration mutation could change the text transformation or weaken the declared successor contract while retaining the same V1 schema and reaching `PASS_NORMALIZATION_ONLY`.

A concrete example is removing `script` from `hidden_tags`, which changes the normalized text payload while the existing non-empty-list check still succeeds.

## Hardening

`tools/validate_d03_rada_bulk_normalization_policy_lock.py` defines the exact current V1 semantic projection and fails closed on drift in:

- normalization name;
- UTF-8 -> Windows-1251 fallback identity;
- NFKC and whitespace behavior;
- exact hidden-tag and block-tag vectors;
- record-ID prefix;
- exact output record fields, including `source_encoding` and content hashes;
- deterministic output contract;
- exact downstream gate sequence;
- zero-training / zero-promotion claim boundary and safe-result label.

The validator emits a stable SHA-256 over semantic state rather than incidental JSON formatting. Adversarial tests mutate the transformation and truth-boundary fields and require rejection.

## Boundary

This is additive `LOCAL_FREE` audit hardening over PR #641. It does not acquire source bytes, credit normalized bytes, fit a tokenizer, update model weights, alter checkpoints, use paid compute, or claim Research Corpus V1 / learned-20M readiness.

A later integration step should make this semantic identity part of canonical materialization/corpus evidence so normalized records are bound not only to source bytes but also to the exact transformation that produced them.
