# D05 -> D07 verified snapshot binding

Status: experimental S0 engineering contract. This document does not grant audit or promotion authority.

## Problem closed

D05 checkpoint-v1 now exposes `prepare_checkpoint_load()` and `load_verified_checkpoint()` so every manifest and payload byte can be verified once and consumed from one immutable in-memory snapshot. The D07 first-party adapter still used the older two-operation pattern: `verify_checkpoint(path)` to derive ModelSpec/tokenizer/diagnostics, followed by `load_checkpoint(path)` to reopen the directory and load weights.

That left a cross-layer check/use gap. A compatible checkpoint directory could change between those operations, allowing the backend's reported checkpoint identity to come from one verified manifest while model weights came from a later compatible checkpoint. D05 core itself was snapshot-safe; the D07 adapter was not consuming that guarantee end-to-end.

## Contract

`load_first_party_backend()` now:

1. calls `prepare_checkpoint_load()` exactly once;
2. derives ModelSpec, tokenizer identities, context and diagnostics from that verified snapshot manifest;
3. constructs the canonical D01 model without loading foreign weights;
4. calls `load_verified_checkpoint()` with the same `VerifiedCheckpoint` object and full git/model/tokenizer/vocab/dataset/run identity constraints;
5. refuses any manifest drift between preflight and load result;
6. keeps an internal defensive manifest copy and returns only defensive copies through `backend.manifest`;
7. does not restore training RNG for inference.

No model architecture, tokenizer semantics, checkpoint format, training behavior, raw Base prompt semantics or server protocol is changed.

## Regression evidence

`tests/test_first_party_snapshot_binding.py` exercises two failure boundaries:

- after the D05 verified snapshot is created, the entire source checkpoint directory is deleted before model load; first-party inference must still load exactly the snapshotted weights and report the snapshotted checkpoint identity, proving no checkpoint byte is reopened;
- mutating a caller-visible `backend.manifest` copy must not alter diagnostics or the internal verified identity.

Exact-head GitHub Actions is the authority for this branch. Queued or in-progress runs are not PASS.

## Truth boundary

This package is LOCAL_FREE/CI-only and remains EXPERIMENTAL. It does not claim AUDIT-A/AUDIT-B PASS, CANDIDATE/STABLE promotion, Windows runtime support, Transformers/vLLM/GGUF/llama.cpp parity, paid compute authorization, or any instruction/alignment/refusal/personality/domain-specialization behavior. Canonical Base remains random-initialized and pretraining-only.
