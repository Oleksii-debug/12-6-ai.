# D05 checkpoint-v1 closed-world manifest contract

Checkpoint-v1 integrity is more than a valid `MANIFEST.sha256`. A checksum-consistent manifest can still be semantically unsafe if an older reader silently ignores new fields, accepts malformed identity values, or accepts a serialization declaration that no longer matches the actual v1 contract.

This package makes checkpoint-v1 a closed-world schema at the canonical `prepare_checkpoint_load()` boundary. New semantics must use a new checkpoint format version rather than being inserted into v1 and silently ignored.

## Exact v1 contract

The top-level manifest has exactly seven fields: `format`, `format_version`, `created_at_utc`, `checkpoint_id`, `identity`, `files`, and `serialization`.

The identity object has the exact fields emitted by `CheckpointIdentity` plus the derived hashes and environment snapshot. Required identity values are revalidated with the same type/value rules used at save time: exact Git/hash identities, positive parameter count, non-negative seed/step/tokens, non-empty model/training/optimizer mappings, valid scheduler shape, non-empty precision, and optional exact environment-lock hash. The environment snapshot must remain a non-empty mapping and all derived hashes must recompute exactly.

The payload inventory remains exactly `weights.safetensors`, `state.safetensors`, and `state.json`. Each file record contains exactly `sha256` and `bytes`; unknown per-file semantics are rejected.

The serialization declaration is exact: SafeTensors model/state tensors, canonical JSON state tree, and `pickle=false`. `pickle=true`, numeric lookalikes, or undeclared serialization fields are rejected.

`created_at_utc` must be an ISO-8601 UTC `Z` timestamp. `format_version` must be an integer, not a JSON boolean that compares equal to integer `1` in Python.

## Failure boundary

Schema validation happens after the manifest bytes and `MANIFEST.sha256` are checked but before checkpoint payload decode or any model/trainer/RNG mutation. Regression tests deliberately recompute `MANIFEST.sha256` and, where identity/file records change, recompute `checkpoint_id`. These are therefore semantic-schema adversarial cases rather than ordinary corrupted-checksum cases.

This contract does not provide cryptographic authorship/authenticity; SHA-256 checks prove internal content identity, not who produced the bytes. Release/audit authority remains external. It also does not change checkpoint format version, introduce pickle, modify Base behavior, authorize paid compute, or claim CANDIDATE/STABLE status.
