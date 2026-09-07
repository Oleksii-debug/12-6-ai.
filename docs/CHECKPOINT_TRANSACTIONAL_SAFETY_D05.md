# D05 checkpoint publish/load transactional safety

Status: **EXPERIMENTAL / LOCAL_FREE**. This document strengthens checkpoint-v1 mechanics; it does not promote an S0 candidate or override AUDIT-A/AUDIT-B.

## Problem closed by this package

The earlier checkpoint-v1 implementation verified a staging checkpoint, then implemented `overwrite=True` by deleting the existing destination before renaming the new directory. A process/filesystem failure in that gap could destroy a previously valid checkpoint. Loading also verified files by pathname and then reopened those pathnames for SafeTensors/state decoding, leaving a check/use window. Finally, state decoding occurred after model weights had already been applied, so a checksum-consistent but structurally invalid trainer/RNG payload could fail after model mutation.

## Publish contract

Checkpoint-v1 directories are immutable after publication. `save_checkpoint()` still accepts the historical `overwrite` argument for source compatibility, but an existing destination fails closed even when `overwrite=True`. A checkpoint is fully written and verified in a sibling staging directory and only then published with a directory rename. The implementation never deletes an existing valid destination checkpoint.

A checkpoint directory has an exact five-entry inventory:

- `manifest.json`
- `MANIFEST.sha256`
- `weights.safetensors`
- `state.safetensors`
- `state.json`

Sidecars, notes, reports, and evidence belong outside the checkpoint directory. Untracked entries are rejected rather than silently escaping the checkpoint identity.

## Verify/load contract

`prepare_checkpoint_load()` establishes one verified in-memory byte snapshot. It rejects a symlinked checkpoint root, symlink/non-regular payload files, unexpected directory entries, invalid manifest identity hashes, invalid payload hashes/sizes, and checkpoint-id mismatch. Files are opened with `O_NOFOLLOW` where the platform exposes it and are checked with `lstat`/`fstat` identity before the bytes are accepted.

`load_verified_checkpoint()` consumes only that snapshot; it does not reopen checkpoint paths. Before the first model mutation it performs:

1. requested identity checks;
2. weights SafeTensors decode;
3. trainer/RNG SafeTensors decode;
4. state-tree JSON decode and reconstruction;
5. model key/shape/dtype materialization checks;
6. requested optimizer/scheduler-state presence checks;
7. supported Python/NumPy/Torch RNG compatibility preflight.

The D02 trainer adapter now performs canonical nested run-binding checks and the model restore from the same verified snapshot instead of verifying one directory state and then reopening a later one.

## Explicit truth boundary

The guarantee is about checkpoint artifacts and D05-owned preflight: integrity, identity, payload decoding, model compatibility, and supported RNG compatibility failures are detected before D05 applies model state. Arbitrary user-provided `model.load_state_dict()`, optimizer/scheduler `load_state_dict()`, or D02 trainer `load_state_dict()` implementations can themselves mutate and then raise; D05 cannot transactionally roll back unknown third-party object semantics and does not claim that guarantee.

The verified snapshot retains serialized checkpoint payload bytes in memory while decoded arrays are prepared. That is appropriate for current S0 LOCAL_FREE checkpoints and closes pathname TOCTOU precisely. It is **not** a claim that this design is the final S8+/distributed checkpoint mechanism; large sharded/distributed checkpointing should use a scale-appropriate immutable object/shard protocol under D12/D08 contracts rather than copying an entire giant checkpoint into one process.

No pickle is introduced. No foreign pretrained weights, behavioral alignment, instruction tuning, refusal/personality layer, paid GPU/cloud run, CANDIDATE, or STABLE claim is introduced by this package.

## Regression evidence

`tests/test_checkpoint_transactional_safety.py` covers:

- existing checkpoint preservation when `overwrite=True` is attempted;
- unexpected/untracked directory entries;
- symlinked payload rejection;
- source tamper after a verified snapshot, proving the loaded bytes are the verified bytes;
- checksum-consistent malformed state-tree rejection before model mutation;
- checksum-consistent malformed SafeTensors rejection before model mutation;
- invalid supported RNG state rejection before model mutation.

These tests complement, rather than replace, PR #61's exact S0 interrupted/resume equivalence, canonical identity/tamper matrix, and HF-style export parity hooks.
