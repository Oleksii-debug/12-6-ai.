# First-party S0 inference: atomic checkpoint snapshot binding

## Scope

This note documents the D05/D07 boundary used by the canonical raw-Base inference loader. It does not change model architecture, tokenizer semantics, generation policy, sampling, prompts, or serving behavior.

The first-party loader must derive all of the following from one verified immutable checkpoint snapshot:

- checkpoint identity and file inventory;
- ModelSpec and maximum context length;
- tokenizer config/vocabulary/version identities;
- model weights applied to the D07 backend;
- checkpoint/run/dataset/step/tokens diagnostics reported by CLI/server consumers.

## Why one snapshot is required

D05 checkpoint-v1 now provides `prepare_checkpoint_load()` and `load_verified_checkpoint()` so callers can verify a byte snapshot once and consume those exact bytes without reopening the source directory.

A caller that performs `verify_checkpoint(path)` and later `load_checkpoint(path)` creates two independent snapshots. Each operation is individually integrity-safe, but the pathname can resolve to a different fully valid checkpoint between the two calls. If both checkpoints share ModelSpec and tokenizer identities, the second load can succeed while diagnostics retained from the first manifest describe a different git/data/run/step lineage.

That is a checkpoint identity split, not a checksum failure.

## Canonical first-party sequence

`load_first_party_backend()` therefore uses this sequence:

1. `prepare_checkpoint_load(path)` reads and verifies one immutable D05 snapshot.
2. `verified.manifest` is used to validate ModelSpec, parameter count, context, tokenizer config, tokenizer vocabulary and tokenizer version.
3. `TwelveSixDecoder` is reconstructed from that verified ModelSpec.
4. `load_verified_checkpoint(verified, ...)` applies weights from the same snapshot without reopening the pathname.
5. The backend stores the manifest returned by that exact verified load and uses it for diagnostics.

Training RNG state remains intentionally unrestored for inference.

## Regression proof

`tests/test_first_party_atomic_snapshot.py` creates two separately valid checkpoints with:

- identical S0 ModelSpec/tokenizer compatibility;
- different model weights;
- different git, dataset, run, step and token-count identities.

The test captures checkpoint A, replaces its source directory with checkpoint B immediately before the verified load, and proves:

- only one pathname snapshot is prepared by the first-party loader;
- the exact same `VerifiedCheckpoint` object is consumed for weight load;
- loaded model parameters remain byte-equivalent to A, not B;
- backend diagnostics remain bound to A;
- an independent post-load verification of the pathname sees B.

This is a fail-closed provenance/identity property. It is not a promotion claim, a Windows/NVDA runtime claim, or a public-serving security claim.

## Truth boundary

Canonical Base remains random-initialized and pretraining-only. No foreign pretrained weights, instruction/alignment/refusal/personality/domain-specialization behavior, paid compute, CANDIDATE/STABLE promotion, or audit verdict is introduced by this change.
