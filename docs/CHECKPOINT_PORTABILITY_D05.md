# D05 S0 checkpoint cross-architecture portability

## Problem

The canonical S0 stack already has exact D08 locked environments for Linux
x86_64 and Linux aarch64, and D05 checkpoint-v1 is a portable SafeTensors +
JSON format. Existing evidence verifies each locked environment independently,
but it does not yet prove that one exact checkpoint produced on x86_64 can be
transferred byte-for-byte and restored on aarch64.

That is a serialization-portability question, not a training-reproducibility
question.

## Scope

The dedicated workflow `.github/workflows/d05-checkpoint-cross-arch.yml`
creates one small real S0 checkpoint on locked Linux x86_64 and consumes the
same uploaded bytes on locked Linux aarch64.

The producer:

- checks out the exact PR source SHA;
- uses CPython 3.11.16 and the committed D08 x86_64 hash locks;
- constructs the canonical 10,140-parameter random-init S0 model;
- uses the exact D03 train split and D04 byte tokenizer/packing path;
- runs one real D02 optimizer step so model and optimizer/trainer state are
  populated;
- binds Git, ModelSpec, InitSpec, tokenizer config/vocabulary, dataset, split,
  packing, run-manifest, environment-lock, seed, step and token identities;
- saves and verifies the normal D05 checkpoint-v1 bundle;
- records the exact five-file SHA-256 inventory and checkpoint ID in a
  self-hashed producer report.

The aarch64 consumer:

- independently checks out the same exact source SHA;
- uses CPython 3.11.16 and the committed D08 aarch64 hash locks;
- downloads the x86_64 checkpoint artifact from the same workflow run;
- verifies the producer report, checkpoint ID and every transferred file hash;
- restores the checkpoint into a fresh canonical D01 model and D02 Trainer
  through the existing D05 `load_trainer_checkpoint()` API;
- keeps cross-architecture RNG restoration disabled;
- verifies optimizer step and token counters;
- loads the same transferred checkpoint through the existing D05/D07
  first-party inference loader and verifies checkpoint/source/ModelSpec
  diagnostics;
- emits a self-hashed consumer report.

## Commands

Producer:

```bash
python tools/checkpoint_portability.py produce \
  --source-sha "$SOURCE_SHA" \
  --output-dir d05-checkpoint-portability
```

Consumer:

```bash
python tools/checkpoint_portability.py consume \
  --source-sha "$SOURCE_SHA" \
  --bundle-dir d05-checkpoint-portability \
  --output d05-checkpoint-portability-consumer.json
```

Reports can be independently revalidated with the `validate-producer` and
`validate-consumer` subcommands.

## Truth boundary

A green dedicated workflow proves only that the exact D05 checkpoint
serialization and bound model/trainer state can cross the currently supported
locked Linux x86_64 -> Linux aarch64 boundary without byte drift and can be
verified/restored on the second architecture.

It does **not** prove:

- bitwise-identical training trajectories across architectures;
- bitwise-identical inference logits or generated tokens across architectures;
- portable RNG trajectory equivalence;
- GPU, CUDA, mixed-precision, distributed, Windows, macOS, NFS, SMB or
  object-store checkpoint behavior;
- materially paid compute authorization;
- foreign pretrained Base weights;
- instruction/alignment/refusal/personality/domain behavior;
- AUDIT-A/AUDIT-B PASS, CANDIDATE or STABLE promotion.

Cross-architecture numerical equivalence must be a separate measured claim with
explicit tolerances. This package deliberately does not manufacture one from a
successful serialization restore.
