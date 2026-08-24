# D03/D05 HF-style export transactional safety

This package is a late-wave follow-on to strict checkpoint/resume PR #61 and checkpoint filesystem/snapshot hardening PR #85. It does not change model, Trainer, tokenizer, data, evaluation, inference serving, or Base behavior.

## Defects closed

The pre-v2 HF-style exporter called `verify_checkpoint()` and then reopened `weights.safetensors` and `manifest.json` by pathname. That recreated a verify-then-reopen race after PR #85 removed the same class from checkpoint loading. The exporter also implemented `overwrite=True` by deleting an existing destination and wrote directly into the final path, so copy/hook/process failure could destroy prior evidence or leave a partial export.

HF-style export v2 now:

- calls `prepare_checkpoint_load()` once and builds from that exact verified in-memory checkpoint snapshot;
- never reopens canonical source weights or the source manifest after verification;
- stages the complete export in a sibling directory and verifies it before publication;
- keeps existing export directories immutable, including when callers pass `overwrite=True`;
- publishes with an atomic no-replace operation on supported Windows/Linux runtimes, so a destination that appears during the final race window is preserved rather than replaced;
- fails closed instead of silently using a weaker publication primitive on unsupported platforms;
- removes failed staging directories;
- adds `verify_hf_directory()` with exact inventory, regular-file/no-symlink checks, attestation checksum validation, source-manifest identity self-checks, canonical weight hash/size binding, valid config JSON, parity hashes, and conservative compatibility-claim validation;
- versions export and parity envelopes as `12-6.hf-style-export.v2` and `12-6.export-parity-request.v2`.

## Parity-hook reference identity

The external parity hook no longer receives the mutable original checkpoint pathname. When a hook is requested, D05 materializes a private five-file checkpoint-v1 reference directory from the already verified in-memory bytes, verifies that reference with `prepare_checkpoint_load()`, and passes the reference plus unpublished HF staging directory to the hook.

This prevents a valid checkpoint-path replacement after the initial snapshot from causing parity evidence to refer to different source weights/lineage than the export. The temporary reference is removed after the hook finishes, whether it succeeds or fails.

A parity hook may attach evidence, but D03/D05 do not turn that attachment into an architecture/runtime compatibility claim. A distinct D07/independent runtime-parity package remains the correct authority for logits/tokens/decode parity.

## Compatibility truth boundary

The directory is only `HF_STYLE_SAFETENSORS_DIRECTORY`. `model.safetensors` is an exact byte copy of canonical D05 checkpoint weights, but the package records:

- `transformers_architecture = NOT_CLAIMED`
- `runtime_logit_generation_parity = NOT_TESTED`

No Transformers AutoModel, vLLM, GGUF/llama.cpp, GPU, mixed-precision, distributed, or cross-OS parity claim is created by export integrity alone.

## Publication boundary

Sibling staging plus atomic no-replace publication prevents normal write/hook/verification failures from publishing a partial destination and prevents an existing or concurrently created destination from being destroyed. This is not a power-loss durability, directory-fsync, NFS/object-store atomicity, or final S8+/distributed checkpoint/export claim. Current S0 exports are small LOCAL_FREE artifacts.

## Regression evidence required

Tests cover:

- source mutation after verified checkpoint snapshot;
- parity-hook reference consistency after source-path tamper;
- immutable existing destinations;
- concurrent destination creation and preservation;
- hook failure cleanup of staging/reference directories;
- exported-weight tamper;
- attestation tamper;
- symlink payload rejection;
- v2 verifier success and conservative compatibility claims.

## Late-wave ownership

PR #95 owns this exporter surface. Overlapping PR #105 was closed unmerged. PR #111 explicitly closed its overlapping exporter implementation after the ownership refresh and retained only its distinct plan for a parity-only additive descendant on top of #95.

Status remains EXPERIMENTAL. No paid compute, foreign pretrained Base weights, instruction/alignment/refusal/personality/domain-specialization behavior, CANDIDATE/STABLE promotion, or audit verdict is introduced by this package.
