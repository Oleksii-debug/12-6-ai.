# D03/D05 HF-style export transactional safety

This package is a late-wave follow-on to strict checkpoint/resume PR #61 and checkpoint filesystem/snapshot hardening PR #85. It does not change model, Trainer, tokenizer, data, evaluation, inference serving, or Base behavior.

## Defects closed

The pre-v2 HF-style exporter called `verify_checkpoint()` and then reopened `weights.safetensors` and `manifest.json` by pathname. That recreated a verify-then-reopen race after PR #85 removed the same class from checkpoint loading. The exporter also implemented `overwrite=True` by deleting an existing destination and wrote directly into the final path, so copy/hook/process failure could destroy prior evidence or leave a partial export.

HF-style export v2 now:

- calls `prepare_checkpoint_load()` once and builds from that exact verified in-memory checkpoint snapshot;
- never reopens canonical source weights or the source manifest after verification;
- keeps existing export directories immutable, including when callers pass `overwrite=True`;
- publishes with an atomic no-replace operation on supported Windows/Linux runtimes, so a destination that appears during the final race window is preserved rather than replaced;
- fails closed instead of silently using a weaker publication primitive on unsupported platforms;
- adds `verify_hf_directory()` with exact inventory, regular-file/no-symlink checks, attestation checksum validation, source-manifest identity self-checks, canonical weight hash/size binding, valid config JSON, parity hashes, and conservative compatibility-claim validation;
- versions export and parity envelopes as `12-6.hf-style-export.v2` and `12-6.export-parity-request.v2`;
- never exposes the final publication staging pathname to an external parity hook;
- strictly removes hook-visible temporary paths before final staging is created, and aborts publication if cleanup fails;
- builds the final staging tree only after parity evidence is canonicalized in memory and hook-visible paths have been removed, then verifies and atomically publishes that fresh private tree.

## Parity-hook reference identity and isolation

The external parity hook does not receive the mutable original checkpoint pathname or the final publication staging pathname. When a hook is requested, D05 materializes two disposable private paths from the already verified in-memory checkpoint snapshot:

1. a five-file checkpoint-v1 reference directory, verified with `prepare_checkpoint_load()`;
2. an HF-style candidate containing the exact weights/config/source-manifest bytes needed by the parity harness.

The hook receives only those disposable paths. Its mapping result is canonicalized into immutable JSON bytes before cleanup. Both hook-visible paths are then synchronously removed with errors propagated rather than suppressed. Only after cleanup succeeds does D05 create the separate final staging tree from the original verified checkpoint bytes plus the canonicalized parity bytes.

This closes the retained-hook-path verify-to-publish race: a hook that keeps its candidate pathname can no longer mutate the tree later renamed into the destination. It also prevents a source-checkpoint pathname replacement after the initial snapshot from changing parity lineage.

A parity hook may attach evidence, but D03/D05 do not turn that attachment into an architecture/runtime compatibility claim. A distinct D07/independent runtime-parity package remains the correct authority for logits/tokens/decode parity.

## Cleanup boundary

Temporary cleanup is fail-closed. The exporter does not use `ignore_errors=True` for reference, hook-candidate, or final-staging removal. Cleanup attempts remove a real directory recursively or unlink a replaced non-directory/symlink without following it, then verify that the expected temporary pathname is absent. If cleanup raises or the pathname remains, export raises and does not proceed to final publication.

This is an integrity/publication guarantee for exporter-owned temporary pathnames, not a confidentiality guarantee against arbitrary code with the same OS user authority. A parity hook can copy bytes elsewhere, and a hostile same-user process that can discover and mutate the private final staging tree is outside this package's threat model. Closing that stronger filesystem-adversary boundary would require fd/inode/OS-authority binding; it is not claimed here.

## Compatibility truth boundary

The directory is only `HF_STYLE_SAFETENSORS_DIRECTORY`. `model.safetensors` is an exact byte copy of canonical D05 checkpoint weights, but the package records:

- `transformers_architecture = NOT_CLAIMED`
- `runtime_logit_generation_parity = NOT_TESTED`

No Transformers AutoModel, vLLM, GGUF/llama.cpp, GPU, mixed-precision, distributed, or cross-OS parity claim is created by export integrity alone.

## Publication boundary

Private final staging plus atomic no-replace publication prevents normal write/hook/verification failures from publishing a partial destination and prevents an existing or concurrently created destination from being destroyed. No untrusted parity callback is invoked after final staging exists. This is not a power-loss durability, directory-fsync, NFS/object-store atomicity, hostile-same-user-filesystem, or final S8+/distributed checkpoint/export claim. Current S0 exports are small LOCAL_FREE artifacts.

## Regression evidence required

Tests cover:

- source mutation after verified checkpoint snapshot;
- parity-hook reference consistency after source-path tamper;
- immutable existing destinations;
- retained hook-candidate pathname mutation after hook return without mutation of the final publish source;
- fail-closed hook-visible cleanup failure before final staging/publication;
- concurrent destination creation and preservation;
- hook failure cleanup of candidate/reference directories;
- exported-weight tamper;
- attestation tamper;
- symlink payload rejection;
- v2 verifier success and conservative compatibility claims.

## Late-wave ownership

PR #95 owns this exporter surface. Overlapping PR #105 was closed unmerged. PR #111 explicitly closed its overlapping exporter implementation after the ownership refresh and retained only its distinct plan for a parity-only additive descendant on top of #95.

Status remains EXPERIMENTAL. No paid compute, foreign pretrained Base weights, instruction/alignment/refusal/personality/domain-specialization behavior, CANDIDATE/STABLE promotion, or audit verdict is introduced by this package.
