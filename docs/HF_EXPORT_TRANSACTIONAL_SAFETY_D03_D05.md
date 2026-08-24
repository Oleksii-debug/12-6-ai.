# D03/D05 HF-style export transactional safety

This package is a late-wave follow-on to the strict checkpoint/resume work in PR #61 and the filesystem/snapshot hardening in PR #85. It does not change model, trainer, tokenizer, data, evaluation, inference, or Base behavior.

## Defects closed

The pre-v2 HF-style exporter called `verify_checkpoint()` and then reopened `weights.safetensors` and `manifest.json` by pathname. That recreated a verify-then-reopen race after PR #85 had removed the same class from checkpoint loading. The exporter also implemented `overwrite=True` by deleting the existing destination before publishing the replacement, so a failed copy/hook/process could destroy prior valid evidence and leave a partial export.

HF-style export v2 now:

- calls `prepare_checkpoint_load()` once and builds from that exact verified in-memory checkpoint snapshot;
- never reopens canonical source weights or the source manifest after verification;
- stages the complete export in a sibling directory and verifies it before publish;
- keeps existing export directories immutable, including when callers pass `overwrite=True`;
- publishes only after the staged export verifies, and removes failed staging directories;
- adds `verify_hf_directory()` with exact inventory, regular-file/symlink checks, attestation checksum validation, source-checkpoint self-consistency, canonical weight hash/size binding, config/parity hashes, and conservative compatibility-claim validation;
- versions export and parity envelopes as `12-6.hf-style-export.v2` and `12-6.export-parity-request.v2`.

## Compatibility truth boundary

The directory is only `HF_STYLE_SAFETENSORS_DIRECTORY`. `model.safetensors` is an exact byte copy of canonical D05 checkpoint weights, but the package still records:

- `transformers_architecture = NOT_CLAIMED`
- `runtime_logit_generation_parity = NOT_TESTED`

An external D07/independent parity hook may attach evidence, but D03/D05 do not convert that attachment into an architecture/runtime compatibility claim. The hook runs against the verified staging directory before publication so hook failure cannot publish a partial export.

## Publication boundary

The sibling rename prevents a valid non-empty existing export from being destructively replaced and prevents normal hook/write/verification failures from publishing a partial destination. This is not a claim of power-loss durability, distributed/object-store atomicity, NFS semantics, or a final large-stage checkpoint/export format. Current S0 exports are small LOCAL_FREE artifacts.

## Required regression evidence

Tests cover source mutation after snapshot, immutable existing destinations, hook failure cleanup, exported-weight tamper, attestation tamper, symlink payload rejection, v2 verifier success, and preservation of conservative compatibility claims.

Status remains EXPERIMENTAL. No paid compute, foreign pretrained Base weights, instruction/alignment/refusal/personality/domain-specialization behavior, CANDIDATE/STABLE promotion, or audit verdict is introduced by this package.
