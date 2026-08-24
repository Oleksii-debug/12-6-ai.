# D05 S0 GitHub artifact upload/download replay

## Scope

PR #119 already owns retained trained S0 checkpoint generation and producer-side
validation. PR #133 separately owns the reusable portable-bundle trust validator,
including path normalization, symlink/non-regular rejection and complete bundle
manifest checks.

This package owns neither of those surfaces. Its only purpose is to execute the
remaining transport experiment: create a #119 retained checkpoint/evidence payload,
upload it through GitHub Actions, start a **fresh job on a fresh runner**, download the
artifact, prove the extracted file bytes equal the producer bytes, re-run the existing
D05/D07 semantic validator, and execute the installed CLI from the downloaded
checkpoint.

No new model, Trainer, tokenizer, checkpoint format, sampler, server, portable-bundle
validator or release contract is introduced.

## Producer → fresh consumer proof

`.github/workflows/d05-s0-artifact-upload-download.yml` has two jobs.

The producer uses the exact current PR head, CPython 3.11.16 and the existing D08
hash-locked Linux x86-64 environment. It delegates the real 40-step retained
checkpoint build and semantic validation to `twelve_six.inference.s0_artifact` from
#119. Before upload it creates a deterministic `sha256sum` list over every staged
payload file and verifies that list locally. The exact checksum-list SHA-256, GitHub
artifact ID and GitHub artifact digest are passed as job outputs.

The consumer starts on a new Ubuntu runner. It downloads the same-run artifact using
immutable `actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093`
(v4.3.0). It first verifies that the checksum-list file itself matches the producer
hash and then runs `sha256sum --check` across every downloaded payload file. Only after
that byte replay check does it:

1. re-run #119 `s0_artifact validate` against the downloaded checkpoint/evidence;
2. execute the installed plain stdin CLI against the downloaded checkpoint;
3. execute sampled JSON CLI diagnostics and bind checkpoint ID + source Git SHA;
4. emit a self-hashed fresh-consumer report binding the producer artifact ID/digest,
   producer checksum-file hash, checkpoint ID and generation-evidence identity.

This is intentionally a real GitHub Actions upload/download experiment rather than a
second portable-artifact validation framework. General untrusted-bundle path policy
remains #133-owned.

## Truth boundary

Success proves same-workflow GitHub Actions artifact transport/replay for this exact
S0 Linux payload under the declared hash-locked environment. It does not prove
long-term archival, cross-workflow/cross-repository transfer, object stores, NFS/SMB,
arbitrary untrusted bundles, POSIX metadata preservation, Windows/NVDA, GPU or
distributed portability, Transformers/vLLM/GGUF/llama.cpp compatibility, or
cross-hardware bitwise equivalence.

The GitHub artifact digest is recorded as workflow metadata; it is not falsely equated
to the extracted payload checksum list.

Canonical Base remains random-initialized and pretraining-only. No foreign pretrained
weights, hidden chat/system template, instruction/alignment/refusal/personality/domain
behavior, materially paid compute, audit verdict, CANDIDATE or STABLE promotion is
introduced.
