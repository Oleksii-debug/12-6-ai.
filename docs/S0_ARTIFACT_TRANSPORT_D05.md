# D05 S0 artifact transport roundtrip

## Residual gap

The retained S0 generation artifact in PR #119 validates its checkpoint and generation
evidence before `actions/upload-artifact`. That is necessary, but it does not prove a
separate consumer can recover the uploaded object and execute the exact checkpoint
bytes after the GitHub Actions artifact transport boundary.

This package is a downstream D05 transport proof. It does not build a competing S0
model, trainer, checkpoint format, first-party backend, sampler, CLI, or retained
artifact implementation. The producer delegates the entire training/checkpoint/
generation path to `twelve_six.inference.s0_artifact` from #119. The new code binds
only the extracted byte tree and verifies it again in a fresh job after download.

## Transport manifest

`12-6.s0-artifact-transport.v1` binds:

- exact physical repository and full source Git SHA;
- exact D05 checkpoint ID and checkpoint Git SHA;
- exact generation-evidence file SHA-256 and its internal evidence SHA-256;
- deterministic recursive directory topology;
- every regular payload file path, byte count and SHA-256;
- required checkpoint-v1 files plus `s0-generation-evidence.json`;
- a canonical manifest self-hash.

Symlink nodes, non-regular files, absolute/non-canonical paths, path traversal,
missing required payloads, extra or missing files after transport, byte/size drift,
source drift, checkpoint-ID drift, generation-evidence drift, and truth-boundary
weakening fail closed.

The manifest intentionally does **not** bind POSIX mode bits. GitHub artifact
transport is being accepted here as a byte/content transport, not as a Unix filesystem
metadata preservation mechanism. The GitHub archive digest is recorded separately by
the workflow but is not falsely claimed to equal the extracted-tree manifest hash.

## Fresh-job proof

`.github/workflows/d05-s0-artifact-transport.yml` uses two jobs.

The producer checks out the exact PR head, verifies the D08 locked environment and
repository, builds the incumbent #119 real 40-step retained checkpoint/evidence,
validates it, creates the transport manifest, validates the pre-upload tree, and
uploads the bound directory as a 30-day artifact.

The consumer is a new Ubuntu runner with a fresh exact hash-locked environment. It
downloads the same named artifact using immutable
`actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093` (v4.3.0), then:

1. recomputes the full extracted byte inventory against the producer manifest;
2. re-runs D05 checkpoint verification on the downloaded directory;
3. re-runs #119 generation-evidence validation against those downloaded bytes;
4. executes the installed plain stdin CLI against the downloaded checkpoint;
5. executes sampled JSON CLI diagnostics and binds the backend checkpoint/Git
   identities back to the transport manifest;
6. emits a separate self-hashed fresh-consumer report containing GitHub artifact ID
   and archive digest as observed workflow metadata.

This proves consumption after upload/download rather than only pre-upload validity.

## Truth boundary

This remains LOCAL_FREE/free-hosted CPU S0 evidence. It does not imply that the GitHub
artifact is a release, CANDIDATE or STABLE checkpoint. It grants no audit authority and
does not change historical AUDIT-A/AUDIT-B verdicts. It does not claim object-store,
NFS/SMB, arbitrary archive-format, long-term archival, Windows/NVDA, GPU/distributed,
Transformers/vLLM/GGUF/llama.cpp, or cross-hardware bitwise portability.

Canonical Base remains random-initialized and pretraining-only. No foreign pretrained
weights, hidden system/chat template, instruction/alignment/refusal/personality/domain
behavior, or materially paid compute are introduced.
