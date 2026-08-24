# D05 exact-candidate S0 inference acceptance

## Purpose

The canonical inference implementation already exists and is exact-green through the
D05 first-party adapter, strict D05 checkpoint lineage, D07 generation harness and the
local raw-Base completions server. This package does not create a second inference
stack. It adds one exact-candidate acceptance/evidence layer over those existing APIs.

The immediate reason is provenance: the original direct D05 acceptance fixture from
PR #63 predated the stricter checkpoint identity introduced by later D05 work and was
intentionally held during convergence. Current D04/D01 integration proves the composed
quality path, but there was no dedicated retained artifact that exercised the complete
strict inference-facing surface on one current exact head.

## Acceptance path

`python -m twelve_six.inference.acceptance` performs only LOCAL_FREE CPU work:

1. prove the supplied candidate SHA equals the checkout HEAD;
2. load the canonical S0 stage, D04 byte tokenizer and committed D03 train split;
3. construct the D01 random-initialized 10,140-parameter decoder and run D02 Trainer
   for the configured number of optimizer steps;
4. bind the current strict D05 identity including ModelSpec, InitSpec, tokenizer config
   and vocabulary, dataset/split, packing, full run/training config and D08 lock hash;
5. save a checkpoint-v1 artifact and reload it through the canonical first-party
   `InferenceBackend`;
6. prove exact direct-vs-reloaded greedy output and same-seed sampling repeatability;
7. run zero-tolerance logit/token/decode parity through the existing D07 parity harness;
8. prove stop-token, stop-string and exact/over-context semantics;
9. execute the canonical JSON CLI in a fresh Python process and reject ANSI output;
10. open the existing D07 server on an ephemeral loopback port and compare real
    `POST /v1/completions` output against direct canonical generation;
11. construct checksum-valid incompatible tokenizer/context/vocabulary checkpoints and
    a byte-corrupt checkpoint and require every state to fail closed;
12. emit `12-6.s0-inference-acceptance.v1` plus the tiny verified checkpoint artifact.

The runner calls existing model, Trainer, checkpoint, loader, generation, parity, CLI
and server APIs. It does not duplicate decoder architecture, tokenization, sampling,
serialization, OpenAI request semantics or HTTP handling.

## Exact-head workflow

`.github/workflows/d05-s0-inference-acceptance.yml` checks out the exact pull-request
head, verifies that checkout identity, runs the existing D08 locked-environment and
repository checks, creates the hash-locked x86_64 Python environment, runs the focused
acceptance regressions, executes a 40-step acceptance cycle, validates the generated
manifest and retains the evidence/checkpoint for 30 days.

The generic repository CI remains independent. A queued or running workflow is not
PASS evidence; only a terminal successful run bound to the exact source head is
acceptable.

## Raw Base truth boundary

The accepted endpoint is raw pretraining-only Base completion. There is no hidden
system text, role/chat template, instruction alignment, refusal layer, ethics layer,
personality or domain-specialization behavior. `/v1/chat/completions`, `messages` and
streaming remain explicitly unsupported rather than silently approximated.

This package does not claim public-server hardening, authentication, TLS, batching,
streaming, KV-cache performance, Transformers/vLLM/GGUF/llama.cpp parity or external
service compatibility beyond the documented local text-completions subset.

## Windows / NVDA boundary

CLI output remains plain stdout/stderr plus JSON diagnostics with no ANSI/TUI
requirement. A live Windows/NVDA acceptance run is still
`NOT_TESTED_BLOCKED_BY_REPOSITORY_IDENTITY` because the physical GitHub repository name
ends in a period and Windows checkout fails before Python execution. This package does
not hide that repository-governance blocker or convert code-level accessibility into a
Windows/NVDA PASS claim.

## Authority

The evidence is `LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION`.

It does not authorize paid compute, introduce foreign pretrained weights, issue an
AUDIT-A/AUDIT-B verdict, protect `main`, or promote S0 to CANDIDATE/STABLE. Acceptance
manifests hard-code `audits_pass=false` and `promotion_eligible=false` and validate
those truth boundaries fail closed.
