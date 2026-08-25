# D05 trained S0 HTTP inference evidence

## Purpose

The canonical first-party inference, checkpoint, parity, CLI and local server surfaces already exist in the S0 successor lineage. This package does not replace them. It closes the remaining transport-level evidence gap: prove that a **real trained S0 model**, published through the strict D05 checkpoint format and reloaded only through the first-party backend, produces the same canonical raw-Base completion semantics when exercised through the actual loopback HTTP `/v1/completions` endpoint.

The implementation is intentionally additive. It does not edit D01 architecture, D02 Trainer, D03 data, D04 tokenizer/packing, D05 checkpoint core, D06 gates, the D07 generation/sampling/server implementations, D08 locks, or D10 promotion governance.

## Exact path

`python -m twelve_six.inference.http_evidence` performs one LOCAL_FREE CPU cycle:

1. validates the exact checkout SHA and committed S0 dataset identities;
2. constructs the canonical random-initialized 10,140-parameter D01 S0 decoder and D04 byte tokenizer;
3. optimizes only the committed D03 train split with the D02 Trainer and evaluates validation without optimizer mutation;
4. binds ModelSpec, InitSpec, tokenizer config/vocabulary, dataset, exact train split, packing, environment lock, run manifest, seed, step and tokens-seen into a D05 checkpoint identity;
5. publishes and verifies a pickle-free checkpoint-v1 directory;
6. reloads through `load_first_party_backend()` and compares the live trained model with the reloaded backend using the existing D07 parity harness at zero tolerance for logits, greedy tokens and decode output;
7. starts the existing D07 `CompletionHTTPServer` on `127.0.0.1` with an ephemeral port and drives it over real TCP/HTTP;
8. proves greedy HTTP output equals the direct canonical `completion_response()`, same-seed sampling repeats, stop-string stripping works, exact-context completion returns length with zero new tokens, over-context input fails closed, and chat/messages semantics remain rejected;
9. writes a self-hashed `12-6.s0-trained-http-inference-evidence.v1` JSON object and retains the exact trained checkpoint for downstream local serving/replay.

No model architecture, sampling, stop handling, checkpoint serialization, tokenizer behavior or HTTP serving behavior is reimplemented by the evidence runner; it composes the existing maintained project interfaces.

## Dedicated exact-head workflow

`.github/workflows/d05-trained-http-inference.yml` checks out the exact PR head on fixed `ubuntu-24.04` / CPython `3.11.16`, validates the D08 hash-locked x86-64 environment and repository checks, creates the same locked execution environment, runs focused tests, executes a real 40-step evidence cycle, validates the retained JSON against the exact source SHA, runs the full repository pytest suite, and uploads both the trained checkpoint and machine evidence for 30 days.

The output artifact is named `d05-trained-http-inference-<SOURCE_SHA>` and contains:

- `locked-environment-linux-x86_64.json`;
- `s0-trained-http-inference-evidence.json`;
- the exact `trained-checkpoint/` checkpoint-v1 directory.

## Evidence contract

The self-hashed JSON binds candidate/source identity, ModelSpec and InitSpec, parameter count, tokenizer config/vocabulary/version, D03 dataset and split identities, D04 packing identity, D08 environment lock, complete run-manifest hash, real optimization metrics, checkpoint ID/files/step/tokens, first-party diagnostics, zero-tolerance parity results and the HTTP proof matrix.

The validator fails closed when the self-hash changes, source SHA drifts, validation was optimized, training-loss decrease is not proven, checkpoint lineage/serialization changes, direct-vs-reloaded parity fails, any required HTTP proof bit is missing, or the non-promotion truth boundary is weakened.

## Raw Base and authority boundary

This is pretraining-only raw Base completion evidence. The server receives the caller's `prompt` directly. No system prompt, role template, instruction wrapper, refusal layer, ethics/personality/domain-specialization behavior or foreign pretrained weights are added.

The listener is loopback-only for this evidence and is not a public-service hardening claim. The package does not claim Windows/NVDA live runtime support; the repository's physical trailing-period Windows checkout blocker remains separate. It does not claim alternative backend parity, TLS/authentication, streaming, batching, concurrency, throughput, GPU/distributed behavior or paid compute.

A green D05 evidence run is **not** an AUDIT-A/AUDIT-B verdict and does not create CANDIDATE or STABLE promotion authority. S0 remains EXPERIMENTAL until the independent exact-candidate audit/promotion gates are satisfied.
