# D05 S0 trained-checkpoint HTTP parity

This package closes a narrow late-wave integration gap without changing model, tokenizer,
training, checkpoint, evaluation, sampling, or HTTP server semantics.

## Why this exists

PR #88 already composes exact-green D02 training, D04 exact-candidate evaluation, D05
transactional checkpoint safety, and D07 local serving. Its integration oracle proves the
train/checkpoint/reload/evaluation path and separately proves that the D07 server can bind a
canonical model/tokenizer backend. The remaining seam was that the loopback HTTP server had
not been exercised end-to-end with a checkpoint that was actually trained, saved by D05,
verified, and reloaded through the first-party D05/D07 loader.

`python -m twelve_six.inference.s0_http_evidence` is an evidence collector, not a second
inference implementation. It delegates to the existing D01 model, D02 Trainer, D04 tokenizer
and packing identities, D05 checkpoint APIs, D07 first-party adapter, parity harness,
OpenAI-compatible completion handoff, and local server.

## Exact evidence cycle

For the supplied exact checkout SHA the collector:

1. loads frozen S0 ModelSpec/InitSpec and the committed D03 train split;
2. applies the declared seed before random scratch model construction;
3. performs a bounded LOCAL_FREE fp32 CPU training run through the canonical Trainer;
4. binds the exact candidate, ModelSpec, InitSpec, tokenizer config+vocab, dataset manifest,
   train split, packing identity, environment lock, seed, step and tokens-seen into D05;
5. writes a real SafeTensors checkpoint outside the repository;
6. reloads it through `load_first_party_backend`;
7. requires exact zero-tolerance logits/token/decode parity against the in-memory trained model;
8. serves only the reloaded backend on loopback and compares real `/v1/completions` responses
   against direct `completion_response()` for greedy and seeded sampling;
9. proves seeded HTTP sampling repeatability, stop-string behavior, context-limit behavior,
   over-context rejection, chat-endpoint rejection, `/healthz`, and `/v1/models`;
10. writes a self-hashed `12-6.s0-http-parity-evidence.v1` JSON report and retains the generated
    checkpoint as a CI artifact.

Server-generated completion IDs and wall-clock timestamps are intentionally excluded from
semantic response comparison. Prompt text is fixed test input and is not logged by the server.

## Local command

Run from an exact Git checkout with the canonical locked dependencies available:

```text
python -m twelve_six.inference.s0_http_evidence \
  --repo-root . \
  --candidate-sha "$(git rev-parse HEAD)" \
  --output-dir /tmp/d05-s0-http-parity \
  --train-steps 4 \
  --seed 20260825
```

The output directory contains:

- `s0-http-parity-evidence.json`;
- `checkpoint/manifest.json`;
- `checkpoint/MANIFEST.sha256`;
- `checkpoint/weights.safetensors`;
- `checkpoint/state.safetensors`;
- `checkpoint/state.json`.

## Truth boundary

This is LOCAL_FREE/free-hosted CPU S0 evidence. It does not authorize paid compute and does
not claim CANDIDATE, AUDITED_CANDIDATE, STABLE, audit authority, public-server hardening,
TLS/auth, streaming, batching, KV-cache performance, live Windows/NVDA execution,
Transformers/vLLM/GGUF/llama.cpp parity, or cross-hardware bitwise reproducibility.

Canonical Base remains random-initialized and pretraining-only. No system prompt, chat-role
semantics, instruction template, alignment/refusal/personality/domain-specialization behavior,
or foreign pretrained weights are introduced.
