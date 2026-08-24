# S0 first-party inference evidence

This package closes the residual D05/D07 evidence gap after the first-party adapter, transactional checkpoint hardening, local raw-completions server, and D01 successor convergence were already implemented elsewhere.

It is intentionally stacked on PR #88 exact head `59193ada9586d0542b027f46d32ac923841fae7f`. It does not replace or duplicate D01 model architecture, D02 Trainer, D03 data semantics, D04 tokenizer/packing, D05 checkpoint-v1, D06 stage gates, or D07 generation/sampling/server logic.

## What the evidence run proves

`python -m twelve_six.inference.s0_evidence` performs one LOCAL_FREE CPU S0 fixture run using the accepted project components, then retains the resulting checkpoint-v1 artifact and proves the complete first-party inference chain:

- exact candidate SHA and physical repository identity;
- canonical random-initialized, pretraining-only S0 ModelSpec/InitSpec;
- committed D03 train split plus D04 byte tokenizer/vocabulary/packing identities;
- real D02 optimizer steps before inference;
- D05 checkpoint-v1 with SafeTensors and no pickle;
- fresh `load_first_party_backend()` verification/reconstruction before inference;
- exact same-runtime logits, greedy-token and decoded-output parity between the trained in-memory model and the freshly reloaded backend with `atol=0`, `rtol=0`;
- deterministic greedy generation;
- seeded sampled-generation repeatability and direct-vs-reloaded equality;
- token stop, text stop/strip, context-limit stop, and over-context fail-closed behavior;
- raw `/v1/completions` handoff equivalence to canonical greedy generation with no hidden prompt/system/chat semantics;
- corrupt checkpoint rejection;
- model, tokenizer-config and vocabulary identity mismatch rejection before target-model mutation;
- privacy-safe first-party diagnostics;
- machine-report self-hash.

The dedicated workflow uses fixed Ubuntu 24.04 / CPython 3.11.16, verifies the D08 locked environment, creates a hash-locked runtime venv, performs a 40-step real CPU fixture run, validates the report, and uploads both `inference_evidence.json` and the complete trained `checkpoint-v1/` directory as one 30-day Actions artifact.

## Machine report

Schema: `12-6.s0-first-party-inference-evidence.v1`.

The report binds candidate SHA, ModelSpec/InitSpec, data/tokenizer/packing/environment identities, optimizer step/tokens-seen, checkpoint ID and file hashes, backend diagnostics, generation fixtures, exact parity metrics, stop/context behavior, raw completion handoff, fail-closed probes, and an evidence self-hash.

The retained checkpoint is evidence for this LOCAL_FREE exact-head run. It is not a release model, STABLE artifact, quality claim, or authorization for later-stage compute.

## Truth boundary

This package does not claim live Windows/NVDA execution, public-server TLS/auth/rate-limit hardening, streaming, batching, KV-cache performance, Transformers/vLLM/GGUF/llama.cpp parity, GPU/distributed equivalence, CANDIDATE/STABLE promotion, or an independent audit verdict.

Canonical Base remains random-init and pretraining-only. No foreign pretrained weights, instruction/alignment/refusal/ethics/personality/domain-specialization behavior, or materially paid compute is introduced.
