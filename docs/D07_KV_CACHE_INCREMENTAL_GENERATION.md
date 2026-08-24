# D07 model-native KV-cache incremental generation

## Problem

The accepted first-party D07 backend is correct but stateless: every generated token calls the D01 decoder on the complete prompt plus every previously generated token. For a prompt of `P` tokens and `N` decode steps, that repeatedly projects and processes `P + (P+1) + ... + (P+N-1)` input positions.

That is acceptable for the tiny S0 proof model, but it is a structural inference bottleneck before S1+ and longer contexts. The current server/evidence PRs explicitly leave KV-cache performance as not implemented/not tested.

This package adds one collision-safe cache path to the canonical model/backend instead of creating an alternate decoder.

## Design

The cache is model-native and inference-only:

- `AttentionKVCache` stores unexpanded key/value tensors in `[batch, n_kv_heads, sequence, head_dim]` form. GQA therefore keeps only the configured KV heads rather than materializing repeated attention heads in cache memory.
- `DecoderKVCache` binds all layer caches to the exact `ModelSpec` identity, batch size and sequence length.
- prompt prefill executes the normal causal attention path once and retains the K/V tensors produced by the same canonical projections;
- single-token decode computes RoPE at the absolute cached sequence offset, appends one K/V position, then attends the one final query over all prior+current keys;
- single-token cached attention deliberately uses `is_causal=False`: for a single query at the newest position every cached key is legal, while PyTorch's non-square `is_causal=True` semantics would describe a different upper-left mask;
- model training mode is rejected by the cache API; D07 generation sessions temporarily switch to eval and restore the prior mode when closed;
- cache state is ephemeral runtime state. It is not part of `ModelSpec`, `InitSpec`, trainable parameters, `state_dict`, or D05 checkpoint-v1.

The existing stateless `forward()` and `next_token_logits()` remain available. `generate()` uses the incremental session only when a backend exposes `begin_generation`; other D07 backends retain the existing stateless fallback.

## Semantic acceptance

`tests/test_kv_cache_generation.py` requires:

- prompt prefill logits equal the unchanged full-prefix forward path exactly;
- each cached single-token step matches a fresh full-prefix forward within `1e-6` absolute/relative tolerance;
- GQA cache tensors retain `n_kv_heads`, not expanded `n_heads`;
- invalid model identity, batch, layer/sequence state and full-context appends fail closed;
- cache use is rejected while the decoder is in training mode;
- canonical D07 greedy output is identical to the stateless path;
- same-seed top-k/top-p sampling remains repeatable and matches the stateless path on the deterministic S0 fixture;
- session close restores the model's pre-session train/eval mode;
- cache construction changes neither parameter count, state-dict keys nor ModelSpec identity.

## Machine evidence

The dedicated workflow runs:

```text
python tools/run_s0_kv_cache_evidence.py \
  --repo-root . \
  --source-sha "$SOURCE_SHA" \
  --steps 16 \
  --output s0-kv-cache-evidence.json
```

The report compares cached logits to a fresh stateless full-prefix call at every step and records actual ephemeral cache bytes plus decoder input positions processed. The position-count metric is an algorithmic work metric, not a latency, throughput, FLOP or MFU benchmark.

For 16 decode-logit evaluations after a 4-byte-token prompt, the stateless path processes `4+5+...+19 = 184` decoder input positions. The cache path needs the 4-position prefill plus 15 single-token appends = 19 positions. Exact observed logit error and cache bytes remain workflow evidence rather than being hard-coded as a PASS claim in this document.

## Boundaries

This is a D07 inference-mechanics package. It does not change Base training behavior, tokenizer IDs, checkpoint format, generation stop semantics, OpenAI-compatible request semantics or stage promotion.

It does not claim GPU latency, batching, paged KV cache, prefix sharing across requests, speculative decoding, quantized KV cache, distributed serving, vLLM/llama.cpp parity, public-server readiness, materially paid compute, AUDIT-A/B PASS, CANDIDATE or STABLE status. Those require separate evidence and ownership.
