# D07 model-native KV-cache incremental generation

## Problem

The accepted first-party D07 backend was correct but stateless: every generated token called the D01 decoder on the complete prompt plus every previously generated token. For a prompt of `P` tokens and `N` decode steps, that repeatedly projects and processes `P + (P+1) + ... + (P+N-1)` input positions.

That is acceptable for the tiny S0 proof model, but it is a structural inference bottleneck before S3+ and longer contexts. This package extends the canonical model/backend rather than creating an alternate decoder or a second serving stack.

## Model-native cache design

The cache is inference-only and remains inside the accepted D01/D07 path:

- `AttentionKVCache` stores unexpanded key/value tensors in `[batch, n_kv_heads, sequence, head_dim]` form. GQA therefore stores only configured KV heads instead of repeated query-head copies.
- `DecoderKVCache` binds all layer caches to the exact `ModelSpec` identity, batch size and sequence length.
- prompt prefill executes the normal causal attention path once and retains the K/V tensors produced by the canonical projections;
- single-token decode computes RoPE at the absolute cached sequence offset, appends one K/V position, then attends the newest query over all prior and current keys;
- single-token cached attention deliberately uses `is_causal=False`: for the one newest query every cached key is legal, while non-square `is_causal=True` would describe a different mask;
- cache state is ephemeral runtime state. It is not part of `ModelSpec`, `InitSpec`, trainable parameters, `state_dict`, or D05 checkpoint-v1.

The existing stateless `forward()` and `next_token_logits()` remain available. `generate()` uses the incremental session only when a backend exposes `begin_generation`; other D07 backends keep the stateless fallback. This preserves the first-party backend as the correctness reference without forcing exported or third-party runtimes to adopt the Python cache representation.

## Serving lifecycle

The model-level cache already supports equal-length batches: prefill accepts `[batch, sequence]` and cached decode accepts `[batch, 1]`. A focused batch-2 contract now verifies that batched prefill/decode matches the canonical full-prefix path and that batch identity is retained across cache growth.

Multiple independent `S0TorchGenerationSession` objects may also coexist on one backend. Session lifecycle is reference-counted under a lock: the first session records the model's prior train/eval state and enters eval mode; intermediate closes do not restore training mode while another cache is live; the final close restores the original state. This closes the previous overlapping-session race without adding a request scheduler or changing generation semantics.

This is simultaneous-session safety, not continuous batching. Independent sessions still execute independently, and ragged batches are not compacted into one decode launch.

## Memory accounting and GQA

`kv_cache_payload_bytes()` computes the logical unexpanded payload without allocating model/cache tensors:

`2 * n_layers * n_kv_heads * head_dim * sequence_length * batch_size * element_size_bytes`

`S0TorchInferenceBackend.estimate_cache_bytes()` applies the same formula using the model's current parameter element size. Actual session accounting still sums the live K/V tensors, so the estimator is checked against allocated cache payload on the canonical path.

Planning examples for one full-context sequence in BF16/FP16 payload bytes are:

| Geometry | Context | KV heads | KV payload |
| --- | ---: | ---: | ---: |
| current S3 ~10M | 1,024 | 8 | 7.5 MiB |
| S4 ~100M engineering candidate | 2,048 | 12 | 60 MiB |
| S5 ~400M engineering candidate | 4,096 | 4 | 80 MiB |
| S5 MHA-equivalent comparison | 4,096 | 16 | 320 MiB |

S4/S5 values come from the existing D01 scaling candidates and are planning inputs, not frozen stage configurations. The S5 GQA candidate therefore cuts logical K/V payload by 4x versus the same geometry with 16 KV heads. FP32 payload is 2x the BF16/FP16 figures; batch memory scales linearly before allocator/runtime overhead.

## Semantic acceptance

`tests/test_kv_cache_generation.py` retains the incumbent semantic checks for prompt/decode parity, GQA geometry, fail-closed cache identity/context handling, generation semantics, and checkpoint identity.

`tests/test_kv_cache_serving.py` adds only serving-specific contracts:

- batch-2 equal-length prefill and one-token decode consistency;
- overlapping session lifecycle safety and final mode restoration;
- logical memory estimates matching actual cache payload and expected GQA/batch scaling.

This avoids duplicating the incumbent greedy/sampling and per-step parity matrix.

## Machine evidence

The dedicated exact-head workflow keeps the S0 mechanics oracle:

```text
python tools/run_s0_kv_cache_evidence.py \
  --repo-root . \
  --source-sha "$SOURCE_SHA" \
  --steps 16 \
  --output s0-kv-cache-evidence.json
```

For 16 decode-logit evaluations after a 4-token prompt, the stateless path processes `4+5+...+19 = 184` decoder input positions. The cache path needs the 4-position prefill plus 15 one-token appends = 19 positions. This is an algorithmic work metric, not a latency claim.

The workflow also executes the real current S3 ~10M model on CPU:

```text
python tools/run_s3_kv_cache_cpu_benchmark.py \
  --repo-root . \
  --source-sha "$SOURCE_SHA" \
  --prompt-length 64 \
  --decode-steps 8 \
  --repeats 3 \
  --threads 1 \
  --output s3-kv-cache-cpu-benchmark.json
```

It alternates measurement order after warmup and reports medians for full-prefix and cached incremental decode. The workflow records the observed ratio rather than gating on a required speedup because a GitHub-hosted CPU is engineering evidence, not serving-capacity evidence. GPU latency and throughput remain unmeasured until an accelerator run is actually executed.

## Next scale boundary

The current Python cache grows contiguous K/V tensors with `torch.cat` on every one-token decode. That preserves simple exact semantics but still copies the growing cache and is not the target allocator for high-concurrency long-context service. The next first-party optimization should be a bounded preallocated/static cache or a maintained runtime integration, measured on accelerator hardware before adding more scheduling machinery.

Paged/block allocation, continuous batching, prefix sharing, eviction and distributed serving should not be rebuilt here merely to extend the S0/S3 runtime. The existing vLLM handoff is the intended larger-serving seam when its artifact/runtime parity prerequisites are met; vLLM can own paged attention and scheduler-level serving mechanics.

## Boundaries

This package does not change Base training behavior, tokenizer IDs, checkpoint format, sampling/stop semantics, HTTP request semantics or stage promotion. It does not claim ragged/continuous batching, paged KV cache, prefix sharing, speculative decoding, quantized cache, GPU latency, distributed serving, vLLM/llama.cpp runtime parity, public-server capacity/SLA, materially paid compute, AUDIT-A/B PASS, CANDIDATE or STABLE status.
