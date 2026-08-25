# D07 batched raw-Base inference

## Scope and incumbent composition

This package owns the first-party batching seam. PERF-96 composes the accepted model-native
KV-cache incumbent into this existing batching PR instead of creating a third decoder or
attention implementation.

The branch now contains both accepted predecessor histories:

- the stateless batching seam from this PR;
- the model-native KV-cache seam from the D07 KV-cache PR.

The canonical decoder, ModelSpec, InitSpec, trainable parameters, state dict and checkpoint
identity remain unchanged. The cache is still ephemeral inference state. No chat template,
system prompt, instruction/alignment behavior or foreign pretrained weights are added.

This work is a first-party correctness/performance bridge. It is not an attempt to reproduce
vLLM continuous batching, paged attention, prefix sharing, eviction or distributed serving.

## Stateless batching API

`BatchedInferenceBackend.next_token_logits_batch()` and `generate_batch()` retain the original
heterogeneous full-prefix path. The Torch adapter right-pads only after each row's last real
token and gathers logits at that real position. Causal attention prevents future filler
positions from changing that position, and real-token RoPE indices do not move.

The stateless scheduler preserves request-local `GenerationConfig`, RNG state, max-new-token
budget, EOS, stop-token, stop-string and context-limit semantics. `max_padding_tokens=0`
restricts this path to equal-length full-prefix batches.

## Model-native cached batching API

PERF-96 adds `CachedBatchedInferenceBackend.begin_generation_batch()` and
`generate_batch_cached()` on top of the incumbent `DecoderKVCache` implementation.

The cached scheduler has a deliberately narrow contract:

1. Only requests with exactly equal encoded prompt length may share one KV-cache batch.
2. Heterogeneous requests are partitioned into exact-length buckets and each bucket is further
   bounded by `max_batch_size`.
3. No semantic right padding is inserted into a live cache row.
4. Cache row count remains fixed until that bucket drains.
5. A request that finishes is never sampled again and never consumes its RNG stream.
6. If another row still needs decode, a completed physical row receives an internal ordinary
   in-vocabulary filler token solely to preserve rectangular cache geometry.
7. Completed rows are not compacted or selected out of the cache.

This avoids the correctness risks of ragged cache compaction while still coalescing prompt
prefill and one-token cached decode. The backend/session layer independently rejects ragged
prefill, batch-width mismatch, invalid token IDs, context overflow and closed-session reuse.

The filler token is not decoded, returned, exposed as a tokenizer special token or appended to
the logical request result. It affects only a row whose logical request has already completed;
attention remains row-independent, so surviving rows cannot observe it.

## Completed-request row lifecycle

Each cache bucket performs one model-native prefill. Its resulting last-position logits are the
first token-selection logits for every row. Request-local greedy or sampling logic then runs in
exactly the same order as scalar generation.

Before each later cached decode, only rows that still have another semantic generation step
append their selected token. Rows that have already stopped, exhausted max-new-token budget or
reached their logical context boundary receive the internal filler. Because the batch geometry
is fixed, the physical cache may continue growing for a retired row until the longest-lived row
in that bucket finishes.

The scheduler records this cost explicitly as `retired_row_decode_positions`. It never hides
fixed-row waste as useful work.

## RNG and stop semantics

Every request owns `random.Random(request.config.seed)`. Scheduler order, prompt-length bucket,
batch size and neighboring row completion therefore do not share RNG state.

Focused regressions compare cached batching against independent `generate()` requests for:

- real Torch/model-native greedy generation;
- non-trivial seeded sampling with different request budgets;
- EOS;
- stop-token termination;
- stop-string termination and stripping;
- max-new-token completion;
- context/session lifecycle behavior.

Result token IDs, decoded text and stop reasons are required to match the corresponding
independent requests exactly. Numeric batch-vs-batch-1 logit identity is not promoted as a
contract because maintained PyTorch kernels may dispatch different floating-point kernels; the
observable token/stopping contract is exact.

## Memory and work accounting

`CachedBatchGenerationStats` exposes:

- prefill and decode batch-call counts;
- logical prompt positions;
- logical decode positions that independent cached requests actually require;
- scheduled fixed-row decode positions;
- retired-row decode waste;
- independent cached model-call equivalent;
- independent stateless full-prefix position equivalent;
- maximum batch width;
- observed peak logical K/V tensor payload bytes for one live cache bucket.

`peak_cache_bytes` is the payload represented by the live K/V tensors. It is not process RSS,
allocator-reserved memory or a temporary-allocation peak. The incumbent cache currently grows
K/V with contiguous `torch.cat`; a separate accepted static-cache successor can replace that
allocation policy without changing this scheduler contract.

## CPU evidence

The original `tools/benchmark_batched_inference.py` remains the stateless heterogeneous batching
measurement. `tools/benchmark_cached_batched_inference.py` adds the model-native cached path for
canonical S0 and S3 on the locked free CPU environment.

For each stage the cached benchmark uses four equal-length independent requests with different
generation budgets, requires exact greedy result parity, and records:

- independent cached median latency versus cached-batch median latency;
- independent cached model calls versus batched prefill/decode calls;
- scheduled cached positions versus independent stateless full-prefix positions;
- retired-row decode positions;
- batched peak logical K/V payload;
- the logical concurrent payload of the same requests as separate cache sessions.

No speedup threshold is used for latency. CPU timing is recorded as observed. The workflow does
require a genuine algorithmic reduction in model-call count and full-prefix position work for
the benchmark geometry.

## Truth boundary

Exact-head GitHub Actions are the authority for this package. Until the current source head is
terminal green, no PASS claim should be made.

Even when green, this package establishes only bounded first-party raw-Base batching mechanics
on the measured CPU path. It does not claim public-server throughput or SLA, GPU performance,
continuous/ragged batching, cache compaction, paged attention, prefix sharing, speculative
decoding, distributed serving, vLLM equivalence, audit PASS, CANDIDATE or STABLE promotion.
