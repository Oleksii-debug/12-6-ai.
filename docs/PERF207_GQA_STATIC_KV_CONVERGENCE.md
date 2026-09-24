# PERF-207 GQA + static KV convergence

Status: `BLOCKED_PARENT_NOT_ACCEPTED`

This worker does **not** change `TwelveSixDecoder`, model weights, generation semantics, cache format, checkpoint format, tokenizer identity, or serving APIs. It does not implement paged attention.

## Exact inputs reconstructed

- Current D07 cached-batching incumbent: `perf/batched-raw-base-20260825@3b89506958f9d968092dfe99581263c428b395d0` (PR #166). Exact-head CI, D07 KV-cache, and D07 batched-inference workflows are terminal green.
- PERF-94 native-GQA candidate: `perf94/native-gqa-production-20260826@a65c85522f3dbe826862c2f156720d5ceadadfc0` (PR #261). PR remains open/unmerged. Exact-head CI and D07 workflows are green. Its grouped-attention production path calls the existing `sdpa_native_gqa(...)` helper with unexpanded K/V, while MHA remains on direct SDPA.
- PERF-95 named branch: `perf95/static-kv-cache-20260826@75a885e221fdf16f647289df25a7150ef46e7528`. No PERF-95 static-cache PR/evidence gate exists. Relative to the D07 cache head `8f20b5c81c83796492353de3b68e2679b54fc980`, this branch contains MODEL-35 GQA-geometry experiment/config/workflow changes, not a static/preallocated KV-cache successor.

## Blocking code fact

Both the PERF-94 production-GQA head and the branch named PERF-95 still grow each layer cache with:

```python
key = torch.cat((cache.key, new_k), dim=2)
value = torch.cat((cache.value, new_v), dim=2)
```

Therefore the required accepted static-path invariant, **no `torch.cat` growth**, is not satisfied. A convergence patch that pretended otherwise would create a new cache implementation inside PERF-207, violating the instruction to consume PERF-95 rather than replace it.

The current D07 batched-cache PR also explicitly records that its accepted incumbent uses contiguous `torch.cat`, and that PERF-95 had no accepted static-cache evidence gate at that integration point.

## Semantic gates

No combined native-GQA + static-cache path exists, so PERF-207 does not claim combined-path PASS for greedy, seeded sampling, context-boundary, batched cached, EOS, stop-token, or stop-string parity.

The current D07 batching incumbent independently retains exact greedy cached-batch parity on its exact green head and its test contract covers seeded sampling, EOS/stop, and context-boundary behavior. PERF-94 independently retains cached-generation parity while changing only the grouped-attention kernel route. Those parent facts are not promoted into a nonexistent combined-path result.

Because PERF-207 makes no model/runtime mutation, checkpoint/model identity on this worker branch is unchanged by construction.

## Available-hardware benchmark evidence

Only free GitHub-hosted CPU evidence is available in the retained exact-head artifacts. PyTorch is `2.13.0+cu130`, Python `3.11.16`; the benchmark device is `cpu`. Installed CUDA libraries do not imply a visible CUDA device.

### ~1M first-party generation — GQA geometry experiment

Source: MODEL-35 exact-head artifact at `75a885e221fdf16f647289df25a7150ef46e7528`, S2-GQA2 candidate.

- Parameters: 1,066,624.
- Geometry: 4 query heads / 2 KV heads, head_dim 32.
- Prompt: 64 tokens; generated tokens: 6; final cache sequence length: 69.
- Greedy cached/stateless token parity: exact on both retained seeds.
- Actual FP32 K/V tensor payload at sequence 69: 141,312 bytes.
- Full-context K/V payload: 524,288 bytes BF16/FP16; 1,048,576 bytes FP32.
- Mean cached generation time: 13.4867 ms for the six-token generation operation.
- Mean stateless generation time: 21.1776 ms.
- Mean cached throughput: 444.895 generated tokens/s; stateless: 283.321 generated tokens/s.
- Decoder input work: 69 cached positions versus 399 stateless full-prefix positions.
- CPU truth boundary: this MODEL-35 branch uses its CPU expanded-reference fallback for grouped attention; it does not constitute native-GQA CUDA performance evidence.

### ~10M first-party generation — GQA geometry experiment

Source: the same exact-head MODEL-35 artifact, S3-GQA4 candidate.

- Parameters: 10,061,760.
- Geometry: 8 query heads / 4 KV heads, head_dim 40.
- Prompt: 48 tokens; generated tokens: 6; final cache sequence length: 53.
- Greedy cached/stateless token parity: exact on both retained seeds.
- Actual FP32 K/V tensor payload at sequence 53: 407,040 bytes.
- Full-context K/V payload: 3,932,160 bytes BF16/FP16; 7,864,320 bytes FP32.
- Mean cached generation time: 42.6030 ms for the six-token generation operation.
- Mean stateless generation time: 85.1916 ms.
- Mean cached throughput: 140.877 generated tokens/s; stateless: 70.433 generated tokens/s.
- Decoder input work: 53 cached positions versus 303 stateless full-prefix positions.
- CPU truth boundary: this is not native-GQA CUDA performance evidence.

### Current ~10M D07 cached batching incumbent

Source: exact-head D07 batched-inference artifact at `3b89506958f9d968092dfe99581263c428b395d0`.

- Canonical S3 parameters: 10,059,840.
- Batch size: 4; prompt length: 64; generation budgets: 8/4/6/8.
- Greedy results: exact versus four independent cached requests.
- Batched peak logical K/V payload: 4,362,240 bytes FP32.
- Independent-concurrent peak logical K/V payload: 4,270,080 bytes FP32.
- Batched cached median latency: 140.370 ms; independent cached median: 222.983 ms.
- Model calls: 8 batched cached calls versus 26 independent cached calls, 3.25x reduction.
- Scheduled cached input positions: 284; logical cached positions: 278; independent stateless input positions: 1,741.
- This cache metric is logical tensor payload, not allocator-reserved/peak memory.

## Required measurements that remain NOT RUN

- Static/preallocated cache allocation-growth measurement: `NOT_RUN_PARENT_ABSENT`.
- Pure decode-only timing for a static/preallocated path: `NOT_RUN_PARENT_ABSENT`.
- Combined-path greedy parity: `NOT_RUN_PARENT_ABSENT`.
- Combined-path seeded sampling parity: `NOT_RUN_PARENT_ABSENT`.
- Combined-path context-boundary parity: `NOT_RUN_PARENT_ABSENT`.
- Combined-path cached-batch parity: `NOT_RUN_PARENT_ABSENT`.
- Combined-path EOS/stop parity: `NOT_RUN_PARENT_ABSENT`.
- Combined-path CUDA benchmark: `NOT_RUN_NO_AUTHORIZED_VISIBLE_GPU_EVIDENCE`.

## Convergence decision

`DO_NOT_CONVERGE`.

PERF-207 preserves native GQA and static/preallocated KV work as experimental parents. The production/first-party runtime remains on the current D07 batching/cache incumbent. No global switch or cache replacement is made.

A future PERF-207 successor may converge only after a real PERF-95 static/preallocated cache parent exists and passes, on one exact head:

1. fixed-capacity K/V storage at `[batch, n_kv_heads, max_context, head_dim]` or an equivalent bounded first-party representation;
2. cursor/slice append with no decode-time `torch.cat` growth;
3. exact greedy and seeded-sampling parity;
4. context-boundary, EOS, stop-token, stop-string, and batched-cached parity;
5. unexpanded K/V through cache storage and grouped attention;
6. model/InitSpec/state-dict/checkpoint identity invariance;
7. measured cache payload plus allocator growth/peak and decode-only time at ~1M and ~10M under universal bootstrap;
8. target-CUDA evidence only when a free/authorized visible GPU actually exists.
