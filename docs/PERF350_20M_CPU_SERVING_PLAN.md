# PERF-350 — 20M CPU serving plan

Worker: `PERF-350-20M-CPU-SERVING-PLAN`

Status at authoring: `MECHANICS_MEASUREMENT_PENDING_EXACT_HEAD_RUN`.

## Scope

This worker develops CPU serving defaults for a future learned approximately-20M Base without training one and without changing model, tokenizer, checkpoint, cache, or generation semantics. Execution is `LOCAL_FREE` CPU only. Results are valid only for the exact host fingerprint recorded by the benchmark; no other CPU, Windows host, GPU, or serving SLA may be inferred.

The required learned 20M checkpoint and primary 20M ModelSpec are not published at this worker's source cutoff. Current 20M data authority also authorizes no learned external-real campaign. Therefore numerical systems measurement uses a mechanics-only random-init surrogate and must be rerun on the retained primary 20M ModelSpec/checkpoint when that authority appears.

## Mechanics surrogate

The surrogate reuses the accepted 10M native-GQA serving geometry and changes only depth from 12 to 24 layers:

- vocab 256;
- context 1024;
- d_model 256;
- 24 decoder blocks;
- 8 query heads / 2 KV heads;
- head_dim 32;
- SwiGLU d_ff 864;
- full RoPE dim 32, theta 10000;
- tied embeddings, no attention/MLP/LM-head bias;
- exact trainable parameters: 19,935,488, or -0.32256% from 20,000,000.

This is not an architecture candidate and cannot be promoted by PERF-350. Its sole purpose is to exercise the accepted first-party GQA/static-KV mechanics at approximately the requested parameter scale.

At FP32, full-context unexpanded static K/V storage is exactly 12,582,912 bytes per batch row (12 MiB). This is a formula/mechanics property; process RSS is measured separately.

## Frozen bounded benchmark

Each numerical case runs in a fresh child process so process peak RSS is attributable to that case rather than inherited from earlier sweep cells. Each case builds the same deterministic random-init surrogate, uses one warmup and three measured repetitions, and records median/min/max values.

The sweep is deliberately bounded:

- threads: 1, 2, 4, 8 up to the exact host CPU-affinity limit; when affinity is between grid values, that exact bounded affinity is included up to 8;
- thread selection workload: batch 1, prompt 128, static KV, 32 decode tokens;
- batch: 1, 2, 4 at the selected thread count, prompt 128, static KV, 32 decode tokens;
- prefill: 32, 128, 256, 512 prompt tokens at batch 1, static KV, 16 decode tokens;
- cache: static versus retained dynamic reference at batch 1, prompt 128, 64 decode tokens;
- generation length: 16, 64, 128 decode tokens at batch 1, prompt 128, static KV;
- RSS: baseline high-water after model construction, child-process peak RSS, and incremental peak after model construction for every cell.

Static-cache measurement also proves backing-storage identity and allocated bytes do not change during decode. Dynamic-cache physical growth is measured over the representative trace.

## Default decision contract

The benchmark writes `reports/perf350/cpu-serving-plan.json` with the exact host fingerprint and these host-scoped defaults:

- intra-op threads: highest representative batch-1 static decode throughput; candidates within 5% of best are tied and the lower thread count wins;
- inter-op threads: 1;
- interactive batch: 1. Batch 2/4 are capacity measurements only because this benchmark has no queueing/continuous-batching scheduler and therefore cannot justify a higher interactive default;
- representative prefill planning point: 128 tokens. This is not an input limit;
- KV cache: static by default for fixed storage and zero physical decode growth; dynamic remains the explicit reference/fallback. PERF-350 does not assume a latency win;
- default bounded generation budget: 64 new tokens, with measured 16/64/128 costs. This is an operational CPU occupancy default, not a language-quality optimum;
- context limit: unchanged ModelSpec limit of 1024 tokens.

## Required rerun before learned-20M deployment

When the primary retained 20M ModelSpec and learned checkpoint exist, rerun this exact benchmark on the actual deployment CPU. Any geometry, dtype, context, runtime version, CPU model, CPU-affinity, or memory-host change invalidates the numerical defaults. PERF-350 does not extrapolate from the GitHub-hosted CPU to the owner's Windows machine or any future server.

## Not claimed

No learned 20M quality, no primary architecture selection, no Windows numbers, no GPU numbers, no ragged/continuous batching, no public serving SLA, and no cross-hardware performance prediction.
