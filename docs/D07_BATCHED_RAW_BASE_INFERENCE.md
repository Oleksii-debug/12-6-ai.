# D07 batched raw-Base inference

## Scope and collision boundary

This package is based directly on the terminal-green S0 repeatability source and adds a lower-level batching seam without taking ownership of adjacent active surfaces.

It does not modify the canonical decoder, scalar generation loop, sampling/contracts, local HTTP listener, checkpoint loader/export, tokenizer, or KV-cache incumbent. The existing KV-cache PR owns model-native incremental cache mechanics. The server-hardening PR owns the current listener. This package is intentionally consumable by a future request-coalescing server without introducing a second server implementation now.

No chat template, system prompt, instruction behavior, alignment behavior, or foreign pretrained weights are added. The interface remains raw Base text completion.

## API

`BatchedInferenceBackend` extends the existing inference backend contract with one operation:

`next_token_logits_batch(input_ids)`

Each row is one independent causal prefix. `generate_batch()` accepts independent `BatchGenerationRequest` values and coalesces active requests into model microbatches. Requests keep their own:

- prompt/token sequence;
- `GenerationConfig`;
- `random.Random(seed)` state;
- max-new-token budget;
- EOS, stop-token and stop-string termination state;
- context-limit state.

Results are returned in request order even though model scheduling sorts active prefixes by length.

## Heterogeneous prompt lengths: right padding without semantic padding

The canonical decoder has no padding-token semantics and its current forward API does not accept an attention mask. The first credible stateless batch path therefore uses **right padding only** and gathers each row's next-token logits at its last real-token position.

For a causal decoder this is semantically safe for the gathered real position: all filler positions are strictly in its future, so causal attention prevents the real position from attending to them. Real-token RoPE positions also remain unchanged. The filler ID is an ordinary in-vocabulary token used only to materialize a rectangular tensor; it is never decoded, exposed as a special token, or inserted into the logical request prefix.

Focused regression compares heterogeneous right-padded batch logits against independent canonical forwards for each prefix.

Padding cost is explicit rather than hidden. The scheduler records logical positions and right-padding positions and supports `max_padding_tokens`. Setting that limit to zero batches only equal-length prefixes.

## Prefill, decode and KV-cache boundary

This implementation is stateless full-prefix batching. On every generation round it batches the complete current prefix for each active request. It therefore provides real model-call coalescing, but it does **not** claim KV-cached decode.

That boundary is deliberate. The current model-native KV-cache incumbent has one uniform cache sequence length per batch. Right-padding heterogeneous prompts and retaining those filler positions in cache would change subsequent absolute positions and attention history. Requests also finish at different times, which requires cache-row selection/compaction before an efficient decode batch can shrink safely.

A later cache-aware batching layer should consume a repaired/accepted KV-cache incumbent and add explicit row lifecycle plus one of these semantics-preserving prefill strategies:

- length-bucketed prefill with exact-length cache rows;
- explicit attention/position metadata that the canonical model supports end to end;
- maintained serving-backend cache management at larger scale.

This package does not build paged attention. vLLM remains the preferred larger-scale ownership boundary once exported/runtime compatibility is proven.

## Completion lifecycle and seed semantics

Generation proceeds in rounds. Before each round, every request independently fails closed or finishes on max-new-token/context state. Active requests are length-bucketed into microbatches. One next token is selected for each active row using the existing greedy or sampling functions.

Sampling RNG is per request, initialized exactly from that request's seed. Batch order, batch size, a neighbor finishing early, and scheduler bucketing therefore do not consume another request's random stream. Regression compares batched sampled outputs against sequential `generate()` request by request.

EOS, stop-token and stop-string checks preserve the scalar generation order. A request that stops is removed from later model batches without affecting surviving requests.

## Torch adapter and memory accounting

`right_padded_next_token_logits()` is a lower-level canonical-model adapter. It reports:

- batch size and min/max sequence length;
- logical input positions;
- padded input positions and right-padding waste;
- actual input tensor bytes;
- actual full model output-logit tensor bytes.

The last item matters for future 10M/100M serving: the current decoder materializes logits for every padded sequence position even though generation consumes only the last real position per row. That is a known memory/bandwidth inefficiency and is exposed rather than hidden. A future inference-only last-token projection seam can remove that waste if profiling justifies it without changing training semantics.

No KV-cache bytes are reported by this path because it does not use KV caching.

## Benchmark evidence

`tools/benchmark_batched_inference.py` measures independent canonical forwards versus one heterogeneous batch on local CPU for:

- canonical S0, 10,140 parameters;
- canonical S3, 10,059,840 parameters.

The benchmark first requires max-absolute logits agreement within `1e-4`, then records median latency, requests/second, observed speedup, padding cost and tensor memory. The tolerance covers normal floating-point kernel-dispatch drift between batch-1 and batch-N GEMM/attention execution; token-selection and stopping parity remain separately regression-tested. The benchmark does not require speedup to be greater than one: CPU results are recorded as observed, not converted into a serving-throughput claim.

The S3 result is the first measured step toward the requested 10M scale. A 100M-class accelerator benchmark remains future evidence because no free GPU execution is claimed here and CPU batching behavior should not be presented as GPU-serving capacity.

## Truth boundary

This package proves a lower-level raw-Base batching API and CPU mechanics only when its exact-head tests and benchmark workflow pass. It does not claim public-server throughput, SLA, GPU performance, KV-cache batching, prefix sharing, paged attention, distributed serving, streaming, speculative decoding, Transformers/vLLM runtime parity, Windows/NVDA execution, audit PASS, CANDIDATE, or STABLE status.
