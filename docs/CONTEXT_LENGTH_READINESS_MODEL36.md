# MODEL-36 context-length engineering readiness

Worker: `MODEL-36-CONTEXT`

Base architecture audited: `bd11d6a51234027dc82ef55cc09a7128ecf3a074` (`d07/kv-cache-incremental-generation-20260825`).

## Claim boundary

This work establishes engineering readiness and qualification gates. A `max_seq_len` value, a mechanically successful forward pass, RoPE mathematics at a position, or a KV-cache allocation does **not** establish trained or evaluated long-context capability. Capability requires training at the candidate context and held-out evaluation that depends on the additional context.

Canonical S0 and existing stage configs are not changed by this lane.

## Live architecture audit

### RoPE

`ModelSpec` carries `position_embedding="rope"`, `rope_theta`, and explicit `rope_rotary_dim`. The rotary dimension must be positive, even, and no larger than `head_dim`. Current S0-S3 stage configs all use full rotary dimensions:

| Stage | head_dim | rope_rotary_dim | fraction | theta |
| --- | ---: | ---: | ---: | ---: |
| S0 | 10 | 10 | 1.0 | 10000 |
| S1 | 12 | 12 | 1.0 | 10000 |
| S2 | 32 | 32 | 1.0 | 10000 |
| S3 | 40 | 40 | 1.0 | 10000 |

The implementation rotates adjacent dimension pairs. `RotaryEmbedding.cos_sin()` accepts an absolute `position_offset`; incremental decode uses the cached sequence length as that offset. Added tests verify that an offset slice equals the same positions from a full RoPE table.

Partial rotary is already mechanically supported: `apply_rope()` rotates the first `rope_rotary_dim` dimensions and copies the remainder unchanged. Added tests verify that behavior. No current stage uses partial rotary, and this work does not recommend it as a context-extension shortcut. Changing rotary fraction, theta, or a future RoPE-scaling method is a distinct `ModelSpec` experiment and requires training/evaluation evidence.

The RoFormer paper motivates RoPE as a rotational positional encoding with relative-position structure, but mathematical flexibility in sequence length is not evidence that a checkpoint trained at one maximum context generalizes at a larger one.

### Attention

Attention uses `torch.nn.functional.scaled_dot_product_attention`. Dense self-attention arithmetic remains quadratic in sequence length. On CUDA, current PyTorch can dispatch SDPA to FlashAttention-2, memory-efficient attention, or a math implementation depending on inputs/backend; CPU uses the non-CUDA implementation. Therefore a classical `B*L*H*S*S` score tensor is useful as a risk-equivalent planning term but is not a valid claim about actual fused-kernel HBM allocation.

The previous distributed memory estimator modeled activations only as a linear `B*S*hidden*layers` term. That made context scaling systematically optimistic. It is now extended with:

- `linear_activation_bytes_per_rank`;
- `attention_score_equivalent_bytes_per_rank`;
- default `attention_memory_mode="materialized_score_equivalent"`;
- explicit `linear_only` mode only for historical comparison.

The score-equivalent term scales by 4x when context doubles. Target-GPU allocator telemetry remains mandatory before capacity or paid-compute claims.

### KV cache and generation

The incumbent #138 implementation stores **unexpanded** K/V tensors per layer with shape:

`[batch, n_kv_heads, sequence, head_dim]`.

Exact cache bytes are therefore:

`2 * batch * layers * n_kv_heads * sequence * head_dim * element_bytes`.

The high-level generation path already:

- rejects prompts above `max_context_tokens`;
- creates an incremental generation session when supported;
- uses model-native prefill plus one-token decode;
- stops with `context_limit` at the model boundary;
- binds the cache to `ModelSpec` identity;
- rejects decode when the cache is already at `max_seq_len`.

MODEL-36 does not introduce a parallel decoder or cache implementation. The repository-native probe now prefills `S-1` tokens and decodes the final token at absolute RoPE position `S-1`, so the boundary path is exercised on the incumbent cache implementation.

### Packing and document lengths

Canonical S0 packing is deliberately identity-locked to 128 tokens, isolates documents, has no invented BOS/EOS, forbids cross-document packing without a semantic EOS, and overlaps windows by one token so every within-document next-token pair is represented exactly once.

A hidden future-stage blocker existed: the generic packer accepts a different `sequence_length`, but `measure_packed_split()` correctly refuses to produce a canonical S0 manifest for any length other than 128. Without another identity contract, a larger-context experiment could mechanically pack data but could not produce a truthful packing manifest.

MODEL-36 preserves that S0 fail-closed behavior and adds `ContextPackingSpec` version `context-candidate-isolated-v1` plus `measure_context_candidate_packing()`. Future-stage candidate evidence now binds:

- dataset identity and source hash;
- tokenizer config/vocabulary identities;
- candidate sequence length and packing hash;
- document token-length min/p50/p90/p95/max;
- token count and causal loss-token count;
- packed capacity and causal-pair utilization.

Cross-document packing is still forbidden in this v1 candidate contract because the current byte tokenizer has no semantic EOS. A future tokenizer with explicit boundary semantics should introduce a new packing version rather than silently changing this one.

## S0 data evidence

The controlled S0 fixture contains 12 documents and 2,326 UTF-8 byte tokens. Exact document byte-token lengths are:

`110, 134, 138, 141, 143, 145, 237, 237, 251, 258, 263, 269`.

Median is about 191 tokens and maximum is 269. There are 2,314 unique within-document next-token pairs.

Under the current isolated-document, one-token-overlap policy:

| Candidate block | Blocks | Causal-pair capacity | Useful pairs | Utilization |
| --- | ---: | ---: | ---: | ---: |
| 128 | 26 | 3,302 | 2,314 | 70.1% |
| 256 | 15 | 3,825 | 2,314 | 60.5% |
| 512 | 12 | 6,132 | 2,314 | 37.7% |
| 1,024 | 12 | 12,276 | 2,314 | 18.9% |
| 4,096 | 12 | 49,140 | 2,314 | 4.7% |

This is strong evidence **against** promoting S0 itself to 1K/4K. Larger S0 contexts are useful only as mechanics probes.

## Context-dependent memory arithmetic

For batch 1 and bf16/fp16-like 2-byte elements, representative model-native KV cache and one dense score-tensor equivalent are:

| Stage | Context | KV cache | Dense score equivalent |
| --- | ---: | ---: | ---: |
| S1 | 256 | 0.141 MiB | 1.5 MiB |
| S1 | 1K | 0.563 MiB | 24 MiB |
| S1 | 4K | 2.25 MiB | 384 MiB |
| S2 | 512 | 1 MiB | 8 MiB |
| S2 | 1K | 2 MiB | 32 MiB |
| S2 | 4K | 8 MiB | 512 MiB |
| S3 | 1K | 7.5 MiB | 96 MiB |
| S3 | 2K | 15 MiB | 384 MiB |
| S3 | 4K | 30 MiB | 1.5 GiB |

The dense score equivalent is a planning diagnostic, not measured fused-SDPA peak memory. It demonstrates why a compute plan must not extrapolate context cost using only linear activation coefficients.

## Executed local mechanics probes

Environment: PyTorch `2.10.0+cpu`, CPU, 5 Torch threads, batch 1, fp32 parameters/activations, seed 126, no compile. The local script mirrored the audited ModelSpec geometries, RMSNorm, SwiGLU, adjacent-pair RoPE, PyTorch SDPA, residual initialization, and unexpanded cache semantics. The repo-native exact-import probe is committed separately for CI/target-GPU reruns.

These numbers are **not GPU throughput evidence**. CPU kernel selection, warmup and scheduling make individual timings noisy. The useful evidence is mechanical execution, memory trend, and exact cache accounting.

### One random-initialized training step

| Stage | Params | Context | Forward + backward | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| S0 | 10,140 | 128 | 0.0040 s | 366 MiB |
| S0 | 10,140 | 1K | 0.0179 s | 373 MiB |
| S1 | 107,856 | 256 | 0.0154 s | 373 MiB |
| S1 | 107,856 | 512 | 0.0329 s | 380 MiB |
| S1 | 107,856 | 1K | 0.0513 s | 391 MiB |
| S1 | 107,856 | 2K | 0.1574 s | 412 MiB |
| S1 | 107,856 | 4K | 0.3551 s | 459 MiB |
| S2 | 1,066,112 | 512 | 0.0716 s | 415 MiB |
| S2 | 1,066,112 | 1K | 0.1348 s | 460 MiB |
| S2 | 1,066,112 | 2K | 0.4041 s | 548 MiB |
| S2 | 1,066,112 | 4K | 0.8640 s | 725 MiB |
| S3 | 10,059,840 | 1K | 0.6635 s | 748 MiB |
| S3 | 10,059,840 | 2K | 1.3136 s | 1,084 MiB |

Random-init loss and gradient norms were recorded by the probe but are not used as long-context quality evidence.

### KV cache tensor accounting

Observed fp32 cache bytes matched the geometry formula exactly at every measured point:

| Stage | Context | Observed fp32 cache | Formula |
| --- | ---: | ---: | ---: |
| S1 | 256 | 294,912 B | 294,912 B |
| S1 | 1K | 1,179,648 B | 1,179,648 B |
| S1 | 4K | 4,718,592 B | 4,718,592 B |
| S2 | 512 | 2,097,152 B | 2,097,152 B |
| S2 | 1K | 4,194,304 B | 4,194,304 B |
| S2 | 4K | 16,777,216 B | 16,777,216 B |
| S3 | 1K | 15,728,640 B | 15,728,640 B |
| S3 | 2K | 31,457,280 B | 31,457,280 B |

For bf16/fp16 cache storage those byte counts are exactly half, assuming the cache tensor dtype is 2 bytes.

## Stage-triggered recommendation

The machine-readable policy is `configs/context/context_readiness_v1.json`.

### S0 / 10K

Preserve canonical 128. Do not promote on current fixture. 256/512/1K remain mechanics-only probes.

### S1 / ~100K

Keep canonical 256. The next executable qualification candidate is 512:

`configs/context/s1_100k_context_512.experimental.json`.

Promote only after the intended S1 corpus has candidate packing/document-length measurements and short training/evaluation does not regress while target hardware remains comfortable.

### S2 / ~1M

Keep canonical 512 until qualification. The primary next candidate is 1K:

`configs/context/s2_1m_context_1024.experimental.json`.

2K/4K should not leapfrog the 1K data, training-memory and evaluation gates.

### S3 / ~10M

Canonical 1K is sensible as the first context at this scale. The next candidate is 2K, and target-GPU evidence is required:

`configs/context/s3_10m_context_2048.experimental.json`.

4K exists only as a gated research config:

`configs/context/s3_10m_context_4096.research.json`.

Its status is intentionally `RESEARCH_ONLY_UNQUALIFIED`.

### 100M+ / future S4+

4K is a candidate, not a default. Do not select beyond 4K until 4K itself passes all of:

1. intended-corpus document-length and packing-utilization measurement;
2. target-GPU SDPA backend identification and peak-HBM measurement at intended microbatch;
3. stable training at 4K;
4. held-out loss stratified by document length;
5. a long-document or distance-sensitive evaluation that actually requires the added context;
6. inference prefill/decode/cache measurements at the same ModelSpec identity.

Only then should RoPE theta/scaling variants or >4K contexts be considered. A new RoPE scaling method must be versioned in `ModelSpec`; it is not a JSON-only increase to `max_seq_len`.

## Integration with memory/compute planning

Any future compute plan should consume both layers of context accounting:

- `estimate_training_memory()` for model state, linear activations and the conservative dense score-equivalent term;
- `estimate_context_cost()` for exact ModelSpec-derived KV-cache bytes and attention score-equivalent bytes.

Before a paid run, replace conservative planning coefficients with profiler evidence from the exact target GPU, precision, SDPA backend, context, microbatch, gradient/checkpointing policy and parallel topology. Context parallelism can reduce per-rank activation pressure at larger contexts, but it should be introduced only after single-device/fused-attention evidence shows it is needed; it is not free compute.

## Code added or changed

- `src/twelve_six/context_scaling.py`: context cost model, versioned candidate packing measurement, identity-distinct probe specs.
- `src/twelve_six/distributed/memory.py`: exposes quadratic dense-attention score-equivalent planning term.
- `tools/run_context_scaling_probe.py`: exact-repo CPU/GPU mechanics probe using current stage configs and incumbent KV cache.
- `tests/test_context_scaling.py`: RoPE offsets, partial rotary, memory scaling, candidate packing, mechanics.
- `tests/test_context_candidate_configs.py`: candidate config identities and canonical-stage preservation.
- `configs/context/*`: stage-triggered policy and executable experimental ModelSpecs.

## Primary references

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135.
- PyTorch documentation, `torch.nn.functional.scaled_dot_product_attention`, current documentation consulted 2026-08-25.
- PyTorch documentation, `torch.nn.attention`, updated 2026-05-07.
