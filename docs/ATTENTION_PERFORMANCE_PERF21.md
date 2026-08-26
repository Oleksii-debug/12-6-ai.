# PERF-21: SDPA attention performance audit and native-GQA seam

Status: engineering implementation and benchmark package. No GPU execution claim.

## Decision

Keep PyTorch scaled-dot-product attention (SDPA) as the canonical attention primitive. Do not add a custom attention kernel.

The current MHA path is already arranged in a kernel-friendly way: Q/K/V are shaped as `[batch, heads, sequence, head_dim]`, the inner head dimension remains contiguous after the projection transpose, no dense causal mask is materialized, and training/prefill use `is_causal=True`. The incumbent KV-cache PR #138 correctly uses `is_causal=False` for one-token decode because the newest query may attend every cached key; non-square `is_causal=True` would encode different semantics.

The concrete scale defect is GQA. Current canonical attention, including PR #138, expands K/V with `repeat_interleave` before SDPA. That is mathematically correct but materializes grouped K/V at query-head width. S5 has 16 query heads and 4 KV heads, so this creates a 4x K/V materialization before attention. PR #138 retains the cache itself at 4 KV heads, which is the correct cache identity, but `_attend()` expands the growing cache on every decode call.

`src/twelve_six/attention_perf.py` therefore supplies two explicit paths:

- `sdpa_expanded_reference`: the current 12-6 semantics, retained as the parity/benchmark reference;
- `sdpa_native_gqa`: passes unexpanded K/V to PyTorch SDPA and uses `enable_gqa=True` only when query and KV head counts differ.

The helper validates head geometry, dtype, device and shape before calling SDPA. It adds no parameters, no buffers, no ModelSpec/InitSpec/checkpoint state and no separate KV cache.

## Audit

| Area | Finding | Action |
| --- | --- | --- |
| Q/K/V layout | `[B,H,S,D]` via transpose; tensors are not globally contiguous but `stride(-1)==1` | Keep. Do not insert unconditional `.contiguous()` copies before SDPA. |
| Output layout | `.transpose(1,2).contiguous().view(...)` before output projection | Keep; this copy is required for the flattened projection view. |
| Causal mask | No explicit `attn_mask`; training/prefill use `is_causal=True` | Keep. Avoid allocating a dense triangular mask. |
| Cached decode | One query against all cached/current keys uses `is_causal=False` in PR #138 | Keep. This is the correct rectangular-attention semantic. |
| GQA | K/V are manually repeated to query-head count | Replace with native SDPA GQA after runtime lock acceptance. Highest-value change for S5+. |
| Dtype | Projection/autocast dtype flows into SDPA; eval passes dropout 0 | Keep. Do not force Q/K/V to fp32. |
| RoPE | Applied to Q/K before SDPA; trig basis is computed in fp32 then cast | Keep semantics. Per-layer cos/sin recomputation is a secondary profiling target, not a PERF-21 rewrite. |
| KV cache | PR #138 stores unexpanded K/V at `n_kv_heads` | Keep one incumbent cache. Native GQA should consume these tensors directly. |
| `torch.compile` | SDPA helper full-graph capture passes in the local CPU probe | Keep tensor-only seam. Re-test the exact integrated model under the locked runtime/GPU. |
| Kernel selection | PyTorch SDPA dispatcher can choose optimized CUDA implementations | Production should use auto dispatch; force Flash/Efficient/cuDNN/Math only in diagnostics. |

## Scale impact

S3 (~10M) uses MHA 8/8, head dimension 40, max context 1024. Native GQA does not reduce K/V width because there is no grouping. The unusual head dimension 40 should be checked against real GPU dispatcher eligibility before any architecture change; this audit does not change S3 geometry.

S4 (~100M engineering candidate) uses MHA 12/12, head dimension 64, max context 2048. Again there is no GQA K/V reduction. The current mask/layout arrangement is already appropriate for fused SDPA dispatch, so the next evidence is a GPU backend/throughput run rather than another code rewrite.

S5 (~400M engineering candidate) uses GQA 16/4, head dimension 64, max context 4096. This is the first stage where the current repeat path is a material issue. For batch 1 at full context:

- BF16 unexpanded K+V: 4 MiB per layer;
- BF16 materialized expanded K+V: 16 MiB per layer;
- avoidable expansion ratio: 4x.

The temporary does not imply all layers retain that expanded memory simultaneously, but it creates avoidable allocation/memory traffic for every attention call. During incremental decode, PR #138 would repeat the growing cached K/V every token; native GQA removes that Python-level materialization while preserving the unexpanded cache.

## Local evidence

Execution environment: Python 3.13.5, PyTorch 2.10.0+cpu, CUDA unavailable. This is dispatcher/correctness evidence only, not GPU throughput, MFU or capacity evidence.

Focused helper suite: 7 passed. It covers MHA exact parity, GQA forward/backward parity, cached one-token noncausal semantics, canonical non-contiguous transpose layout, S5 byte accounting, fail-closed geometry, and `torch.compile(..., fullgraph=True)` capture. The two tests that import the full PR #138 model are intended for exact branch CI because the local sandbox cannot clone the repository.

Representative CPU forward benchmark, batch 1, fp32, sequence 512:

| Stage geometry | Expanded reference median | Native path median | Native speedup | K/V materialization ratio |
| --- | ---: | ---: | ---: | ---: |
| S3 8/8, Dh40 | 1.714 ms | 1.499 ms | 1.14x | 1x |
| S4 12/12, Dh64 | 2.902 ms | 2.953 ms | 0.98x | 1x |
| S5 16/4, Dh64 | 4.999 ms | 3.507 ms | 1.43x | 4x |

S5 fp32 sequence 1024: expanded 13.096 ms, native 11.690 ms, 1.12x. S5 sequence 128 forward+backward: expanded 2.160 ms, native 1.684 ms, 1.28x. These timings are machine-specific CPU observations and are not extrapolated to GPU.

## Executable GPU benchmark

`tools/benchmark_attention_sdpa.py` benchmarks the expanded reference and native-GQA path for S3/S4/S5 geometry. Auto dispatch is the production-like mode. Backend forcing exists only to expose eligibility/failure reasons.

Examples:

```bash
python tools/benchmark_attention_sdpa.py \
  --stage s5 --device cuda --dtype bfloat16 \
  --phase forward-backward --sequence-length 4096 \
  --warmup 10 --iterations 50 --compile-check
```

```bash
python tools/benchmark_attention_sdpa.py \
  --stage s5 --device cuda --dtype bfloat16 \
  --backend flash --sequence-length 4096
```

Run the second command separately with `--backend efficient`, `--backend cudnn`, and `--backend math` when diagnosing dispatch. A forced fused backend failure is evidence that the exact shape/dtype/runtime is unsupported; it is not a reason to add a custom kernel immediately.

## Integration boundary with PR #138

This package is stacked on PR #138 rather than introducing another cache implementation. The exact integration point is `CausalSelfAttention._attend()`: replace its `_expand_kv()` + direct SDPA call with `sdpa_native_gqa(q, k, v, dropout_p=..., is_causal=...)`. The new integration tests prove the helper matches #138 training/prefill behavior and its single-token cache decode semantics.

Do not merge a second KV-cache path. If #138 moves, rebase this seam onto its accepted exact head and rerun parity plus the benchmark.

## Runtime floor and acceptance gate

The package does not silently raise the project dependency floor. The repository currently declares `torch>=2.5`, while current maintained PyTorch documentation exposes native GQA and multiple CUDA SDPA backends. Exact integration must therefore be accepted only on the D08 hash-locked runtime actually used for the target GPU stage.

Before making native GQA the canonical S5+ path, require:

1. exact-head helper and #138 integration tests green;
2. locked CUDA runtime with `enable_gqa` available;
3. S5 BF16 forward+backward benchmark at configured geometry;
4. auto-dispatch result plus forced-backend diagnostics;
5. finite loss/gradient/update evidence through the actual Trainer;
6. cached generation parity from #138 after the same integration;
7. no ModelSpec, state-dict, checkpoint, random-init or tokenizer identity change.

If the locked runtime cannot execute native GQA for the target backend, keep the expanded reference path and treat the runtime/backend upgrade as the next decision. Do not hide the fallback inside an unmeasured custom kernel.
