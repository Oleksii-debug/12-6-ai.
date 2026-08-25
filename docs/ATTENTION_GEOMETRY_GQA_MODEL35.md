# MODEL-35 grouped-query attention geometry research

## Scope and authority

This package asks one narrow question: at what model stage does reducing K/V head count
become a better engineering default for 12-6 AI than retaining full MHA?

It preserves canonical S0 exactly. It does not freeze S1/S2/S3/S4 architecture, authorize
paid compute, claim model capability, or treat a short synthetic mechanics fixture as
language-model quality evidence.

The experiment is stacked on the D07 model-native KV-cache implementation rather than
creating a second decoder or cache path.

## Implementation audit and scaling fix

`ModelSpec` already represents MHA, GQA and MQA through `n_heads` and `n_kv_heads`.
K/V projection widths and parameter accounting scale with `n_kv_heads`. The D07 cache
also stores unexpanded K/V as:

`[batch, n_kv_heads, sequence, head_dim]`

That part already had the correct GQA memory geometry.

The D07 incumbent path exposed a scaling defect inside `CausalSelfAttention._attend()`:
for GQA/MQA it called `_expand_kv()` and `repeat_interleave()` to materialize K/V back
to query-head width before `scaled_dot_product_attention()`. The persistent cache was
correct, but CUDA attention would still lose grouped K/V geometry at the kernel boundary.

MODEL-35 fixes that defect conservatively. For GQA/MQA tensors on CUDA, `_attend()` now
passes the unexpanded K/V tensors directly to PyTorch SDPA with `enable_gqa=True`. MHA
is unchanged. CPU and other non-CUDA backends retain the established explicit-repeat
reference path instead of relying on an undocumented native-GQA backend contract.

This routing is deliberate. The exact locked runtime is Torch 2.13.0, whose SDPA API
exposes `enable_gqa=True`, but PyTorch documents GQA as experimental and documents
support for FlashAttention, cuDNN attention, math attention on CUDA tensors, and
memory-efficient attention on NVIDIA CUDA. Therefore the patch removes the known CUDA
materialization defect without pretending that every backend has the same native-GQA
support. CPU tests verify the fallback and device-routing contract; target-GPU evidence
is still required before any throughput or memory-performance claim.

Primary background:
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints", arXiv:2305.13245.
- PyTorch 2.13 `torch.nn.functional.scaled_dot_product_attention`, `enable_gqa=True`.

These sources motivate and constrain the implementation. They do not decide 12-6 model
quality or stage geometry.

## Matched parameter candidates

The comparisons keep total parameter count equal or within 0.05%, reallocating K/V
projection savings into SwiGLU `d_ff` instead of simply making the GQA model smaller.

| Stage | Geometry | d_ff | Parameters | Delta vs current MHA |
| --- | ---: | ---: | ---: | ---: |
| S2 current | 4Q / 4KV | 352 | 1,066,112 | baseline |
| S2 GQA | 4Q / 2KV | 395 | 1,066,624 | +0.0480% |
| S2 MQA | 4Q / 1KV | 416 | 1,066,112 | exact |
| S3 current | 8Q / 8KV | 864 | 10,059,840 | baseline |
| S3 GQA | 8Q / 4KV | 971 | 10,061,760 | +0.0191% |
| S3 GQA-strong | 8Q / 2KV | 1024 | 10,059,840 | exact |
| S3 MQA | 8Q / 1KV | 1051 | 10,061,760 | +0.0191% |
| S4 research | 12Q / 4KV | 2581 | 100,376,832 | -0.00765% vs recorded 100,384,512 MHA candidate |

S4 is analytical/engineering extrapolation only. The recorded D01 S4 MHA candidate is
12Q/12KV, D=768, L=10, Dh=64, d_ff=2240, context 2048.

## Parameter allocation

At S3 the near-matched comparison reallocates capacity rather than discarding it:

| Geometry | Attention total across 6 layers | MLP total across 6 layers |
| --- | ---: | ---: |
| 8Q / 8KV | 2,457,600 | 4,976,640 |
| 8Q / 4KV | 1,843,200 | 5,592,960 |
| 8Q / 2KV | 1,536,000 | 5,898,240 |
| 8Q / 1KV | 1,382,400 | 6,053,760 |

This matters for interpretation: a GQA loss result is not being bought simply by reducing
the whole model parameter count.

## KV-cache scaling

For one sequence the full-context cache is:

`2 * layers * n_kv_heads * head_dim * context * bytes_per_element`

At bf16:

| Stage / geometry | Full-context KV cache |
| --- | ---: |
| S2 4/4 | 1.00 MiB |
| S2 4/2 | 0.50 MiB |
| S2 4/1 | 0.25 MiB |
| S3 8/8 | 7.50 MiB |
| S3 8/4 | 3.75 MiB |
| S3 8/2 | 1.875 MiB |
| S3 8/1 | 0.9375 MiB |
| S4 12/12 recorded MHA | 60 MiB |
| S4 12/4 research GQA | 20 MiB |

The savings multiply directly with batch/concurrent sequences. At S4, batch 8 is
approximately 480 MiB vs 160 MiB of bf16 KV cache before allocator/runtime overhead.
That is large enough to affect serving design; S2's absolute saving is much smaller.

GQA does not reduce the number of query-head attention scores at fixed `n_heads`.
Its direct inference benefits here are narrower K/V projections, smaller persistent cache,
less K/V bandwidth, and on the CUDA-native path no explicit replication of K/V from
`n_kv_heads` to `n_heads` before SDPA.

## Supporting LOCAL_FREE experiment

Before exact-head CI, MODEL-35 ran a real CPU/fp32 functional reproduction of the
current architecture, Trainer-like AdamW settings and KV-cache mechanics. It used three
seeds (101, 202, 303), a deterministic structured synthetic next-token fixture, and short
runs only. These numbers are supporting mechanics evidence, not locked-runtime authority.

### S2 approximately 1M

| Geometry | Mean initial loss | Mean final loss | Mean first grad norm | Median step | Cached generation | Observed cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MHA 4/4 | 7.6716 | 5.6940 | 2.983 | 18.96 ms | 758 tok/s | 307,200 B |
| GQA 4/2 | 7.6573 | 5.5946 | 2.890 | 22.29 ms | 813 tok/s | 153,600 B |
| MQA 4/1 | 7.6653 | 5.6099 | 3.037 | 17.46 ms | 757 tok/s | 76,800 B |

No S2 loss/mechanics signal is strong enough to justify changing the current MHA default.
The absolute full-context cache saving is only 0.5 MiB bf16 for 4/2 GQA.

### S3 approximately 10M

| Geometry | Mean initial loss | Mean final loss | Mean first grad norm | Median step | Cached generation | Observed cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MHA 8/8 | 9.0871 | 4.6680 | 12.533 | 84.05 ms | 141.5 tok/s | 906,240 B |
| GQA 8/4 | 9.0703 | 4.4628 | 13.028 | 70.57 ms | 161.3 tok/s | 453,120 B |
| GQA 8/2 | 9.0827 | 4.8073 | 12.924 | 74.69 ms | 146.7 tok/s | 226,560 B |
| MQA 8/1 | 9.0841 | 5.3111 | 13.214 | 78.12 ms | 151.7 tok/s | 113,280 B |

The fixture is too small and artificial to rank language-model quality. It does show that
8/4 GQA has normal initialization, finite gradients and short-run optimization while
halving cache. Stronger sharing did not produce a better mechanics signal in this probe.

In the supporting local environment, direct PyTorch `enable_gqa=True` matched the
explicit-repeat output with maximum tested logit error 0.0. Host-specific CPU native-GQA
microprobe timings are retained only as diagnostics because PyTorch's documented GQA
support boundary is CUDA and the project CPU reference path intentionally remains the
explicit-repeat implementation. Exact Torch 2.13 workflow evidence and target-GPU
measurements are required before runtime performance claims.

## Stage-triggered recommendation

1. **S0: preserve exactly.** No attention-geometry change.
2. **S1/S2 through approximately 1M: keep MHA as the default.** GQA remains a
   comparator because the absolute cache benefit is small and tiny-run evidence does not
   justify introducing more sharing.
3. **S3 approximately 10M: first GQA qualification trigger.** Prefer 8Q/4KV
   (two query heads per KV head) as the primary GQA candidate, with current 8/8 MHA kept
   as the control. Do not freeze GQA until a longer real-corpus run and target-runtime
   evidence pass.
4. **S4 approximately 100M+: GQA becomes the default engineering candidate.** Use
   12Q/4KV, d_ff=2581, 100,376,832 parameters for qualification unless S3 evidence
   rejects GQA. Keep a matched MHA control for the first S4 pilot.
5. **MQA is not the default.** Retain it only as an aggressive cache/serving comparator
   until larger real-corpus evidence shows no unacceptable loss degradation.

The trigger is therefore not "GQA is universally better". It is the point where cache and
K/V-bandwidth savings become material enough to justify the extra geometry choice while
short-run mechanics remain healthy.

## Acceptance before changing a stage default

A GQA stage candidate must pass all of the following before freeze:

- exact ModelSpec identity and parameter accounting;
- finite multi-seed training on the intended tokenizer/corpus;
- no material validation-loss regression against near-matched MHA;
- exact cached-vs-stateless generation parity within the established numerical tolerance;
- exact-lock CUDA native-GQA execution on target hardware, with safe CPU fallback retained;
- measured KV-cache bytes equal the configured `n_kv_heads` geometry;
- generation/throughput evidence on the target hardware before any performance claim;
- checkpoint manifests continue to bind the exact ModelSpec and InitSpec;
- no S0 identity change and no promotion implied by this research package.
