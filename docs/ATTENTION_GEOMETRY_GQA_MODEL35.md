# MODEL-35 grouped-query attention geometry research

## Scope and authority

This package asks one narrow question: at what model stage does reducing K/V head count
become a better engineering default for 12-6 AI than retaining full MHA?

It preserves canonical S0 exactly. It does not freeze S1/S2/S3/S4 architecture, authorize
paid compute, claim model capability, or treat a short synthetic mechanics fixture as
language-model quality evidence.

The experiment is stacked on the D07 model-native KV-cache implementation rather than
creating a second decoder or cache path.

## Current implementation audit

`ModelSpec` already represents MHA, GQA and MQA through `n_heads` and `n_kv_heads`.
K/V projection widths and parameter accounting scale with `n_kv_heads`. The D07 cache
also stores unexpanded K/V as:

`[batch, n_kv_heads, sequence, head_dim]`

That part has the correct GQA memory geometry.

The remaining scaling defect is inside `CausalSelfAttention._attend()`: for GQA/MQA it
calls `_expand_kv()` and `repeat_interleave()` to materialize K/V back to query-head
width before `scaled_dot_product_attention()`. Therefore current Product semantics gain:

- fewer K/V projection parameters;
- fewer stored KV-cache bytes;

but do not preserve grouped K/V geometry inside the attention kernel. This is a
performance/scaling defect, not a semantic correctness defect.

The exact locked x86 runtime is Torch 2.13.0. Its SDPA API exposes `enable_gqa=True`.
MODEL-35 therefore measures native-GQA parity and host-specific timing against the
existing repeat-expansion path. A Product kernel change must remain conditional on exact
locked-runtime evidence; CPU results are not a CUDA fused-kernel claim.

Primary background:
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints", arXiv:2305.13245.
- PyTorch 2.13 `torch.nn.functional.scaled_dot_product_attention`, `enable_gqa=True`.

These sources motivate the experiment only. They do not decide 12-6 geometry.

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

In the supporting environment, PyTorch native `enable_gqa=True` matched the explicit
repeat path with maximum tested logit error 0.0. Attention microprobes were faster for
native GQA, but those host-specific CPU timings are not recorded as Product performance
claims. Exact Torch 2.13 workflow evidence is required.

## Stage-triggered recommendation

1. **S0: preserve exactly.** No attention-geometry change.
2. **S1/S2 through approximately 1M: keep MHA as the default.** GQA remains a
   comparator because the absolute cache benefit is small and tiny-run evidence does not
   justify introducing more sharing.
3. **S3 approximately 10M: first GQA qualification trigger.** Prefer 8Q/4KV
   (two query heads per KV head) as the primary GQA candidate, with current 8/8 MHA kept
   as the control. Do not freeze GQA until a longer real-corpus run and exact runtime
   kernel evidence pass.
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
- exact-lock native/fused GQA kernel support, or an explicitly measured fallback;
- measured KV-cache bytes equal the configured `n_kv_heads` geometry;
- generation/throughput evidence on the target hardware before any performance claim;
- checkpoint manifests continue to bind the exact ModelSpec and InitSpec;
- no S0 identity change and no promotion implied by this research package.
