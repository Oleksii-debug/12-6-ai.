# MODEL-142 — ~10M architecture transfer matrix

## Question

Does an architecture choice that looked useful below / around 1M parameters still help at the current approximately 10M S3 geometry when parameter count, corpus, tokenizer, optimizer, initialization and optimized-token budget are held fixed?

This is deliberately not another architecture search. The matrix contains three candidates and changes only KV-head grouping, with FFN width adjusted algebraically to keep the total parameter budget matched.

Exact Product source base: `fb9c6d9b73ce436d637077892d73edf136fcaeac`.

Current incumbent binding:

- `configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json`
- 10,000,640 trainable parameters
- ModelSpec `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`
- InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`

## Evidence intake: what was and was not transferred

### Depth / width

MODEL09 PR #242 predeclared a strong exact-token 500K depth/width experiment, but its exact-head workflow failed at focused tests before the candidate matrix executed. It therefore produced no scientific winner that MODEL-142 is allowed to transfer.

Decision: **do not reopen depth/width at 10M**.

### FFN ratio

MODEL-12/13 PR #232 executed the approximately 1M exact-parameter FFN allocation matrix. It explicitly set `quality_winner = null`: the S0 fixture is compatibility-only and was not authorized to select 1M quality from its loss. Ratios 2.50–3.00 were carry-forward candidates, not a winner.

Decision: **do not create a standalone 10M FFN-ratio sweep**. `d_ff` changes in MODEL-142 only compensate the attention parameter surface so total N remains matched.

### Head count / head dimension

PR #232 also executed the 100K MHA head-count sweep. `3 heads x head_dim 16` was a provisional CPU latency result. Mean final held-out CE was 3.0062 versus 2.9796 for the 4-head control, so it is not a quality winner.

Decision: **keep 8 query heads / head_dim 32 fixed**.

### GQA / MHA

MODEL-35 PR #172 is the strongest usable architecture signal. Its three-seed LOCAL_FREE mechanics study declared approximately 10M as the first GQA qualification stage. At its S3 comparator, mean final loss was:

- 8Q/8KV MHA: 4.6680
- 8Q/4KV GQA: 4.4628
- 8Q/2KV GQA: 4.8073
- 8Q/1KV MQA: 5.3111

Those were short synthetic mechanics runs, not language-model quality evidence, but they predeclared **8Q/4KV** as the primary S3 transfer candidate while retaining MHA as control.

Decision: **this is MODEL-142's sole transfer axis**.

### Tokenizer allocation

MODEL37 PR #167 measured a repeatable BPE-472 mechanics advantage over bytes, but explicitly retained `NO_FREEZE_REPEATABILITY_BLOCKED` for tokenizer selection and proposed stage-aware vocabulary searches rather than a frozen tokenizer. Varying vocabulary here would also destroy the exact current 10M control.

Decision: **hold S0 byte tokenizer / V256 fixed for every candidate**.

### Initialization

MODEL-34 PR #169 retained Normal(0, 0.02) plus residual output scale `0.02/sqrt(2L)`. The later MODEL-19 decisive seed experiment did not complete successfully enough to replace it.

Decision: **hold InitSpec v1 exactly fixed**.

## Predeclared 10M candidates

| candidate | role | Hq/Hkv | head dim | d_ff | parameters | delta vs incumbent |
|---|---|---:|---:|---:|---:|---:|
| `incumbent_gqa_8q2kv` | current 10M control | 8/2 | 32 | 864 | 10,000,640 | 0 |
| `transfer_gqa_8q4kv` | strongest small-scale transfer candidate | 8/4 | 32 | 821 | 9,997,568 | -3,072 (-0.0307%) |
| `mha_8q8kv` | partial-transfer isolation control | 8/8 | 32 | 736 | 10,000,640 | 0 |

All retain V256 / context 1024 / D256 / L12 / RoPE / pre-RMSNorm / SwiGLU / tied embeddings / no bias / no dropout.

Parameter allocation is intentionally rebalanced rather than allowing GQA to win by simply being smaller:

| candidate | attention total | MLP total | embedding | total |
|---|---:|---:|---:|---:|
| incumbent 8Q/2KV | 1,966,080 | 7,962,624 | 65,536 | 10,000,640 |
| transfer 8Q/4KV | 2,359,296 | 7,566,336 | 65,536 | 9,997,568 |
| MHA 8Q/8KV | 3,145,728 | 6,782,976 | 65,536 | 10,000,640 |

## Shared training controls

Every run uses the same committed project-authored S0 corpus and held-out split:

- dataset: `s0-tiny-controlled-v1`
- dataset identity: `bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89`
- byte tokenizer: `s0-byte-v1`
- batch: 4
- sequence length: 64
- 32 optimizer updates
- exactly 8,064 valid causal optimization targets
- seeds: 1515, 1516, 1517
- AdamW: lr 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0
- constant schedule, zero warmup
- global clip norm 1.0
- fp32 CPU
- deterministic algorithms

The corpus is tiny and repeatedly cycled. This experiment therefore answers a bounded transfer/optimization question only; it cannot establish representative-corpus 10M quality.

## LOCAL_FREE reconstruction run completed before exact-head Actions

The connected execution environment could not DNS-clone GitHub, so the first execution reconstructed the exact Product ModelSpec/model semantics and committed S0 byte records from GitHub reads. Runtime matched the repository's measured CPU environment: Python 3.13.5 / PyTorch 2.10.0+cpu / CUDA unavailable. The committed exact-head workflow is the higher authority once terminal.

Three paired seeds x three candidates were trained for the full 8,064-token budget.

| candidate | mean held-out BPB | SD | mean final train BPB | mean optimization wall | effective tokens/s |
|---|---:|---:|---:|---:|---:|
| incumbent 8Q/2KV | 4.631778 | 0.104868 | 5.508289 | 6.684 s | 1,206 |
| transfer 8Q/4KV | **4.491343** | **0.023361** | **5.056719** | 6.200 s | 1,301 |
| MHA 8Q/8KV | 4.612210 | 0.018677 | 5.274484 | **5.866 s** | **1,375** |

Paired final held-out BPB deltas (`8Q/4KV - comparator`; lower is better):

- versus incumbent: -0.022345, -0.217008, -0.181952; 3/3 wins; mean delta -0.140435 BPB (~3.03% of incumbent mean BPB)
- versus MHA: -0.093769, -0.106837, -0.161993; 3/3 wins; mean delta -0.120866 BPB (~2.62% of MHA mean BPB)
- incumbent versus MHA is mixed across seeds, so the reconstruction does not support a clean 8Q/2KV-vs-MHA quality ordering.

### Optimization / layer health

The detailed seed-1515 instrumentation recorded finite gradients in every trainable element and final nonzero gradient utilization of 100% for all three candidates. Changed-element utilization was effectively complete (>99.9997% per candidate; a handful of float elements ended exactly at their initial value), so no dead block or unused projection was detected.

At the final step:

- incumbent per-layer gradient norm range: 0.0295–0.0661; attention activation RMS 0.0631–0.2410; MLP RMS 0.0673–0.1297
- 8Q/4KV per-layer gradient norm range: 0.0354–0.0685; attention activation RMS 0.0296–0.1497; MLP RMS 0.0560–0.1395
- MHA per-layer gradient norm range: 0.0423–0.1707; attention activation RMS 0.0519–0.2314; MLP RMS 0.0459–0.1064

No layer produced non-finite activation/gradient statistics or a missing update path. The 8Q/4KV candidate did not show a late-layer gradient blow-up relative to the controls in this bounded run.

All three candidates clipped on all 32 updates. That is an important limitation: the run is useful for matched transfer direction and health, but the absolute optimizer regime is gradient-limited and is not sufficient evidence for architecture promotion.

### Memory and KV cache

Seed-1515 measured FP32 model / Adam state tensor surfaces were approximately:

- incumbent: 40,002,560 model bytes + 80,005,560 optimizer-state bytes
- 8Q/4KV: 39,990,272 + 79,980,984 bytes
- MHA: 40,002,560 + 80,005,560 bytes

The health harness also retains a full FP32 initial-parameter snapshot, so its process RSS high-water (roughly 765–782 MiB) is intentionally higher than ordinary training and must not be mistaken for deployment memory.

Theoretical full-context batch-1 KV storage at 1024 tokens:

- incumbent 8Q/2KV: 3 MiB BF16 / 6 MiB FP32
- transfer 8Q/4KV: 6 MiB BF16 / 12 MiB FP32
- MHA 8Q/8KV: 12 MiB BF16 / 24 MiB FP32

Thus 8Q/4KV gives up 3 MiB BF16 cache relative to the aggressive incumbent but still halves MHA KV storage.

## Decision

The reconstructed three-seed bounded evidence supports the **transfer signal** for 8Q/4KV: it wins held-out BPB against both controls in every paired seed and remains numerically healthy. It also reproduces the direction of MODEL-35 rather than the 8Q/2KV incumbent's short-run ordering.

However, MODEL-142 does **not** freeze or replace the current 10M incumbent from this fixture. The correct status is:

`8Q/4KV = PREFERRED_NEXT_10M_TRANSFER_CANDIDATE`

`current 8Q/2KV = CANONICAL_INCUMBENT_UNCHANGED`

Promotion requires a representative corpus/tokenizer run with the same matched matrix (or at minimum incumbent vs 8Q/4KV), a less saturated clipping regime justified independently by optimizer evidence, and exact-head retained results. If the exact-head workflow reverses the paired BPB ordering or exposes a real layer-health divergence, the fallback is `KEEP_10M_INCUMBENT`.

## Exact-head execution

`.github/workflows/model142-10m-transfer-matrix.yml` runs all 9 trajectories in fresh processes under the repository's locked linux-x86_64 environment and retains per-run JSON plus an aggregate decision report as a 30-day artifact.

No paid compute, foreign weights, architecture freeze, stage promotion, tokenizer freeze, or broad capability claim is introduced.
