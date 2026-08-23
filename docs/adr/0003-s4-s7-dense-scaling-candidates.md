# ADR-0003: S4-S7 dense scaling candidates and target solver

Status: engineering candidate package; **not a canonical stage freeze**.

## Decision

Add a deterministic parameter-target solver and checked-in S4-S7 dense architecture candidates so later
compute/data planning can use exact tensor dimensions without pretending that parameter count or an illustrative
Drive ladder already selected the final architecture.

The solver varies SwiGLU hidden width around an analytic optimum for caller-supplied architecture templates.
It does not choose a tokenizer, benchmark winner, training recipe, execution backend, or promotion state.

The checked-in candidate files explicitly declare:

- `status = engineering_candidate_not_frozen`;
- `promotion_allowed = false`;
- `requires_preceding_stage_pass = true`;
- exact parameter count;
- exact ModelSpec identity hash;
- exact InitSpec identity hash;
- random-init Base lineage.

This preserves the project rule that a next-stage ModelSpec is frozen only after the preceding stage passes its
gate. These files are planning/evaluation inputs, not canonical checkpoints.

## Solver contract

For a supplied dense template, the v1 baseline is bias-free, tied-embedding, pre-RMSNorm, RoPE, SwiGLU and a
final RMSNorm. Let:

- `V` = vocabulary size;
- `D` = residual width;
- `L` = layer count;
- `Hq` = query heads;
- `Hkv` = KV heads;
- `Dh` = head dimension;
- `F` = SwiGLU hidden width;
- `Q = Hq * Dh`;
- `KV = Hkv * Dh`.

The exact parameter equation is:

`V*D + L*(2*D*(Q+KV) + 3*D*F + 2*D) + D`.

For fixed template geometry this is affine in `F`. `solve_dense_scaling_candidates` solves the real-valued
optimum, evaluates the nearest valid `d_ff_multiple` widths, computes exact `ModelSpec.parameter_count()` for
each candidate, filters by requested relative error, deduplicates by ModelSpec SHA-256 identity, and returns a
deterministically sorted shortlist.

No model tensors need to be instantiated to search billion-parameter candidates.

## Current engineering candidates

| Stage | Target | Exact params | Relative target error | Core geometry | Status |
|---|---:|---:|---:|---|---|
| S4 | 100M | 100,384,512 | +0.384512% | V=32K, D=768, L=10, Hq/Hkv=12/12, Dh=64, F=2240 | not frozen |
| S5 | 400M | 400,598,016 | +0.149504% | V=32K, D=1024, L=20, Hq/Hkv=16/4, Dh=64, F=5120 | not frozen |
| S6 | 1B | 999,106,560 | -0.089344% | V=32K, D=2048, L=18, Hq/Hkv=32/8, Dh=64, F=6720 | not frozen |
| S7 | 3B | 2,998,029,312 | -0.065690% | V=32K, D=3072, L=24, Hq/Hkv=24/8, Dh=128, F=10368 | not frozen |

ModelSpec identities:

- S4: `dc9fd9e605cbc007aa20ad29f2220c7ebade875564d68016e93d1dc2489cd693`
- S5: `9abfb6d1ac2e9c28fac20aff4ae804ad54b4102ce6f1bdeeadddf5a56027f28c`
- S6: `cc64cbe94a461c364f063652098e55bdcf640d4be756ee1e743a23dda3de7261`
- S7: `1d7145ff738e61b730e918126748050d289161f5051948e99a22aa15c20873d5`

Current InitSpec v1 identity remains
`86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`.

These hashes identify configuration semantics only. They are not checkpoint hashes, run IDs, dataset identities,
or claims of capability.

## Why MHA at S4 and GQA candidates afterward

S4 keeps the Drive illustrative 12-head MHA shape to minimize architecture novelty at the first ~100M planning
point. S5-S7 use GQA-shaped candidate templates to exercise the explicit `n_kv_heads` scaling path before large
serving economics matter. This is a candidate design choice, not evidence that GQA will win the final S5-S7
benchmark comparison. MHA alternatives remain valid solver templates.

Execution kernels remain outside ModelSpec. The same semantic candidate may be evaluated with transparent
PyTorch SDPA or later supported fused/distributed implementations without changing its model identity.

## Vocabulary and context boundary

`vocab_size=32768` is held constant in these D01 candidates because the Drive ladder uses 32K at S4 and because
future tokenizer selection belongs to D04. A future D04 tokenizer experiment may require a different vocabulary,
which necessarily creates a different ModelSpec identity and parameter count. D01 must not silently re-label the
candidate as identical.

Likewise, context lengths are planning semantics in these candidates (2K/4K/4K/8K) and must be re-evaluated with
D04/D08/D07 data, memory and serving evidence before any stage freeze.

## Explicit non-decisions

This ADR does **not**:

- freeze S4, S5, S6 or S7 as canonical;
- authorize training or paid compute;
- define token budgets or data mixtures;
- claim general language, multilingual, reasoning or assistant capability;
- introduce instruction tuning, safety alignment or domain specialization into Base;
- select muP or another parameterization/training-scaling method;
- define S8+ dense dimensions;
- define S13/S14 MoE expert count, router, top-k, active-parameter budget or expert parallel topology.

S8+ dense candidates require new benchmark/hardware evidence. S13/S14 remain a separate sparse-MoE-first research
line as already recorded by the project scaling ladder and R01 research lane.

## Required evidence before any candidate becomes canonical

At minimum: preceding stage PASS; tokenizer/vocabulary identity; data and token-budget decision; exact memory and
compute plan; training stability evidence; checkpoint/resume; generation; evaluation; independent audits; and a
new ADR explicitly selecting the winning ModelSpec. A checked-in candidate JSON is not sufficient evidence.
