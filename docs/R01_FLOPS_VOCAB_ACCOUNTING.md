# R01 FLOPs-aware scale and vocabulary accounting

## Purpose

The merged R01 campaign correctly blocks learned 100M work until learned 20M evidence exists, but total parameter count alone is not a sufficient experimental axis once the byte vocabulary is replaced by a learned tokenizer. Vocabulary size changes tied embedding parameters and dense LM-head projection work. Holding total parameters constant while changing vocabulary also changes how many parameters remain for the transformer body.

This package makes those quantities explicit without fitting a tokenizer, training a model, freezing a 100M ModelSpec, or authorizing compute.

## Exact MODEL-341 parameter proof

Bound baseline:

- vocabulary: 256
- width: 320
- layers: 16
- query heads: 10
- KV heads: 2
- head dimension: 32
- SwiGLU hidden dimension: 1080
- two block RMSNorm scales and one final RMSNorm scale
- tied token embedding / LM head
- bias-free Q/K/V/O and SwiGLU linear maps

Per layer:

- Q: `320 × 320 = 102,400`
- K: `320 × 64 = 20,480`
- V: `320 × 64 = 20,480`
- O: `320 × 320 = 102,400`
- attention projections: `245,760`
- SwiGLU gate/up/down: `3 × 320 × 1080 = 1,036,800`
- two RMSNorm scales: `640`
- block total: `1,283,200`

Across 16 blocks: `20,531,200`.

Add tied embedding `256 × 320 = 81,920` and final RMSNorm `320`:

`20,531,200 + 81,920 + 320 = 20,613,440` parameters.

The utility rejects head geometry that violates `d_model = n_heads × head_dim` or GQA grouping that violates `n_heads % n_kv_heads == 0`.

## Why vocabulary must be separated

With tied embeddings, increasing vocabulary by ΔV adds `ΔV × d_model` parameters, not zero. Tying avoids a second independent output matrix, but inference/training still computes logits over the vocabulary, so output-projection work grows with `d_model × vocab_size`.

Two comparisons therefore answer different questions:

1. **Fixed geometry:** keep depth/width/FFN/head geometry constant and measure the parameter/FLOP increase caused by vocabulary alone.
2. **Fixed total parameters:** change vocabulary and compensate via `d_ff` so the total parameter target stays approximately fixed. This is useful accounting, but it is not a scientifically matched architecture because the transformer-body capacity changed.

The package reports both and never treats them as interchangeable.

## FLOP convention

The utility reports dominant matrix-multiplication planning terms per token:

- Q/K/V/O projection work;
- SwiGLU gate/up/down work;
- attention context work (`QKᵀ` + `AV`) as a sequence-length-dependent term;
- vocabulary projection work.

One multiply-add is counted as 2 FLOPs. A planning training estimate uses `3 × forward dominant-matmul FLOPs` to represent forward plus the two main backward matmul families. The common `6 × parameter_count` number is shown only as a comparison proxy, never as exact compute.

These are analytical planning estimates. Fused kernels, causal masking, recomputation, optimizer work, memory traffic, padding, device utilization, and wall-clock throughput require runtime profiling on the eventual exact stack.

## Candidate-only arithmetic anchors

Two clean deep-thin/GQA arithmetic anchors are included for the later R01 E30 experiment design:

- ~50M: vocab 512, width 384, 25 layers, 6 Q heads / 2 KV heads, head_dim 64, d_ff 1408 → **50,596,992 parameters**.
- ~100M: vocab 512, width 512, 26 layers, 8 Q heads / 2 KV heads, head_dim 64, d_ff 2048 → **99,117,568 parameters**.

They are intentionally labelled `PLANNING_ONLY_NOT_MODELSPEC`. The learned 20M curve, real tokenizer, corpus supply, throughput, memory, checkpoint reliability and compute budget must determine whether either geometry is worth testing. They are not launch configurations.

## Research basis

The package follows the already-merged R01 direction:

- Hoffmann et al., *Training Compute-Optimal Large Language Models* — joint model/data scaling rather than parameter count in isolation: https://arxiv.org/abs/2203.15556
- Liu et al., *MobileLLM* — evidence for deep-thin, embedding sharing and GQA in sub-billion regimes: https://arxiv.org/abs/2402.14905
- Tao et al., *Scaling Laws with Vocabulary* — vocabulary size as a scaling variable rather than a free constant: https://arxiv.org/abs/2407.13623
- DeepSeek-AI, *DeepSeek LLM* — small-model compute accounting should not blindly collapse all costs into a single parameter-count proxy: https://arxiv.org/abs/2401.02954

Research citations motivate experiments; they do not authorize training or replace measurement on this repository.

## Operator use

From repository root:

`python tools/r01_flops_vocab_accounting.py`

Focused regression suite:

`python -m pytest -q tests/test_r01_flops_vocab_accounting.py`

No dedicated Actions workflow is added. Generic CI is the exact-head execution authority.

## Truth boundary

`LOCAL_FREE_PLANNING_ONLY`.

Tokenizer fitting is not authorized. Model training is not executed. Optimizer updates are zero. Paid compute is not authorized. The 100M ModelSpec remains unfrozen. Candidate geometries are arithmetic planning surfaces only.
