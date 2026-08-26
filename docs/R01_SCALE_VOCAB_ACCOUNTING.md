# R01 scale and vocabulary accounting

Status: `PLANNING_ONLY`. This package performs no tokenizer fit, model training, corpus mutation, optimizer update, stage promotion, GPU provisioning, or paid compute authorization.

## Why this exists

The current MODEL-341 mechanics authority is exactly 20,613,440 random-init parameters at byte vocabulary 256. A future learned tokenizer changes two different quantities at once: parameter allocation and per-token output-projection compute. Therefore a nominal 20M/50M/100M comparison is not scientifically clean if it reports only total parameters.

The package keeps two comparisons separate:

1. **Fixed geometry.** Only `vocab_size` changes. This exposes the direct tied/untied embedding parameter tax and vocabulary-logit FLOPs while transformer-body parameters stay unchanged.
2. **Fixed total-parameter target.** Vocabulary changes and `d_ff` is re-solved on a declared integer granularity. This exposes the capacity displacement: a larger vocabulary consumes parameter budget that would otherwise live in the transformer body.

Neither comparison selects a tokenizer or freezes a ModelSpec.

## Exact parameter algebra

For decoder width `d`, query width `q = n_heads * head_dim`, KV width `k = n_kv_heads * head_dim`, FFN width `f`, vocabulary `V`, and `L_layers` blocks, the bias-free incumbent family uses:

- token embedding: `V * d`;
- attention weights per block: `d*q + d*k + d*k + q*d = 2*d*(q+k)`;
- SwiGLU weights per block: `3*d*f`;
- two pre-RMSNorm weights per block: `2*d`;
- final RMSNorm: `d`;
- untied LM head, when enabled: an additional `V*d`;
- LM-head bias, when enabled: an additional `V`.

The stdlib implementation also accounts for the repository's optional attention/MLP biases. For MODEL-341 (`V=256`, `d=320`, 16 blocks, 10 Q heads, 2 KV heads, head dimension 32, `f=1080`, tied embeddings), this reproduces exactly `20,613,440` parameters.

## FLOP convention

`dominant_matmul_flops_per_token` is a planning estimate, not a wall-clock claim. One multiply-add counts as two FLOPs. It reports separately:

- attention Q/K/V/output projections;
- SwiGLU dense projections;
- logical causal attention-context work for QK^T plus attention-value products;
- full-vocabulary logit projection.

For the causal-context term, a fully occupied sequence of length `S` has `S*(S+1)/2` unmasked query/key pairs. Averaged per token across the sequence, QK^T plus A@V contributes `2 * q * (S+1)` FLOPs per block under this logical triangular convention. Actual kernel/runtime cost can differ because implementations may execute different masked work, fuse operations, or incur communication/memory overhead.

The training value is deliberately labeled an estimate and uses a configurable default multiplier of 3 over dominant forward matmuls. Norms, softmax, RoPE, activation elementwise work, optimizer work, data I/O, checkpointing, padding inefficiency, communication, and kernel overhead are excluded.

A `6*N` value is emitted only as a comparison proxy. It is not presented as exact because small-model attention/context cost and vocabulary projection cost can be material, while parameter count also mixes embedding allocation with transformer capacity.

## Candidate-only surfaces

The configuration includes three accounting probes, all `PLANNING_CANDIDATE_ONLY`:

- `R01-50M-GQA-A`: 49,726,976 parameters at byte vocab 256;
- `R01-100M-GQA-A`: 99,753,216 parameters at byte vocab 256;
- `R01-100M-MHA-EXISTING-CONTROL`: 99,897,600 parameters, reproducing the existing S4 MHA engineering-candidate geometry for comparison.

The GQA probes are research surfaces, not a replacement for learned-20M evidence. A future 100M freeze must be based on measured learned-20M loss-vs-exposure, BPB, throughput, memory, checkpoint/recovery behavior, tokenizer identity, and available unique causal-loss supply.

## Research interpretation

DeepSeek LLM documents why simple parameter-count compute proxies can be materially wrong when attention/context and vocabulary costs are omitted or conflated. *Scaling Laws with Vocabulary* (arXiv:2407.13623) treats vocabulary size as an explicit scaling variable. MobileLLM (arXiv:2402.14905) supports deep/thin, embedding-sharing and GQA as useful sub-billion candidate biases rather than automatic architecture law. Tensor Programs V / muTransfer (arXiv:2203.03466) is a later candidate for measured cross-scale hyperparameter transfer; ordinary parameterization does not justify copying 20M hyperparameters into 100M.

## Truth boundary

This accounting package can make a comparison better specified. It cannot make Research Corpus V1 terminal, authorize tokenizer fitting, prove a learned model, select final architecture, or authorize material compute. The learned-20M critical path remains data authority -> decontamination/split/packing -> exact unique-loss ledger -> tokenizer decision -> checkpoint integrity -> bounded pilot -> explicit compute/training authorization -> learned evidence.
