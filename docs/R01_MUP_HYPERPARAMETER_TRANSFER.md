# R01 μP hyperparameter-transfer preregistration

## Purpose

This package extends the merged R01 20M → 100M scaling campaign with a bounded research question: whether a deliberately implemented maximal-update parameterization can reduce hyperparameter retuning cost across the 12-6 AI width ladder.

It does not adopt μP, change MODEL-341, authorize training, authorize paid compute, change the corpus, or freeze a 50M/100M ModelSpec. Standard parameterization remains the control until measured evidence justifies another choice.

## Why this matters

The current MODEL-341 candidate has exact mechanics at 20,613,440 randomly initialized parameters. Scaling that architecture by parameter count alone does not make the optimizer recipe transferable. Ordinary parameterization and maximal-update parameterization make different scale-dependent choices for initialization, learning rates and update magnitudes.

Tensor Programs V reports zero-shot hyperparameter transfer across model sizes under maximal-update parameterization. This project treats that result as a hypothesis to reproduce under its own tied-embedding, RMSNorm, RoPE, SwiGLU and grouped-query-attention architecture rather than as permission to copy a recipe blindly.

## Coordinate-check gate

Before any learned transfer claim, the μP candidate must pass a preregistered width coordinate check at fixed depth and head dimension. The contract uses d_model/n_heads probes 192/6, 256/8 and 320/10 while retaining 16 layers, head_dim 32, two KV heads, tied embeddings, RoPE, RMSNorm and SwiGLU.

The probes must use the same exact input identity and comparable seed/optimizer/loss conditions. Evidence must include per-layer activation RMS, update-to-weight ratios, gradient norms and NaN/Inf events. Width transfer and depth transfer are separate claims; success on one cannot be used as evidence for the other.

The implementation choice remains deliberately unfrozen. Any implementation must explicitly verify tied-embedding handling, attention-logit scaling, initialization scaling, optimizer parameter-group scaling and scheduler behavior. Checkpoint identity must record the parameterization so a standard-parameterization checkpoint cannot be resumed as μP or vice versa without a compatibility failure.

## Transfer experiment

The future comparison has two arms:

- SP-CONTROL: the current standard parameterization, tuned independently at the measured scale.
- MUP-CANDIDATE: maximal-update parameterization after coordinate checks pass.

The first learning-rate experiment is a small relative multiplier grid of 0.5×, 1× and 2× around the measured incumbent. Future targets are the exact 20,613,440-parameter candidate, nominal 50M and nominal 100M. A zero-shot transfer claim is evidence to be earned, not a default assumption.

Batch size, sequence length, training duration, weight decay, dropout, depth and token budget are not declared automatically transferable.

## Data-constrained pilot

Research Corpus V1 is still the gating authority. If the exact eligible corpus is smaller than a desired exposure budget, a future bounded pilot may compare 1, 2 and 4 epochs over the same decontaminated split identity. This is an experiment about repeated exposure, not a way to inflate unique data capacity.

The ledger must report unique non-ignored causal-loss positions and total loss-position exposures separately. Repeating one position four times still contributes one unique position. The contract authorizes no change to the project's no-repetition policy by itself.

This distinction follows data-constrained scaling work showing that controlled repetition can retain value for a limited number of epochs before diminishing returns, while preserving the difference between unique data and repeated exposure.

## Metrics and decision boundary

Cross-tokenizer quality remains bits per byte. Required evidence also includes held-out loss, loss curves, per-layer activation RMS, update-to-weight ratios, gradient norms, throughput, peak memory, unique positions, total exposures and checkpoint/resume equivalence.

μP cannot replace standard parameterization without measurement. It cannot justify a 100M ModelSpec without learned 20M evidence. Repeated exposure cannot satisfy a unique-loss budget. One successful seed cannot authorize stage promotion, and final-test records cannot be used for hyperparameter selection.

## Research references

- Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer — https://arxiv.org/abs/2203.03466
- Scaling Data-Constrained Language Models — https://arxiv.org/abs/2305.16264
- MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases — https://proceedings.mlr.press/v235/liu24ce.html
- Training Compute-Optimal Large Language Models — https://arxiv.org/abs/2203.15556

## Truth boundary

LOCAL_FREE engineering and preregistration only. No model training, optimizer update, corpus mutation, tokenizer fit, final-test access, stage promotion or paid compute is authorized or claimed by this package.
