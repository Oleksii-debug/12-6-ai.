# External Scale Calibration: 20M → 100M → 1B

Date: 2026-08-26

Execution boundary: `LOCAL_FREE`. This document does not authorize or execute long training or paid compute.

## Decision

`DATA_FIRST_DO_NOT_LAUNCH_LONG_20M_TRAINING`

The current primary engineering candidate is MODEL-341 at exactly 20,613,440 random-initialized parameters. Its mechanics qualification is useful, but parameter count is not learned capability. Current external-real training authority still exposes zero authorized unique optimized targets, so a long learned 20M campaign must remain blocked.

## Why this matters

The project already separates mechanics from meaningful learned work. This calibration makes the scale of the data requirement explicit and prevents a small pipeline-validation corpus from being mistaken for a pretraining corpus.

Project-local guards currently use two distinct bands:

- meaningful science: 0.5–2 unique nonignored causal targets per parameter;
- full-pretraining planning: 5–20 training tokens per parameter.

These are planning envelopes, not universal optima.

For MODEL-341 (20,613,440 parameters), those bands imply approximately 10.3M–41.2M unique targets for a meaningful no-replay science campaign and 103.1M–412.3M training tokens for full-pretraining planning.

For 100M parameters, the corresponding planning ranges are 50M–200M unique targets and 500M–2B training tokens.

For 1B parameters, they are 500M–2B unique targets and 5B–20B training tokens.

## External calibration

Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022), provides the classic compute-optimal scaling reference. Chinchilla used about 1.4T tokens for 70B parameters, roughly 20 tokens per parameter. Source: https://arxiv.org/abs/2203.15556

Muennighoff et al., *Scaling Data-Constrained Language Models* (JMLR 2025; NeurIPS work originally published in 2023), shows that in data-constrained fixed-compute regimes several epochs of repeated data can remain useful, with up to roughly four epochs sometimes close in loss to using unique data, before diminishing returns become increasingly important. This supports controlled repetition experiments, but repeated tokens must never be relabelled as unique data. Source: https://jmlr.org/papers/v26/24-1000.html

Meta reports that the Chinchilla-optimal reference for an 8B model is around 200B tokens, while Llama 3 continued improving when trained far beyond that amount, up to 15T tokens. This is evidence that inference-efficient smaller models can rationally be trained beyond simple compute-optimal ratios. Source: https://ai.meta.com/blog/meta-llama-3/

Hugging Face reports that SmolLM 135M and 360M were each trained on 600B tokens. This is not a target for 12-6, but it is a useful reality check: practical small-model training can be extremely data-intensive when quality and inference efficiency matter. Source: https://huggingface.co/blog/smollm

## Operational policy

1. Keep the no-replay unique-loss ledger as the authority baseline.
2. Do not launch a long 20M campaign until the research corpus, tokenizer, checkpoint corruption matrix, optimizer mechanics, selection split and final split are exact-bound and terminal.
3. If data remains scarce after a terminal unique corpus exists, test controlled repetition as a separate preregistered experiment. Do not inflate unique-capacity accounting with repeated epochs.
4. Do not move to a meaningfully trained 100M Base merely because 100M mechanics fit memory. Require evidence that the available data and held-out curves justify the scale.
5. Treat 1B as a later systems target. A serious 1B Base implies data requirements in the billions of tokens even under the project's conservative planning envelope.

The machine-readable contract is `configs/research/external_scale_calibration_v1.json`; `tools/validate_external_scale_calibration.py --self-test` validates the fail-closed invariants without importing project model dependencies.
