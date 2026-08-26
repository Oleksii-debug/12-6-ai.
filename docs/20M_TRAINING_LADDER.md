# 20M training ladder

## Decision

The exact MODEL-341 configuration is a mechanically qualified random-initialized decoder with 20,613,440 parameters. The currently preregistered 20,000,000 unique optimized-target campaign is retained, but its semantic role is narrowed to an end-to-end learning/recovery pilot. It must not be described as sufficient evidence that a general 20M Base is fully trained.

Long training remains blocked until a terminal exact corpus identity, shard identity and non-zero authorized unique-loss capacity exist. This document does not authorize paid compute.

## Why the distinction matters

The current Research Corpus V1 effort is necessary to prove provenance, rights, quality, privacy, deduplication, evaluation decontamination, deterministic splits/shards and loss accounting. That is an engineering and scientific-control milestone. Corpus correctness and corpus scale are separate questions.

Hoffmann et al., *Training Compute-Optimal Large Language Models* (arXiv:2203.15556), trained Chinchilla at 70B parameters on about 1.4T tokens. The approximately 20 tokens-per-parameter ratio is useful here as a planning reference. It is not asserted to be the exact optimum for a 20.6M model because the paper's fitted regime is much larger than this stage.

For MODEL-341, the reference calculation is:

`20,613,440 parameters * 20 = 412,268,800 unique optimized targets`.

The ladder therefore separates:

1. 20M cumulative targets — pipeline, optimizer, learning-signal, checkpoint and recovery pilot.
2. 100M cumulative targets — intermediate scientific checkpoint.
3. 200M cumulative targets — intermediate scientific checkpoint.
4. 412,268,800 cumulative targets — Chinchilla-style reference-scale evaluation candidate.

Each milestone is a stop-and-evaluate boundary, not an instruction to spend compute automatically. A worse validation trend, data-quality defect, recovery defect, contamination finding, numerical instability or poor scaling efficiency stops progression.

## Small-model evidence boundary

TinyStories (Eldan and Li, arXiv:2305.07759) shows that models below 10M parameters can produce coherent multi-paragraph text when trained on a deliberately constrained synthetic story distribution. That is useful evidence that very small transformers can learn meaningful structure; it is not evidence that a similarly small unrestricted Base has broad language competence.

MobileLLM (Liu et al., arXiv:2402.14905) is relevant to the later 100M-1B architecture path because it reports strong sub-billion results from deep-thin architectures, embedding sharing and grouped-query attention. 12-6 should test those choices with controlled ablations rather than assuming that parameter count alone determines quality.

SmolLM2 (Ben Allal et al., arXiv:2502.02737) documents a modern data-centric small-model regime in which a 1.7B model is trained on roughly 11T tokens. This is evidence that useful modern small models are often deliberately trained beyond a simple compute-optimal reference. It does not justify copying that token budget into 12-6 without measured benefit and an explicit compute budget.

## 100M and 1B planning references

Using the same planning ratio only as a first-order reference gives:

- nominal 100M parameters -> 2B tokens;
- nominal 1B parameters -> 20B tokens.

These figures are capacity-planning anchors, not fixed training prescriptions. Before either scale is authorized, 12-6 must use learning curves from the smaller stages to fit its own data/compute scaling model, estimate wall-clock and monetary cost, and decide whether additional tokens or additional parameters provide the better marginal return.

## Promotion gates

No materially paid long run may start unless all applicable gates are terminal and bound to the exact candidate lineage: corpus/shard identity, purpose-specific rights and provenance, quality/privacy/dedup, evaluation decontamination, tokenizer identity, D05 corruption rejection before live mutation, save/load/resume and RNG continuation, bounded training numerics, held-out evaluation separation, exact run config, hardware profile, cost estimate, artifact destination and explicit compute authorization.

The D05 corruption finding is a P0 gate. A checkpoint loader that silently casts incompatible tensor dtypes, accepts malformed optimizer geometry or accepts invalid counters cannot be trusted for a long campaign even when ordinary save/load tests pass.

## Operational consequence

Research Corpus V1 should be completed because it unlocks the first real learning pilot. It should not be treated as the final data scale for the 20.6M Base. After the 20M-target pilot is terminal, the next decision is based on measured held-out loss, learning-curve slope, tokenizer fertility, data-mixture behavior, checkpoint integrity and compute efficiency. Only then should the project proceed to 100M, 200M and the reference-scale target.

The same rule applies to 100M and 1B model growth: scale only from measured evidence, not from a parameter-count milestone alone.
