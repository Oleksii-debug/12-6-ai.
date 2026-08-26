# Scaling Data-Budget Policy

## Decision

The project must keep **model parameter count**, **source bytes**, **post-tokenization training tokens**, and **unique causal-loss positions** as separate quantities.

The current primary architecture has 20,613,440 parameters. Source-registry capacity is owned by the data lane and changes as exact-head source admissions succeed, fail, or are corrected. This scaling policy therefore **does not embed a live source-byte snapshot**. A rapidly changing source registry is not a corpus identity, token count, or causal-loss ledger.

That separation is deliberate: during active source qualification, an optimistic source vector can be superseded within minutes by a fail-closed audit. Copying the current byte count into a long-lived scaling contract creates a race and can turn stale planning data into false scientific authority.

## Research reference

Hoffmann et al. (2022) found that compute-optimal model size and training-token count should scale together. The commonly used Chinchilla planning point is approximately 20 training tokens per parameter. This repository uses **20x only as a planning reference**, not as a universal optimum or a model-quality guarantee.

Later work strengthens the reason not to treat 20x as a ceiling. Data-constrained scaling work shows that repeated data has diminishing value; inference-aware scaling and over-training studies show that smaller models can rationally be trained on substantially more tokens; Llama 3 reports continued gains far beyond its Chinchilla-optimal token count. DataComp-LM also shows that curation and filtering quality materially change outcomes.

Therefore the 12-6 AI policy is:

1. do not convert source bytes to tokens by assumption;
2. do not use a source-capacity acquisition target as a training-token budget;
3. do not count replayed tokens as unique loss positions;
4. keep live source-registry facts in the data authority rather than duplicating them in scaling policy;
5. require exact corpus, split, tokenizer, post-tokenization token-count and loss-mask identities before long training;
6. treat a run below the 20x planning reference as a bounded scaling/smoke experiment unless a preregistered scaling experiment justifies a different budget;
7. choose larger data budgets from scaling experiments, evaluation results and compute constraints rather than from an arbitrary fixed multiplier;
8. keep explicit compute authorization separate from scientific readiness.

## Current planning numbers

For the exact 20,613,440-parameter MODEL-341 geometry:

- 20x reference: 412,268,800 unique loss tokens;
- 50x exploration point: 1,030,672,000 unique loss tokens;
- 100x exploration point: 2,061,344,000 unique loss tokens;
- rough dense-transformer training compute at 20x using `6 * N * D`: 50,989,669,036,032,000 FLOPs.

For future planning only:

| Stage | Parameters | 20x tokens | 50x tokens | 100x tokens |
| --- | ---: | ---: | ---: | ---: |
| 20M primary | 20,613,440 | 412,268,800 | 1,030,672,000 | 2,061,344,000 |
| 100M target | 100,000,000 | 2,000,000,000 | 5,000,000,000 | 10,000,000,000 |
| 1B target | 1,000,000,000 | 20,000,000,000 | 50,000,000,000 | 100,000,000,000 |

These numbers are planning references. They do not imply that 20x is enough for the desired downstream capability, that 100x is affordable, or that data repetition is equivalent to unique data.

## Data-authority contract

The live source registry may report bytes and independent source families for acquisition planning. Those values remain external to this policy because they are volatile and because source admission is earlier than corpus materialization.

Promotion from source acquisition to learned-stage readiness requires terminal exact-head authority, an immutable materialized corpus, global deduplication, evaluation decontamination, quality/privacy review, deterministic split/shard/pack construction, tokenizer identity, exact post-tokenization token accounting and exact unique causal-loss accounting.

No source-registry byte total, no source-capacity target, no queued CI run and no stale snapshot can substitute for those artifacts.

## Long-training gate

A learned-stage long run remains fail-closed until the project has immutable evidence for:

- corpus identity and train-split identity;
- tokenizer identity;
- exact post-tokenization train-token count;
- exact unique causal-loss position count and document-boundary loss mask;
- evaluation decontamination;
- deduplication;
- quality/privacy review;
- checkpoint/resume integrity;
- a preregistered token/step budget and stopping rule;
- explicit compute authorization.

The machine-readable policy and validator are in `configs/scaling/data_budget_policy_v1.json` and `tools/validate_scaling_data_budget_policy.py`.

## References

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.
- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264.
- Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448.
- Gadre et al., *Language models scale reliably with over-training and on downstream tasks*, arXiv:2403.08540.
- Meta, *Introducing Meta Llama 3*.
- Li et al., *DataComp-LM*, arXiv:2406.11794.
