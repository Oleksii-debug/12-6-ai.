# Scaling Data-Budget Policy

## Decision

The project must keep **model parameter count**, **source bytes**, **unique post-tokenization corpus tokens**, **unique causal-loss positions**, and **total training-token exposures** as separate quantities.

The current primary architecture has 20,613,440 parameters. Source-registry capacity is owned by the data lane and changes as exact-head source admissions succeed, fail, or are corrected. This scaling policy therefore **does not embed a live source-byte snapshot**. A rapidly changing source registry is not a corpus identity, token count, causal-loss ledger, or training budget.

That separation is deliberate: during active source qualification, an optimistic source vector can be superseded within minutes by a fail-closed audit. Copying the current byte count into a long-lived scaling contract creates a race and can turn stale planning data into false scientific authority.

## Research reference

Hoffmann et al. (2022) studied model size and the **total number of tokens used for training** under a compute budget. The commonly used Chinchilla planning point is approximately 20 training-token exposures per parameter. It is not a requirement for 20 unique corpus tokens per parameter.

That distinction matters in a data-constrained project. Muennighoff et al. (2023) explicitly vary repeated-data training and show that limited replay can retain substantial value while further repetition eventually gives diminishing returns. Unique corpus size, replay/epoch count and total token exposure therefore need separate ledgers.

Later inference-aware and over-training work also shows why 20x is not a universal ceiling. Smaller models may rationally receive substantially more total token exposure depending on inference economics, downstream quality and compute constraints. Data curation remains a separate determinant of quality.

Therefore the 12-6 AI policy is:

1. do not convert source bytes to tokens by assumption;
2. do not use a source-capacity acquisition target as a training-token budget;
3. do not relabel total token exposures as unique corpus tokens or unique causal-loss positions;
4. record unique post-tokenization data, replay/epoch policy and total training-token exposure separately;
5. keep live source-registry facts in the data authority rather than duplicating them in scaling policy;
6. require exact corpus, split, tokenizer, post-tokenization and loss-mask identities before long training;
7. use the 20x figure as a compute-planning reference rather than a hard minimum, maximum or quality guarantee;
8. require an explicit scaling rationale for planned budgets materially below or above that reference;
9. keep explicit compute authorization separate from scientific readiness.

## Planning numbers

For exact MODEL-341 with 20,613,440 parameters:

- 20x reference: 412,268,800 total training-token exposures;
- 50x exploration point: 1,030,672,000 total training-token exposures;
- 100x exploration point: 2,061,344,000 total training-token exposures;
- rough dense-transformer training compute at 20x using `6 * N * D`: 50,989,669,036,032,000 FLOPs.

For future planning only:

| Stage | Parameters | 20x exposures | 50x exposures | 100x exposures |
| --- | ---: | ---: | ---: | ---: |
| 20M primary | 20,613,440 | 412,268,800 | 1,030,672,000 | 2,061,344,000 |
| 100M target | 100,000,000 | 2,000,000,000 | 5,000,000,000 | 10,000,000,000 |
| 1B target | 1,000,000,000 | 20,000,000,000 | 50,000,000,000 | 100,000,000,000 |

These are exposure-budget references. They do not state how many unique tokens the project currently owns, how many epochs should be run, whether 20x is sufficient for the target capability, or whether 50x/100x is affordable or optimal.

## Data-authority contract

The live source registry may report bytes and independent source families for acquisition planning. Those values remain external to this policy because they are volatile and because source admission is earlier than corpus materialization.

Promotion from source acquisition to learned-stage readiness requires terminal exact-head authority, an immutable materialized corpus, global deduplication, evaluation decontamination, quality/privacy review, deterministic split/shard/pack construction, tokenizer identity, exact unique post-tokenization token accounting, exact unique causal-loss accounting and an explicit replay/exposure plan.

No source-registry byte total, no source-capacity target, no replayed exposure count, no queued CI run and no stale snapshot can substitute for those artifacts.

## Long-training gate

A learned-stage long run remains fail-closed until the project has immutable evidence for:

- corpus identity and train-split identity;
- tokenizer identity;
- exact post-tokenization unique train-token count;
- exact unique causal-loss position count and document-boundary loss mask;
- evaluation decontamination;
- deduplication;
- quality/privacy review;
- checkpoint/resume integrity;
- preregistered total token-exposure budget;
- preregistered replay policy and epoch cap;
- preregistered stopping rule;
- explicit compute authorization.

The machine-readable policy and validator are in `configs/scaling/data_budget_policy_v1.json` and `tools/validate_scaling_data_budget_policy.py`.

## References

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.
- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264.
- Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448.
- Gadre et al., *Language models scale reliably with over-training and on downstream tasks*, arXiv:2403.08540.
- Meta, *Introducing Meta Llama 3*.
- Li et al., *DataComp-LM*, arXiv:2406.11794.
