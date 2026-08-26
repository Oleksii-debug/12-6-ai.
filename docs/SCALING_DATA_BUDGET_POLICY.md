# Scaling Data-Budget Policy

## Decision

The project must keep **model parameter count**, **source bytes**, **post-tokenization training tokens**, and **unique causal-loss positions** as separate quantities.

The current primary architecture has 20,613,440 parameters. At the fail-closed live cutoff recorded by this policy, NEXT100-063 V2 / PR #538 reports 266,476 pre-global-dedup source bytes across 10 independent families and a separate frozen acquisition target of 20,000,000 source bytes. It authorizes zero balanced no-replay loss positions. None of those source-byte numbers is a training-token count, and none can authorize a learned-20M long run.

The live source snapshot is provenance-bound in the machine policy to PR #538 V2 head `7da63b7d85b65b1508ef5c7d73bfa8d56e718c9f` and registry identity `934933896a4b3b01dd58cd18d13bcc36245913f83412c6b3f697c64dd03e4d4d`. V2 supersedes the earlier V1 vector because V1 incorrectly credited unsupported CPython accepted-byte capacity and exact-head-failed Pydantic/Rich admissions. A newer source authority requires an explicit policy refresh; stale numbers must not drift silently.

## Research reference

Hoffmann et al. (2022) found that compute-optimal model size and training-token count should scale together. The commonly used Chinchilla planning point is approximately 20 training tokens per parameter. This repository uses **20x only as a planning reference**, not as a universal optimum or a model-quality guarantee.

Later work strengthens the reason not to treat 20x as a ceiling. Data-constrained scaling work shows that repeated data has diminishing value; inference-aware scaling and over-training studies show that smaller models can rationally be trained on substantially more tokens; Llama 3 reports continued gains far beyond its Chinchilla-optimal token count. DataComp-LM and FineWeb/FineWeb-Edu also show that curation and filtering quality materially change outcomes.

Therefore the 12-6 AI policy is:

1. do not convert source bytes to tokens by assumption;
2. do not count replayed tokens as unique loss positions;
3. require exact corpus, split, tokenizer, post-tokenization token-count and loss-mask identities before long training;
4. treat a run below the 20x planning reference as a bounded scaling/smoke experiment unless an explicit preregistered scaling experiment justifies a different budget;
5. choose larger data budgets from small-scale scaling experiments, evaluation results and compute constraints rather than from an arbitrary fixed multiplier;
6. keep paid compute authorization separate from scientific readiness.

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
- a preregistered token/step budget and stopping rule.

The machine-readable policy and validator are in `configs/scaling/data_budget_policy_v1.json` and `tools/validate_scaling_data_budget_policy.py`.

## References

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.
- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264.
- Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448.
- Gadre et al., *Language models scale reliably with over-training and on downstream tasks*, arXiv:2403.08540.
- Meta, *Introducing Meta Llama 3*.
- Li et al., *DataComp-LM*, arXiv:2406.11794.
