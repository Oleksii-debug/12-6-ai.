# RESEARCH-140 practical paired-run decision rules

## Scope

This policy is for architecture, tokenizer-model, optimizer, and schedule research comparisons. It is deliberately narrower than release evaluation. A research winner may be selected only from a predeclared `selection_validation` metric. Training metrics and diagnostic suites may explain behavior but cannot choose the winner. Final-test metrics are rejected by the selector and are not inputs to the research decision report.

The rule is designed for the small repeat counts that are realistic for 12-6. It does not turn two or three seeds into asymptotic hypothesis tests.

## Paired repeat unit

A repeat is a matched seed/run in which baseline and candidate share the intended controls: corpus/split identity, tokenizer where applicable, optimized-token or compute budget, initialization/data trace policy, evaluation identity, and all non-varied hyperparameters. Different model scales, different validation splits, or different training budgets are not silently pooled as seed repeats.

For lower-is-better metrics the oriented paired delta is `baseline - candidate`; for higher-is-better metrics it is `candidate - baseline`. Positive therefore always means the candidate is better.

## Reported statistics

For every comparison the selector records:

- mean and median oriented paired delta;
- sample variance and standard deviation across paired deltas;
- candidate/baseline win counts and exact numeric ties;
- effect size as paired mean delta divided by observed run-to-run standard deviation;
- a deterministic empirical bootstrap interval for the mean delta;
- the explicit practical materiality threshold;
- a repeat-count recommendation.

For `n <= 5`, the bootstrap distribution is fully enumerated (`n^n` resamples). For larger `n`, a fixed-seed bootstrap is used. The interval is descriptive empirical uncertainty, not a p-value or significance claim.

## Decision rule

`INSUFFICIENT_REPEATS` is mandatory below three paired repeats. A large one-seed delta can justify a follow-up, not a promotion.

At three or more repeats, `CLEAR_WIN` requires all of the following in one direction: mean and median clear the materiality threshold; the bootstrap interval stays on the winning side of zero; at least 75% of repeats win; and the paired signal is at least 0.8 observed run-noise standard deviations (or all observed deltas are identical and nonzero). `CLEAR_WIN` can name either the candidate or the baseline.

`PRACTICAL_TIE` requires the mean, median, and complete descriptive bootstrap interval to remain inside the `[-materiality, +materiality]` band. This is intentionally stronger than saying the point estimate is small.

Anything else at three or more repeats is `UNSTABLE`: sign reversals, a wide interval, or a material point estimate that is not repeatable enough.

## Materiality

Materiality is part of the experiment definition and must be fixed before winner selection. It is metric-specific; RESEARCH-140 does not impose one universal BPB/loss/accuracy number. Existing precommitted thresholds should be reused when available. The retroactive report uses an explicit 0.005 held-out-loss floor for small fixture comparisons where no stronger precommitment exists, and uses the repository's precommitted TRAIN-49 0.5% validation-loss scale where that experiment already defined one.

A numerical delta smaller than materiality cannot become a research winner merely by winning every seed.

## Repeat planning

The minimum promotion repeat count is three, not ten. `CLEAR_WIN` and `PRACTICAL_TIE` require no automatic extra seeds. `UNSTABLE` results use observed paired standard deviation to estimate how many repeats would make an approximate 90% half-width no larger than the materiality floor. Automatic exploratory planning is capped at seven total repeats. If the comparison is still unstable at that cap, improve the experiment design, budget, data, or effect size rather than automatically purchasing more seeds.

## Retroactive findings

The explicit three-seed MODEL-13 supporting h2-vs-h4 paired deltas all favor h2 numerically but average only about `7.4e-5` held-out loss. Under a 0.005 materiality floor the result is `PRACTICAL_TIE`, demonstrating why win count alone is insufficient.

TRAIN-43 warmup has executed one-seed evidence. Both the tiny ~1M difference and the large ~100K difference remain `INSUFFICIENT_REPEATS` for promotion; they are follow-up evidence, not repeatability evidence.

TRAIN-48 beta2 has one source-equivalent run per model scale in the inspected evidence. The two scales are not pooled as two seeds. Each research winner decision is therefore `INSUFFICIENT_REPEATS` despite material-looking point estimates.

TRAIN-50 declares two schedule seeds but the inspected branch/PR does not retain completed result values. Two seeds are useful exploration; a proposed promotion should add one paired repeat rather than jumping to a ten-seed campaign.

MODEL-09 depth/width and TOK-37 model-informed tokenizer selection are also single-seed/config-limited from the inspected retained surfaces. Deterministic tokenizer mechanics remain valid, but stochastic model-quality promotion requires paired repeats.
