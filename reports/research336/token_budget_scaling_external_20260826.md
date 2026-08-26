# RESEARCH-336 — external-real token-budget scaling

Worker: `RESEARCH-336-TOKEN-BUDGET-SCALING-EXTERNAL`

Execution profile: `LOCAL_FREE`

Status: `BLOCKED_NO_TERMINAL_EXTERNAL_REAL_TRAINING_CORPUS`

Recorded: 2026-08-26

## Mission

Measure ~500K / ~1M / ~3M model quality at multiple identical no-replay optimized-token checkpoints on Research Corpus V1, with no parameter arm receiving extra exposure, and report quality gain per parameter and compute.

## Result

No optimizer step is scientifically authorized at this cutoff. Therefore no numerical 500K/1M/3M scaling curve is reported and no quality, parameter-efficiency, or compute-efficiency winner is claimed.

This is a fail-closed scientific result, not a runtime failure.

## Current authority reconstruction

### Training corpus

DATA-300 v2 binds the five-source external-real candidate but explicitly records it as `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`.

DATA-301 is the terminal execution result for that candidate and reports `TERMINAL_BLOCKED`, with null final corpus identity, null final shard identity, no full five-source unique-loss ledger, and `authorized_balanced_no_replay_capacity = 0`.

DATA-305 later remains `BLOCKED_NO_EXACT_CORPUS_IDENTITY`; exact decontamination cannot execute without a terminal materialized training corpus identity and bound record inventory.

Thus there is no positive authorized Research Corpus V1 training exposure to spend.

### Selection-validation

EVAL-303 now composes a nonempty immutable selection-validation authority:

- composite identity: `7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd`;
- 10 records total;
- UA: 8;
- EN: 2;
- code: 0.

This removes the earlier empty-selection blocker for UA/EN checkpoint selection, but does not create or authorize a training corpus. Code-aware selection quality remains unavailable because the code selection stratum is still empty.

### No-replay capacity

DATA-294 measures 173,355 exact one-pass nonignored causal targets only for the older three-object DATA-229 text inventory:

- UA: 88,564;
- EN: 84,791;
- code: 0.

That ledger is not a full five-source final-corpus ledger and cannot be relabelled as authority for DATA-300/DATA-301 Research Corpus V1.

DATA-295/RESEARCH-313 retain the family-constrained authorized no-replay budget at 0 for the current candidate. Source bytes are not optimized-token counts, and replay, replacement sampling, duplicated documents, source aliases, epochs, or padding may not manufacture exposure.

## Parameter arms bound for the successor experiment

The currently discoverable scratch model authorities provide these exact trainable parameter counts:

| nominal arm | exact trainable parameters | ModelSpec identity |
| --- | ---: | --- |
| ~500K | 467,808 | `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a` |
| ~1M | 1,037,696 | `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07` |
| ~3M | 3,213,120 | `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc` |

Parameter deltas for scaling-efficiency reporting are therefore:

- 500K -> 1M: 569,888 parameters;
- 1M -> 3M: 2,175,424 parameters;
- 500K -> 3M: 2,745,312 parameters.

No parameter count is rounded inside the actual efficiency calculation.

## Frozen successor comparison contract

Execution may begin only after a successor authority publishes all of the following before optimizer step 1:

1. a terminal exact external-real Research Corpus V1 training identity and immutable shard identities;
2. an exact full-corpus unique optimized-loss-position ledger with positive one-pass capacity;
3. decontamination/quality/privacy/diversity gates applicable to the exact materialized corpus;
4. a nonempty immutable selection-validation authority for every modality whose quality will be claimed;
5. tokenizer identity and optimizer/training recipe frozen identically across all three parameter arms except for model size.

### Equal-exposure rule

Let `B_corpus` be the exact authorized one-pass unique optimized-target capacity of the terminal training corpus. Let `B_500`, `B_1M`, and `B_3M` be any independently preregistered local-compute ceilings for the three arms.

The experiment-wide budget is:

`B_common = min(B_corpus, B_500, B_1M, B_3M)`.

Every arm must consume exactly `B_common` nonpadding optimized causal loss positions, in the same frozen data order, with no replay and no replacement sampling.

No arm may continue past the common endpoint merely because it is cheaper or faster.

### Identical checkpoints

Checkpoint selection/evaluation is performed at the same cumulative optimized-target counts in every arm. Unless a successor preregistration freezes more granular exact counts, use the exact reachable positions nearest to:

- 25% of `B_common`;
- 50% of `B_common`;
- 75% of `B_common`;
- 100% of `B_common`.

The resulting integer target counts are computed once from the frozen ledger, recorded before training, and reused unchanged for all parameter arms. Initialization at 0 exposure may be evaluated descriptively but is not a trained checkpoint.

### Randomness

Use paired seeds across arms. Each included seed must receive the same optimized-target exposure in every parameter arm. A seed missing from any arm is excluded from direct paired scaling claims rather than compensated with extra runs in another arm.

### Quality metric

Primary quality metric: immutable selection-validation aggregate BPB, lower is better.

Also report:

- UA BPB;
- EN BPB;
- code BPB only when a nonempty independently authorized code selection set exists;
- source-family BPB where labels are available;
- macro modality BPB and worst-modality BPB where all claimed modalities are represented.

Best and final checkpoints remain separate. Final-test material is not used to choose model size, token budget, tokenizer, checkpoint, or hyperparameters.

## Scaling-efficiency metrics

For checkpoint `c`, let `Q_m(c)` be aggregate BPB for model arm `m`. For a smaller arm `a` and larger arm `b`, define positive quality gain as:

`DeltaQ(a->b,c) = Q_a(c) - Q_b(c)`.

Positive values mean the larger model improved BPB; negative values mean it regressed.

### Quality gain per added parameter

`parameter_efficiency(a->b,c) = DeltaQ(a->b,c) / ((P_b - P_a) / 1,000,000)`

Report as BPB reduction per one million added trainable parameters for:

- 500K -> 1M;
- 1M -> 3M;
- 500K -> 3M.

### Quality gain per compute

Record cumulative training FLOPs by the repository's common scaling-compute accounting method at each exact checkpoint. Also record wall-clock CPU time as an operational secondary metric, not as the primary cross-machine efficiency denominator.

For paired arms at the same optimized-token checkpoint:

`compute_efficiency(a->b,c) = DeltaQ(a->b,c) / ((F_b(c) - F_a(c)) / 1e12)`

Report as BPB reduction per additional 1e12 training FLOPs. If the denominator is nonpositive because of an accounting defect, fail closed rather than emitting a ratio.

Also report absolute BPB versus cumulative FLOPs for each arm so the incremental ratio is not interpreted without its underlying curve.

## Requested outcome table at this cutoff

| requested outcome | RESEARCH-336 result | reason |
| --- | --- | --- |
| 500K quality at identical checkpoints | `NOT_MEASURABLE_YET` | No authorized positive external-real training exposure. |
| 1M quality at identical checkpoints | `NOT_MEASURABLE_YET` | Same blocker. |
| 3M quality at identical checkpoints | `NOT_MEASURABLE_YET` | Same blocker; prior LEARN-319 also failed closed. |
| 500K -> 1M gain/parameter | `NOT_MEASURABLE_YET` | Requires paired BPB at identical exposure. |
| 1M -> 3M gain/parameter | `NOT_MEASURABLE_YET` | Requires paired BPB at identical exposure. |
| 500K -> 3M gain/parameter | `NOT_MEASURABLE_YET` | Requires paired BPB at identical exposure. |
| gain/compute | `NOT_MEASURABLE_YET` | No authorized training FLOPs were spent. |
| scaling winner | `NONE` | No valid numerical comparison executed. |

`NOT_MEASURABLE_YET` must not be converted to numeric zero. Zero would falsely assert measured equality.

## Execution accounting

- optimizer updates: **0**;
- optimized training targets consumed by RESEARCH-336: **0**;
- replayed targets: **0**;
- paid compute: **0**;
- final-test outcomes read: **no**;
- numerical scaling winner claimed: **none**.

No parameter arm received extra exposure; all three received zero because the shared prerequisite failed before training.

## Unblock condition

Run RESEARCH-336 only after a successor terminal corpus authority publishes an exact positive one-pass optimized-target capacity for the materialized external-real training corpus and every other hard gate above is satisfied. At that point bind `B_common`, the exact checkpoint vector, all model/tokenizer/optimizer identities, and the paired seed set before optimizer step 1.
