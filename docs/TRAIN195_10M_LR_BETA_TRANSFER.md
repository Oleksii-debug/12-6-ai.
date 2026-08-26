# TRAIN-195 — 10M LR / beta2 transfer

## Current decision

`BLOCKED_MISSING_PREREQUISITE_AUTHORITY`.

This is a scientific fail-closed result, not a failed implementation and not a numerical optimizer comparison. At preregistration time there is no recoverable TRAIN-125 authority that identifies the strongest small-scale LR-transfer prediction, and there is no accepted TRAIN-194 clipping decision. Absolute LR candidates and a clipping threshold are therefore deliberately absent.

RECOVER-180 PR #332 explicitly states that it claims no clipping result and remains blocked until a non-pathological 10M clipping decision exists. TRAIN-195 must not substitute clip=1.0: existing 10M and TRAIN-48 evidence shows that threshold can saturate essentially every update, which would confound the intended beta2 second-moment transfer test.

## Evidence consumed

TRAIN-48 PR #208 compared beta2 0.95 / 0.98 / 0.99 for 128 updates at ~95K and ~468K. Its source-equivalent LOCAL_FREE evidence was finite and monotonically favored 0.99 in held-out loss, but clip=1.0 activated on every observed update. RESEARCH-140 PR #284 therefore correctly classifies the 0.99-vs-0.95 result as `INSUFFICIENT_REPEATS`: model scales are not seed replicates, and fewer than three paired repeats cannot promote.

The primary 10M beta2 transfer pair is preregistered as 0.95 versus 0.99. Beta2=0.98 is not discarded scientifically; it is retained only as a conditional midpoint if 0.99 is genuinely unstable while 0.95 is stable and the quality/time-to-quality tradeoff remains materially unresolved. This avoids reopening a full beta grid.

## Staged design after the blockers clear

Stage A tests three learning rates at 0.8x / 1.0x / 1.25x the exact TRAIN-125 prediction, holding beta2=0.95. The ratios are log-symmetric and intentionally narrow. No absolute LR is materialized before TRAIN-125 authority is bound.

Stage B takes the Stage-A selected LR and compares beta2=0.95 versus 0.99. Each compared stage requires at least three paired seeds, with identical initial weights and identical data traces within each seed. A one-seed run may reveal a fatal instability but cannot select or promote a winner.

For beta2=0.99, each candidate trajectory must run for at least 300 optimizer updates, corresponding to three nominal `1/(1-beta2)` second-moment time constants. Longer execution may be inherited from the accepted 10M campaign, but it may not be shortened below this floor merely to save wall time.

## Frozen variables

Before the first optimizer update, gradient clipping, weight decay, epsilon, schedule family, batch geometry, model/tokenizer/corpus identities, and beta1 must be immutable. Epsilon=1e-8 and beta1=0.9 are retained from the controlled AdamW lineage. The remaining control values are intentionally unresolved because the missing TRAIN-125/TRAIN-194 authorities are the sources that must settle them; unrelated historical recipes disagree on weight decay and clipping and are not silently mixed.

## Measurements and numerical safety

The execution must retain selection-validation BPB, training loss, pre/post global gradient norms, post/pre norm ratio, update/weight ratio, raw and bias-corrected `exp_avg_sq` RMS/quantiles, clip frequency, nonfinite gradient/state counts, loss spikes, optimized tokens, wall time, and time-to-quality.

NaN/Inf checks must occur before clipping. A nonfinite gradient, optimizer state, loss, or parameter is a fatal numerical failure and can never be hidden by clipping.

## Decision rule

Use the current RESEARCH-140 paired decision protocol. Promotion requires at least three paired repeats and a `CLEAR_WIN` that is material, directionally consistent, and stable under the current bootstrap/uncertainty rule. `PRACTICAL_TIE` retains the less intrusive incumbent. `UNSTABLE` and `INSUFFICIENT_REPEATS` cannot promote. Final-test metrics are excluded from winner selection.

## Execution environment

This branch is stacked on ENV-151 exact head `c127216ecf9722bea1964c7488cb7ff0f8cdebe4` and its universal exact execution bootstrap. The gate workflow uses the `runtime,tests` capability closure. `LOCAL_FREE` only; `paid_compute=false`.

## Truth boundary

No TRAIN-195 training trajectory has been executed from this preregistration. No validation BPB, time-to-quality result, beta2 winner, LR winner, or 10M optimizer default is claimed. The current deliverable is a machine-enforced prerequisite gate that prevents the swarm from fabricating the two missing upstream decisions.
