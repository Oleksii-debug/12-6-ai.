# TRAIN-344 — ~20M optimizer transfer contract

Worker: `TRAIN-344-20M-OPTIMIZER-TRANSFER-CONTRACT`

## Decision

Preregister a narrow AdamW transfer from terminal learned-10M evidence, but do not authorize a learned ~20M optimizer comparison while the final-corpus no-replay ledger remains blocked.

The contract is frozen before any ~20M optimizer result. It has exactly three LR candidates and no beta, weight-decay, clipping, batch, schedule, warmup, precision or architecture sweep.

## Terminal 10M evidence consumed

`LEARN-217` terminally executed the 10,000,640-parameter S3 GQA Base under DATA-25 to 2,000,060 actual optimized causal targets with no replay. The direct executed optimizer/runtime anchor is AdamW LR `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, clip `1.0`, constant schedule, no warmup, sequence 256, microbatch 1, accumulation 1 and deterministic FP32 CPU.

That LR is an executed stable incumbent, not a proven optimum.

`TRAIN-243` then executed three paired seeds under the frozen learned-10M identity. `TRAIN-325` consumes the terminal result and retains gradient clip `1.0`: unclipped, q95 `3.4`, and q90 `2.1` did not satisfy the preregistered combined gates. The clip result is scoped to that tested learned-10M identity and is not a universal norm law.

`TRAIN-125` remains a bounded LR-scaling prior only. Its retained fit is:

`lr_best = 1.8126471043504123e-3 * (parameters / 100000)^(-0.4614142226617872)`

The nominal 10M→20M ratio from the exponent is `0.7262954159190345`. Applying it to the terminal executed 10M LR gives an anchor near `2.1789e-4`. The fit itself predicts about `1.5725e-4` at nominal 20M. Because neither value is a terminal 20M optimum, TRAIN-344 brackets them with only three preregistered mechanics candidates: `1.6e-4`, `2.2e-4`, `2.6e-4`.

No fourth LR may be added after observing a TRAIN-344 result.

## Frozen ~20M optimizer controls

- optimizer: AdamW;
- LR candidates: `1.6e-4`, `2.2e-4`, `2.6e-4`;
- beta1/beta2: `0.9 / 0.95`;
- epsilon: `1e-8`;
- weight decay: `0.1`;
- global gradient clip: `1.0`;
- schedule: constant;
- warmup: 0;
- sequence length: 256;
- microbatch: 1;
- gradient accumulation: 1;
- precision: FP32;
- deterministic algorithms: enabled.

TRAIN-344 does not invent a ~20M geometry. Before optimizer step 1, the probe runner requires an exact mechanically qualified ModelSpec from the preregistered ~20M model lane with parameter count inside 18M–22M. If no such config is supplied, the only valid state is `BLOCKED_MISSING_20M_MODELSPEC` with zero optimizer updates.

## Bounded stability probe

The probe is mechanics-only and uses deterministic synthetic token IDs. Every LR arm reconstructs identical random initial weights and consumes the same 32-batch trace.

Per arm:

- 32 optimizer updates;
- 255 causal targets per full sequence;
- exactly 8,160 optimized targets.

Across all three LR arms: exactly 24,480 optimized synthetic targets.

The probe records finite loss, pre-clip global gradient norm returned by the Trainer, clip activation, optimizer step, exact `tokens_seen`, parameter finiteness, a fixed parameter-slice update/weight diagnostic and CPU wall time.

Valid result labels are only `STABLE_MECHANICS_ONLY`, `UNSTABLE_MECHANICS_ONLY`, or `BLOCKED_MISSING_20M_MODELSPEC`. The probe has zero authority to select a best LR or claim quality.

## Learned token budget

`RESEARCH-313-20M-DATA-CAPACITY-GATE` currently authorizes exactly **0 unique nonignored causal loss positions** from a terminal final corpus for future ~20M learned work. DATA-301 remains terminal-blocked and reports zero authorized balanced no-replay capacity.

Therefore TRAIN-344 executes no learned-corpus optimizer comparison. The 24,480 synthetic probe targets are mechanics evidence and do not consume or enlarge a learned-data budget.

A later learned optimizer experiment requires a successor contract after a terminal corpus identity and exact immutable one-pass unique-loss ledger exist. That successor must freeze its exact learned token budget before optimizer step 1; TRAIN-344 does not pre-authorize it.

## Claim boundary

LOCAL_FREE only. No paid compute. No foreign weights. No SFT, RLHF or DPO. No architecture expansion. No huge sweep. No optimizer promotion, stage promotion, production-readiness claim, or learned ~20M quality result can come from this contract.
