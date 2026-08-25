# RESEARCH-06 fixed-token scaling + MODEL-09 ~500K depth/width

## Live incumbent

This work is intentionally stacked on RESEARCH41 PR #162 at exact source
`9ff78ea31c34fd434015d5bc512596ce5dac766a`.  RESEARCH41 already established a
fixed research family at 95,568 / 267,912 / 467,808 / 1,037,696 parameters using
the real `TwelveSixDecoder`, D02 `Trainer`, byte tokenizer, S0 packaged train and
held-out validation splits, fixed seed/init/optimizer, context 256 and one fixed
cyclic batch trace.  This package extends that incumbent rather than creating a
second scaling framework.

## Token-accounting defect repaired

RESEARCH41 labeled its checkpoints 4,096 / 16,384 / 65,536 requested optimized
tokens.  The frozen batch trace has batch size 4 and sequence length 64, hence
exactly `4 * (64 - 1) = 252` valid shifted causal loss targets per optimizer
update.  The prior runner used `ceil` and therefore actually evaluated after
4,284 / 16,632 / 65,772 optimized tokens.

Those historical observations were still equal-token comparisons across all four
models, but they did not satisfy an exact requested-budget contract.  RESEARCH-06
makes the actual reachable counts the explicit budgets and fails closed for any
budget that is not divisible by 252.  No partial update, silent masking change,
rounding or altered batch trace is introduced.

At every update the runner proves:

- `StepMetrics.tokens == 252`;
- `Trainer.tokens_seen == optimizer_step * 252`;
- every checkpoint lands exactly on its declared budget;
- held-out evaluation leaves model state, Trainer state, optimizer step and
  optimized-token count unchanged;
- evaluation contributes exactly zero optimized tokens.

At 16,632 optimized tokens every candidate is saved with the existing D05
checkpoint-v1 adapter, verified, restored into newly constructed model/Trainer
objects, fingerprint-compared, and then continued.  Any identity or accounting
drift aborts the run.

## Fixed controls

Both experiments use the same controls:

- byte tokenizer / vocabulary size 256;
- model maximum context 256;
- `research41-byte-stream-cyclic-v1` train trace;
- packaged S0 train split only for optimization and packaged validation split only
  for held-out evaluation;
- `InitSpec v1`: Normal(0, 0.02) with residual `sqrt(2L)` scaling;
- AdamW, lr `3e-4`, betas `(0.9, 0.95)`, eps `1e-8`, weight decay 0;
- constant learning rate, no warmup, gradient clip 1.0;
- fp32 CPU, seed 1337, deterministic algorithms;
- batch size 4, sequence length 64, gradient accumulation 1;
- exact optimized-token budgets 4,284 / 16,632 / 65,772.

The packaged S0 corpus is intentionally tiny and recycled.  Results are controlled
local generalization evidence, not representative-corpus scaling law or stage
promotion evidence.

## MODEL-09 predeclared candidate family

The candidate set is fixed before validation is observed.  Only model geometry
changes.  All models use MHA with four query/four KV heads and tied token
embedding/output weights.

| label | layers | d_model | head_dim | d_ff | exact params |
| --- | ---: | ---: | ---: | ---: | ---: |
| shallow_wide | 2 | 136 | 34 | 384 | 496,808 |
| mid_shallow | 3 | 112 | 28 | 320 | 502,544 |
| balanced | 4 | 96 | 24 | 280 | 495,456 |
| deep_narrow | 6 | 80 | 20 | 224 | 497,680 |
| very_deep_narrow | 8 | 72 | 18 | 184 | 503,496 |

`ModelSpec.parameter_breakdown()` is the parameter authority.  It includes the
256×`d_model` tied embedding matrix exactly and requires `lm_head_extra == 0`.
No new ModelSpec abstraction or approximate parameter counter is introduced.

## Measurements and decision rules

Each run retains held-out validation loss and byte-token BPB (`loss / ln(2)`) at
every exact budget; the `6*N*T` compute proxy; train loss as optimization telemetry
only; normalized pre-clip gradient norms; clip frequency; update/weight ratios;
step and end-to-end wall time; exact model and AdamW tensor-state bytes; process
RSS high-water mark; checkpoint size/save/load time and exact resume proof.

MODEL-09 additionally records per-block activation RMS/max magnitude, normalized
block gradient norm and block update/weight ratio over the whole trajectory.

RESEARCH-06 ranks the four sizes by final held-out validation loss, validation
improvement per parameter, per `6*N*T`, and per end-to-end wall second.  The
predeclared primary-research-vehicle rule is the smallest candidate whose final
held-out loss is within 5% relative of the best observed model at 65,772 tokens.
This deliberately trades a small loss gap for a materially cheaper iteration
vehicle when justified by measurements.

MODEL-09 selects the predeclared ~500K geometry with the lowest final held-out
loss; exact ties are broken only by lower median optimizer-step time.  No
additional geometry is introduced after reading validation results.

## Authority boundary

The dedicated GitHub Actions workflow uses the repository's exact locked Linux
runtime and LOCAL_FREE CPU resources.  Machine JSON is self-hashed and validated
before upload.  No paid compute, stage freeze, promotion, representative scale
corpus claim, capability claim, instruction tuning or foreign pretrained weights
are introduced by this experiment.
