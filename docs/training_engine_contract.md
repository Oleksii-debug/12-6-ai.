# D02 training engine contract

Status: S0 implementation package. This module deliberately does **not** own model
architecture, tokenizer IDs, dataset semantics, or durable checkpoint file formats.

## Batch/model boundary

A microbatch always contains `input_ids: LongTensor[B, T]`. Two explicit causal-target
forms are supported and may not be mixed:

1. Raw/unshifted labels: optional `labels: LongTensor[B, T]`. If omitted, `input_ids`
   are used as labels. `causal_lm_loss` shifts internally so logits at position `t`
   predict label `t + 1`.
2. Already-aligned packed pairs: `target_ids: LongTensor[B, T]`, optionally with
   binary `loss_mask: Tensor[B, T]`. `causal_pair_loss` does **not** shift again.
   This is the contract used by D04 `PackedCausalExample` and prevents a double shift.

The model is called as `model(input_ids)` and may return:

- a logits tensor `[B, T, V]`;
- a mapping containing a `logits` tensor; or
- an object with a `.logits` tensor.

Token telemetry counts only actual optimized targets: shifted non-ignored tokens for
raw labels, or non-ignored targets selected by `loss_mask` for aligned packed pairs.

## Exact gradient accumulation

Accumulation is normalized by **valid target tokens**, not by the number of
microbatches. This matters for D04 padded tails and any future variable-token batch.
For each accumulation window the trainer:

1. backpropagates each microbatch's mean loss multiplied by its valid-token count;
2. accumulates those loss-sum gradients without stepping;
3. unscales once at the optimizer boundary;
4. divides accumulated gradients by the exact total valid-token count;
5. measures the pre-clip normalized gradient norm, clips if configured, then steps.

This is equivalent to one combined batch under the same token-level mean objective,
even when microbatches have different `loss_mask` cardinalities. Tests compare the
resulting AdamW parameter update directly against a single combined batch.

`StepMetrics.loss` is the real mean loss of the current microbatch.
`StepMetrics.update_loss` is populated on optimizer boundaries with the exact
token-weighted mean loss of the full accumulation window. `learning_rate` is the rate
actually applied to that optimizer update, not the scheduler's next-step rate.

## Training loop and recovery

`Trainer.run()` consumes an arbitrary iterable of batch mappings until the configured
`max_steps` optimizer-step target. Dataset order, epochs, and sampling remain outside
D02 ownership. The loop:

- resumes from the current committed optimizer step rather than resetting counters;
- requires enough microbatches to reach `max_steps` and fails loudly on exhaustion;
- emits real per-microbatch metrics through an optional callback;
- invokes checkpoint hooks only after committed optimizer/scheduler steps;
- supports checkpoint cadence and always emits the final checkpoint hook at max steps;
- wraps checkpoint-hook failure as `CheckpointHookError` stating that the optimizer
  step is already committed and must not be blindly replayed.

A checkpoint-safe state must satisfy both conditions:

- no partial accumulation group is pending; and
- `micro_step == optimizer_step * gradient_accumulation_steps`.

The trainer also refuses checkpointing while an optimizer/scheduler update has an
ambiguous incomplete outcome or while accumulation statistics are pending. D05's
trainer adapter can therefore serialize `Trainer.state_dict()` without silently
losing accumulated gradients.

## Numerical safety invariants

- Loss must be finite before backward proceeds.
- Gradients must be finite after unscale and before an optimizer update.
- Gradient accumulation is exact over valid target tokens.
- An optimizer/scheduler step occurs only at an accumulation boundary.
- Partial or failed accumulated gradients cannot be checkpointed as completed state.
- Gradient clipping happens after unscale and token normalization, before the update.
- Telemetry reports real loss, update loss, applied learning rate, pre-clip gradient
  norm, token count, micro-step, optimizer-step, and whether the optimizer stepped.
- `state_dict` exposes trainer-owned optimizer/scheduler/scaler/counter state for
  D05 serialization and deep-copies mutable optimizer/scheduler state.
- Resume refuses trainer-config mismatch and inconsistent/corrupt counters.
- Warmup scheduling starts at `1 / warmup_steps` of base LR and reaches base LR on
  the final warmup update without an off-by-one duplicate first rate.

## Precision

S0 CPU smoke tests use fp32. bf16 autocast is available where the selected backend
supports it. fp16 is rejected on CPU rather than pretending to run mixed precision.

## Current S0 integration status

The D02 trainer is model-agnostic and has deterministic CPU convergence evidence on
a test-only learnable bigram stub. That stub is not a canonical model.

D01 PR #24 now provides a real decoder candidate and D03/D04 provide deterministic
data/token contracts, but an integration blocker was found: D01 S0 currently declares
`vocab_size=256`, while D04 frozen `s0-byte-v1` declares `vocab_size=259` with valid
IDs 0..258. Canonical D01+D04 integration must fail closed until these identities are
reconciled. Setting D01 S0 vocab to 259 would make the current 10,140-parameter design
10,200 parameters and remain near the S0 ~10K target.

Final S0 training acceptance also requires the exact integrated D01/D03/D04/D05/D06
candidate, observed CI, and independent audit evidence.
