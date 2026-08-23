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
   D04 can collate its packed examples into this form explicitly, preventing a double
   shift.

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

The trainer refuses checkpointing while accumulation statistics are pending. It also
enters a fail-closed poisoned state after a non-finite loss/gradient, backward failure,
or ambiguous optimizer/scheduler failure. Once poisoned, that Trainer object refuses
further training, serialization, and in-place `load_state_dict()` recovery.

This fresh-instance rule is intentional: trainer-only state cannot prove model weights
were restored if an optimizer transition may have partially committed. Canonical
failure recovery is therefore:

1. verify the D05 checkpoint and all required identities/hashes;
2. restore/construct the checkpoint model weights;
3. construct a **fresh** Trainer with the exact training configuration;
4. load the verified trainer-owned optimizer/scheduler/scaler/counter state;
5. restore RNG state under the D05 reproducibility contract;
6. continue from the committed optimizer boundary.

A clean Trainer can load a verified `TrainerState` normally. D05 owns the durable file
format and full model/RNG restore ordering; D02 owns the trainer-side safety contract.

## Numerical safety invariants

- Loss must be finite before backward proceeds.
- Gradients must be finite after unscale and before an optimizer update.
- Gradient accumulation is exact over valid target tokens.
- An optimizer/scheduler step occurs only at an accumulation boundary.
- Partial or failed accumulated gradients cannot be checkpointed as completed state.
- Ambiguous failed transitions require fresh-instance checkpoint recovery.
- Gradient clipping happens after unscale and token normalization, before the update.
- Telemetry reports real loss, update loss, applied learning rate, pre-clip gradient
  norm, token count, micro-step, optimizer-step, and whether the optimizer stepped.
- `state_dict` exposes trainer-owned optimizer/scheduler/scaler/counter state for
  D05 serialization and deep-copies mutable optimizer/scheduler state.
- Resume refuses trainer-config mismatch and inconsistent/corrupt counters.
- Warmup scheduling starts at `1 / warmup_steps` of base LR and reaches base LR on
  the final warmup update without an off-by-one duplicate first rate.
- Hyperparameter configuration rejects non-finite values, invalid ranges, unsupported
  scheduler/precision modes, and type/value mismatches before training starts.

## Precision

S0 CPU smoke tests use fp32. bf16 autocast is available where the selected backend
supports it. fp16 is rejected on CPU rather than pretending to run mixed precision.
GPU numerical parity remains a separate evidence requirement for later stages.

## Current S0 integration status

The D02 trainer is model-agnostic and has deterministic CPU convergence evidence on
a test-only learnable bigram stub. That stub is not a canonical model or capability
evidence.

The earlier D01/D04 vocabulary mismatch is resolved at the current contract level.
D01 PR #24 uses S0 `vocab_size=256` with exactly 10,140 trainable parameters. D04 PR
#23 now uses a raw UTF-8 byte tokenizer with IDs `0..255`, vocab 256, no semantic
special tokens, so tokenizer/model vocabulary compatibility is exact. D04 can produce
raw shifted labels or aligned `target_ids + loss_mask`, both supported by D02.

D03 provides the deterministic controlled train/validation corpus, and D05 provides
the checkpoint substrate. D10 currently owns selective composition and has only a
partial experimental composition, not a full D01-D08 S0 candidate.

Final S0 training acceptance therefore still requires one exact D10-composed green
candidate containing D01-D08, a real CPU D01+D03+D04 training-loss decrease with
finite numerics, D05 save/load/interrupted-resume evidence, D06 held-out/stage-gate
evidence, D07 generation from the trained/reloaded checkpoint, and independent audits
bound to that exact candidate SHA.
