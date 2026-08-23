# D02 training engine contract

Status: S0 implementation package. This module deliberately does **not** own model
architecture, tokenizer IDs, dataset semantics, or durable checkpoint file formats.

## Batch/model boundary

A microbatch is a mapping containing `input_ids: LongTensor[B, T]` and optionally
`labels: LongTensor[B, T]`. If labels are omitted, `input_ids` are used as labels.
The model is called as `model(input_ids)` and may return:

- a logits tensor `[B, T, V]`;
- a mapping containing a `logits` tensor; or
- an object with a `.logits` tensor.

`causal_lm_loss` shifts internally so logits at position `t` predict label `t + 1`.

## Safety invariants

- Loss must be finite before backward proceeds.
- Gradients must be finite before an optimizer update.
- Gradient accumulation divides loss by the exact configured accumulation count.
- An optimizer/scheduler step occurs only at an accumulation boundary.
- Partial accumulated gradients are never silently presented as a completed step.
- Gradient clipping happens after unscale and before the optimizer update.
- Telemetry reports real loss, learning rate, pre-clip gradient norm, token count,
  micro-step, optimizer-step, and whether the optimizer actually stepped.
- `state_dict` exposes trainer-owned optimizer/scheduler/scaler/counter state for
  D05 serialization. It refuses resume under a mismatched trainer configuration.

## Precision

S0 CPU smoke tests use fp32. bf16 autocast is available where the selected backend
supports it. fp16 is rejected on CPU rather than pretending to run mixed precision.

## Current S0 integration status

The trainer is model-agnostic and tested with a test-only learnable bigram stub.
That stub is not a canonical model. Final S0 convergence evidence remains blocked
until D01 publishes the real ~10K random-init model and D03/D04 publish the
deterministic corpus/token path.
