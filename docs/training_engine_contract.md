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
