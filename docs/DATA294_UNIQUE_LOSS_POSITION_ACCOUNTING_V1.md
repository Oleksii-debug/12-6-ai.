# DATA-294 — Unique Loss Position Accounting V1

Worker: `DATA-294-UNIQUE-LOSS-POSITION-ACCOUNTING-V1`

Mode: `LOCAL_FREE` only. No long training is executed by this work unit.

## Purpose

DATA-294 defines the exact unit that a future external-real Trainer run is allowed to
consume: a document-local causal loss position, not a source byte count, padded tensor
slot, packed window slot, optimizer step, or epoch.

The authority is intentionally built on the exact DATA-229 real-snapshot registry
identity `1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486`.
It does not invent DATA-230 or MILESTONE-238 authority and does not promote the current
three-document DATA-229 inventory into a research-scale corpus.

## Exact position semantics

For V1 the tokenizer is frozen to `s0-byte-v1` with exact config and vocabulary hashes.
Every normalized UTF-8 byte is exactly one token. BOS/EOS are absent and documents are
isolated.

For each training-authorized source document:

1. bind the exact normalized payload SHA-256 and normalized byte length from DATA-229;
2. materialize all reserved-evaluation byte ranges as half-open `[start, end)` offsets;
3. split the document at every reserved range before token accounting;
4. treat each remaining contiguous segment independently;
5. for a segment with `n` tokens, authorize target indices `1 <= i < n`;
6. therefore that segment owns exactly `max(n - 1, 0)` unique optimized targets.

A position key is `(segment_identity_sha256, target_token_index)`. Segment identity
binds source ID, normalized payload SHA-256, byte bounds, tokenizer config hash and the
position policy. No position crosses a source-document boundary or a reserved-eval gap.

Padding is storage only and contributes exactly zero optimized targets.

If the accepted tokenizer changes away from the exact byte tokenizer, V1 fails closed:
a new ledger must be built with an exact tokenizer-to-byte-span mapping. Source-byte
counts must not be relabelled as loss-token counts.

## Reserved evaluation bytes

The reservation manifest must exactly cover every source whose DATA-229
`model_training` right is `ALLOWED`. Unknown, missing, overlapping, out-of-bounds or
payload-mismatched reservation entries are rejected.

The current DATA-229 source records have no inline reserved-evaluation ranges, so the
committed DATA-294 manifest binds an empty range list for each of the three exact
training payloads. This is not a decontamination-clean claim. EVAL final-test bytes must
remain outside training sources, and future corpus construction must materialize any
record-level reserved ranges before rebuilding this ledger.

## Current one-pass maximum

Current DATA-229 input contains 173,358 normalized bytes across three independent
source documents. Document isolation removes one non-predictable initial token per
document, producing exactly 173,355 unique optimized targets in one pass.

| Dimension | Key | Documents | Normalized bytes | Reserved eval bytes | Unique targets |
| --- | --- | ---: | ---: | ---: | ---: |
| Language | `en` | 2 | 84,793 | 0 | 84,791 |
| Language | `uk` | 1 | 88,565 | 0 | 88,564 |
| Modality | `text` | 3 | 173,358 | 0 | 173,355 |
| Family | `en.standardebooks.manual` | 2 | 84,793 | 0 | 84,791 |
| Family | `ua.rada.open-data.laws-texts` | 1 | 88,565 | 0 | 88,564 |

Code contributes zero in this exact DATA-229 registry cutoff. DATA-227 and DATA-228 were
not terminally consumed by DATA-229 and are not silently added here.

Ledger identity:
`9a1cd57c52459bdc6e4bb2d46047a47713e10d9a5be7b0a4b86f041ba6f62bd0`

This means a 20M unique-target campaign is **not authorized** by the current corpus.
The current exact one-pass maximum is 173,355 targets. A later external-real corpus must
first publish its own rebuilt ledger and may authorize at most its exact one-pass total.

## Trainer replay guard API

`ExposureBudgetGuard` accepts an exact ledger and an authorized target budget. The
budget may not exceed the ledger one-pass maximum.

Before each Trainer microbatch, orchestration supplies exact segment-local target
intervals. `authorize_batch` rejects:

- any target interval outside the ledger;
- any overlap with previously consumed positions;
- any batch whose claimed positions do not equal the Trainer optimized-target count;
- any exposure that would exceed the authorized budget.

`train_microbatch_with_exposure` counts valid Trainer targets before execution, reserves
the positions, then calls `Trainer.train_microbatch`. Reservation occurs before
forward/backward. If training fails, the reservation remains consumed; blindly replaying
the same data is forbidden.

`ExposureBudgetGuard.state_dict()` is deterministic and self-hashed. Resume must restore
the matching Trainer checkpoint and the matching exposure state. The ledger identity,
authorized budget and consumed intervals are immutable across resume.

`assert_guarded_checkpoint_safe` requires `Trainer.tokens_seen == consumed_targets` at a
checkpoint boundary. A future long campaign must store both states atomically or fail
closed rather than reconstructing exposure from epoch/step counters.

## Truth boundary

DATA-294 proves exposure accounting mechanics and the exact current DATA-229 one-pass
capacity. It does not claim:

- DATA-230 is terminal;
- current source-family diversity is sufficient;
- current corpus is decontamination-clean against all reserved evaluation material;
- code exists in the current DATA-229 ledger;
- 20M unique targets are presently available;
- a non-byte tokenizer has the same target count.

A long 20M campaign is permitted only after a terminal external-real corpus authority
publishes enough eligible unique positions under a rebuilt deterministic ledger, with
all reserved evaluation bytes excluded and the Trainer run bound to the replay guard.
