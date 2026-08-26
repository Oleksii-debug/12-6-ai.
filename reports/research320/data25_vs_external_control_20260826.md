# RESEARCH-320 — DATA-25 vs external-real matched corpus-origin control

Worker: `RESEARCH-320-DATA25-VS-EXTERNAL-CONTROL`

Status: `BLOCKED_PRECONDITIONS_NOT_MET`

Execution profile: `LOCAL_FREE`

Recorded: 2026-08-26 14:04:18 UTC / 17:04:18 Europe/Uzhgorod

## Scientific question

Compare corpus origin while holding the approximately 500K model, tokenizer, optimizer, random seed set and optimized causal loss-position exposure fixed. External origin is not treated as a quality prior. The null hypothesis is that origin alone does not determine quality.

## Frozen matched contract

- Model: 467,808 parameters; ModelSpec SHA-256 `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a`.
- Tokenizer: `s0-byte-v1`, vocabulary 256, no special tokens.
- Optimizer: AdamW, lr 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0, clip norm 1.0, constant schedule, fp32.
- Batch size 8; sequence length 128.
- Seed set: `[1337]`. This is intentionally the singleton seed authority currently bound by the incumbent 500K contract; no additional seeds are invented.
- DATA-25 historical matched anchor: 948,504 optimized tokens.
- Exposure rule: both arms must consume exactly the same number of nonpadding optimized causal loss positions, with no replay or sampling with replacement; maximum exposure is `min(948504, terminal_external_real_one_pass_unique_optimized_targets)`.
- Best checkpoint: minimum aggregate BPB on common frozen immutable selection-validation only. Best and final remain separate.
- External-real final test is unread until selection is frozen.

## DATA-25 arm

DATA-25 identity is `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`. Its train split contains 20,000,775 byte tokens across 43,238 documents and was rebuilt twice with identical corpus identity and shard hashes. It contains project-authored data and zero external-training-eligible sources, so it is a mechanics/control corpus rather than evidence of external-source representativeness.

The historical 500K anchor has 948,504 optimized tokens and a recorded best/final BPB of 0.2645455968814711. That value is reference-only: it is not a RESEARCH-320 matched outcome and cannot establish a corpus-origin winner.

## External-real arm gate

The current DATA-300 v2 contract is executable as a build contract but explicitly says the corpus itself is `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`. The exact candidate inventory has 183,061 normalized bytes, five source objects and four independent families: UK 1, EN 1, code 2.

The family/balance authority sets the current family-constrained no-replay budget to 0. Additional hard blockers remain for quality, privacy, balance/diversity, nonempty immutable selection-validation, the full unique-loss-position ledger, and two independent clean builds. Therefore there is no positive authorized external-real exposure that can be matched to DATA-25 without violating the frozen corpus contract.

## Evaluation gate

EVAL-233 preserves an immutable 16-document UA/EN final test, but it is not selection-eligible, hyperparameter-selection-eligible or tokenizer-fit-eligible. Its evaluation release is currently blocked because external-corpus decontamination cannot be completed without a terminal corpus identity.

Common immutable selection-validation currently contains 0 documents. Thus no checkpoint selection can lawfully freeze, and the final test cannot be opened for RESEARCH-320.

## Run result

Optimizer updates executed: **0**.

Matched training executed: **no**.

Selection evaluation executed: **no**.

Final-test evaluation executed: **no**.

This is a fail-closed scientific result, not an infrastructure failure. Executing either training arm now would create an unmatched or policy-invalid comparison.

## Requested outcomes

| Outcome | RESEARCH-320 result | Reason |
| --- | --- | --- |
| Cross-corpus transfer | `NOT_MEASURABLE_YET` | No matched external training arm and no common nonempty immutable selection set. |
| Memorization | `NOT_MEASURABLE_YET` | Requires matched trained checkpoints plus train-vs-selection/final-test gaps; final test remains sealed. |
| Source-family generalization | `NOT_MEASURABLE_YET` | Requires matched checkpoints and family-labeled common selection/final-test evaluation. |
| Corpus-origin winner | `NONE` | No scientific comparison has executed; external origin receives no assumed advantage. |

## Preregistered measurements after unblock

Cross-corpus transfer: aggregate/UK/EN/code selection BPB, family BPB, external-final-test BPB after selection lock, and paired BPB deltas between arms.

Memorization: train-to-selection BPB gap, train-to-final-test BPB gap after lock, zero-overlap verification against evaluation, and authorized continuation-copy diagnostics without exposing reserved text.

Source-family generalization: per-family BPB, macro family BPB, worst-family BPB and family dispersion.

## Unblock conditions

1. Publish a terminal deterministic external-real corpus identity and final training shards.
2. Prove a positive one-pass unique optimized-loss-position capacity after family/balance constraints, with no replay.
3. Publish a nonempty terminal immutable common selection-validation authority excluded from both training arms and final test.
4. Complete quality, privacy, dedup/decontamination, diversity and two-clean-build gates.
5. Re-resolve all authorities and hashes immediately before optimizer step 1 and compute the exact common exposure.

Until all five conditions hold, `NO_SCIENTIFIC_RESULT_BLOCKED` is the only authorized RESEARCH-320 verdict.
