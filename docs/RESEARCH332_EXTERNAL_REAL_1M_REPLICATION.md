# RESEARCH-332 — External-real ~1M multi-seed replication

Worker: `RESEARCH-332-1M-EXTERNAL-REPLICATION`

Execution profile: `LOCAL_FREE`

## Verdict

`BLOCKED_REFERENCE_RUN_NEVER_AUTHORIZED_NO_NONZERO_FROZEN_TRAINING_BUDGET`

No seed was trained. Optimizer updates across all replication seeds: **0**. No BPB or source-family result is claimed, and run-to-run variance is not estimable.

This is the only scientifically valid replication outcome under the currently published authority. The parent LEARN-318 contract reconstructs a 1,037,696-parameter random-init Base and the canonical tokenizer/optimizer, but its exact DATA-300 v2 training authority has `family_constrained_no_replay_budget = 0`, `corpus_state = NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`, and explicitly forbids optimizer step 1. A multi-seed experiment cannot manufacture a reference learning run that never existed.

## What is frozen

The replication control is bound to the LEARN-318 authority gate and keeps every non-seed training choice fixed:

- model: decoder-only Base, exactly **1,037,696 parameters**;
- ModelSpec SHA-256: `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07`;
- initialization lineage: random initialization only; no foreign pretrained weights;
- tokenizer: `s0-byte-v1`, vocabulary 256, no special tokens;
- tokenizer config SHA-256: `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`;
- tokenizer vocab SHA-256: `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`;
- optimizer: AdamW, LR `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0`;
- schedule: constant, no warmup;
- global gradient clipping: `1.0`;
- precision: FP32;
- packing: document-isolated, sequence length 128;
- batch size: 8;
- corpus contract identity: `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`;
- DATA-300 source authority SHA: `8ea7f830e50a23754d189dd4134f4afad76a7ee9`;
- current exact candidate inventory: 5 source objects, 183,061 admitted normalized bytes;
- candidate strata: Ukrainian 88,565 bytes, English 84,793 bytes, code 9,703 bytes.

The phrase “~1M” is model scale here, not evidence of a one-million-token training budget. The only currently authorized optimized-target budget is **0**.

## Preregistered seed panel

Once a successor corpus authority makes the exact replication executable, use five independent initialization seeds:

`1337, 2027, 4099, 7919, 104729`

Seed is the only intended experimental factor. Corpus identity, source membership, split identities, tokenizer, ModelSpec, optimizer hyperparameters, precision, packing, sequence length, batch size, checkpoint fractions, evaluation sets and realized optimized-target budget must be byte-/value-identical across seeds. Data order must be fixed independently of the initialization seed so the reported variance isolates initialization/training stochasticity rather than corpus resampling.

No seed may receive sampling with replacement, document replication, replay, alias inflation, or padding counted as data.

## Frozen token-budget rule

For every seed, the training budget is identical and must be fixed before model-quality observation:

`B = min(reference_preregistered_budget, terminal_family_constrained_one_pass_unique_train_optimized_targets)`

Under the currently published DATA-300 v2 authority, `B = 0`.

If a successor terminal corpus becomes available, the successor run must publish one exact nonzero `B`; that exact same `B` is then used for all five seeds. A seed that fails early is a failed/censored run, not permission to redistribute its unused budget to another seed.

## Required evaluation protocol when unblocked

For every seed:

1. Train from a fresh random initialization under the frozen control.
2. Save chronological checkpoints at identical 0/25/50/75/100% budget boundaries.
3. Select best checkpoint only by minimum aggregate BPB on the frozen immutable selection-validation authority.
4. Retain chronological final separately from selected best.
5. Run a fresh-process checkpoint resume proof.
6. Prove evaluation does not mutate model or Trainer state.
7. Keep final-test bytes/outcomes inaccessible until checkpoint selection is frozen.
8. Report BPB for aggregate, Ukrainian, English, code, and every admitted source family with enough held-out bytes to score honestly.

Code-aware selection must not be silently inferred from an empty code-selection component; a nonempty code evaluation authority is required for any code-selection claim.

## Variance report specification

Primary statistic: aggregate selection-validation BPB of the frozen selected checkpoint for each seed.

For aggregate BPB and each UA/EN/code/source-family BPB, report:

- all five seed values;
- arithmetic mean;
- sample standard deviation (`ddof=1`);
- minimum and maximum;
- range;
- median;
- coefficient of variation where the mean is nonzero;
- 95% t-interval for the mean, explicitly marked low-powered at `n=5`;
- best-versus-worst seed delta in absolute BPB and percent relative to the five-seed mean.

Also report whether source-family ranking is stable across seeds. Do not collapse a family to a language average if family-level held-out evidence exists.

A replication claim requires at least 4/5 completed seeds. If fewer than four complete, the verdict is `INSUFFICIENT_REPLICATION_COMPLETION`; do not publish a stable-variance claim.

## Current replication matrix

| Seed | Training | Optimizer updates | Aggregate BPB | UA BPB | EN BPB | Code BPB | Source-family BPB |
|---:|---|---:|---|---|---|---|---|
| 1337 | BLOCKED | 0 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 2027 | BLOCKED | 0 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 4099 | BLOCKED | 0 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 7919 | BLOCKED | 0 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 104729 | BLOCKED | 0 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

Variance: `NOT_ESTIMABLE_NO_COMPLETED_SEEDS`.

## Independent blocker confirmation

Later repository evidence does not unblock training:

- DATA-301 reports `TERMINAL_BLOCKED`, no corpus/shard identity, and zero authorized balanced no-replay capacity.
- DATA-312 independently measures only 1 Ukrainian family and 1 English family and returns `FAIL_DIVERSITY`; the preregistered minimum is 2 independent families per stratum, with additional family-share limits also violated.
- EVAL-303 repairs part of the selection-validation gap by publishing a nonempty UA/EN composite, but code selection remains empty and EVAL-303 explicitly does not claim DATA-300 is built/frozen/terminal.

Therefore the blocker is not lack of seed orchestration. It is lack of a nonzero terminal external-real corpus authority suitable for the reference run itself.

## Unblock condition

Do not start any of the five seed runs until one successor corpus identity simultaneously provides:

- terminal/frozen corpus and shard identities;
- positive family-constrained one-pass unique optimized-target capacity;
- passing exact quality and privacy authorities on the same materialization;
- terminal exact/near dedup and evaluation decontamination;
- cluster-safe split and full post-split unique-loss ledger;
- diversity/balance pass with at least two independent families in each UA, EN and code stratum;
- immutable nonempty selection-validation authority adequate for the requested reporting;
- two independent byte-identical clean builds.

Once those conditions hold, rerun this exact five-seed preregistration with no other experimental changes. Until then, BPB, source-family performance and variance remain `NOT_AVAILABLE_NOT_RUN` rather than zero.
