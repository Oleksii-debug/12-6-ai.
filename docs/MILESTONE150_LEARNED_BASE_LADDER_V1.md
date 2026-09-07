# MILESTONE-150 — LEARNED BASE LADDER V1

## Purpose

MILESTONE-150 is a convergence execution over accepted 12-6 components. It does not introduce a new model framework, trainer, tokenizer, checkpoint format, evaluator, or inference runtime.

The milestone creates a directly comparable learned scratch-Base ladder at approximately 100K, 500K, and 1M parameters under one explicit corpus/tokenizer/packing/evaluation truth model. The ~10M rung is intentionally excluded unless genuine learned evidence exists under the same truth model.

## Canonical live repository identity

GitHub repository: `Oleksii-debug/12-6-ai.`

The trailing period is part of the repository name.

## Selected convergence truth

- corpus: DATA-25 corpus V0.1
- required corpus identity: `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`
- tokenizer: `s0-byte-v1`
- tokenizer config SHA-256: `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`
- tokenizer vocabulary SHA-256: `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`
- vocabulary: 256 raw UTF-8 byte IDs
- special tokens: none
- normalization: none
- packing: incumbent document-isolated packing, sequence length 128
- evaluation split: DATA-25 validation
- evaluation strata: `uk`, `en`, `code`
- common metric: autoregressive held-out cross-entropy plus bits per raw UTF-8 byte

DATA-25 is currently project-authored and has zero external training-eligible sources. MILESTONE-150 therefore does not claim externally representative corpus coverage.

## Retained geometry family

The ladder uses the existing controlled byte-native family rather than reopening architecture search.

| rung | parameters | d_model | layers | heads / kv heads | head_dim | d_ff | ModelSpec SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| 100K | 95,568 | 48 | 3 | 4 / 4 | 12 | 128 | `4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470` |
| 500K | 467,808 | 96 | 4 | 6 / 6 | 16 | 256 | `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a` |
| 1M | 1,037,696 | 128 | 5 | 8 / 8 | 16 | 352 | `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07` |

All three use the incumbent pre-norm RMSNorm + RoPE + SwiGLU decoder semantics with tied word embeddings and no attention/MLP bias.

InitSpec for every rung is the incumbent scratch initialization:

- family: normal
- std: 0.02
- residual branch scaling: `sqrt_2_layers`
- InitSpec SHA-256: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`

## Training contract

Every comparable rung uses the same bounded LOCAL_FREE contract:

- random initialization only
- Trainer incumbent
- AdamW
- learning rate: 3e-4
- betas: 0.9 / 0.95
- epsilon: 1e-8
- weight decay: 0
- warmup: 0
- scheduler: constant
- gradient clipping: 1.0
- precision: FP32
- batch size: 8
- sequence length: 128
- maximum optimizer steps: 1000
- deterministic seed: 1337
- checkpoint steps: 0, 250, 500, 750, 1000
- mandatory fresh-process resume: checkpoint 500 -> first resumed optimizer step 501 -> step 1000

The runner records exact optimized tokens rather than inferring them from nominal batch geometry.

## Fresh retained-checkpoint verification

For each rung, the learned best checkpoint and final checkpoint are retained and freshly verified. If the best checkpoint is also the final checkpoint, the single physical checkpoint satisfies both roles.

Verification requires:

1. D05 integrity verification and load.
2. Exact source SHA, ModelSpec, parameter-count, tokenizer, corpus, run-manifest and step identities.
3. First-party D07 logits on fixed UA/EN/code prompts, computed twice with exact float32 SHA-256 snapshots.
4. Fresh held-out aggregate and `uk`/`en`/`code` evaluation matching the recorded result.
5. Evaluation non-mutation.
6. Fresh first-party greedy Base generation matching the retained generation snapshot.
7. Reproducibility/run-manifest self-hash and checkpoint binding validation.

A rung cannot appear in the final machine-readable ladder with `fresh_verification=PASS` unless all these gates pass.

## 10M truth boundary

Existing ~10M work is useful engineering evidence for ModelSpec execution, training mechanics, distributed/runtime paths, and short update probes. It is not equivalent to a learned checkpoint evaluated under this DATA-25/tokenizer/evaluation identity.

Therefore MILESTONE-150 records:

`INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE`

for 10M and omits unsupported held-out BPB, per-stratum quality, retained learned checkpoint, and raw learned-generation fields.

## Machine-readable output

The final artifact is `milestone150-evidence/ladder-report.json`. It contains:

- exact source and truth identities;
- exact ModelSpec and InitSpec for every retained rung;
- optimizer/run config and optimized tokens;
- best and final checkpoint identities;
- aggregate and `uk`/`en`/`code` BPB;
- compute wall time, observed throughput, and process RSS evidence;
- fresh-process resume evidence;
- random-init, best-checkpoint, and final raw Base generation snapshots;
- fresh retained-checkpoint verification evidence;
- quality ranking;
- efficiency ranking;
- pairwise scaling-improvement measurements;
- explicit 10M incompleteness.

The report is self-hashed and validated on the exact workflow head.

## Explicitly absent claims

MILESTONE-150 makes no claim of intelligence, production readiness, alignment, or instruction following. It uses no foreign pretrained weights, SFT, RLHF, DPO, or paid compute.

The milestone is evidence generation, not stage-promotion authority.
