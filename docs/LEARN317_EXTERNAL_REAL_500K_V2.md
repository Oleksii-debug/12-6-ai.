# LEARN-317 — External-real ~500K Base V2

Worker: `LEARN-317-EXTERNAL-REAL-500K-V2`

Execution profile: `LOCAL_FREE`

## Terminal verdict

`BLOCKED_NO_FROZEN_TERMINAL_EXTERNAL_REAL_CORPUS`

No model training was started. Optimizer updates: **0**. Optimized loss positions consumed: **0**. Replay: **0**.

This is fail-closed, not a failed optimization run. The exact upstream DATA-301 authority at head `d07b27ae399d104d2a01cb24b5a85fc4d9bcf800` is terminal-blocked: `corpus_frozen=false`, `corpus_terminal=false`, `corpus_identity=null`, `shard_identity=null`, and `authorized_balanced_no_replay_capacity=0`. The blocker is upstream corpus authorization, not trainer mechanics.

Starting even one optimizer step from prebuild candidate bytes, DATA-25, project-authored data, synthetic data, or another nonterminal substitute would violate the requested exact-frozen-candidate control and the no-replay rule.

## Frozen scratch control

The V2 control retains the comparable LEARN-234 scratch geometry:

- Parameters: 467,808
- Decoder width: 96
- Layers: 4
- MHA heads / KV heads: 6 / 6
- Head dimension: 16
- SwiGLU FFN dimension: 256
- Sequence length: 128
- Batch size: 8
- ModelSpec SHA256: `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a`
- InitSpec SHA256: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`
- Scratch initialization only; no foreign weights.

Canonical tokenizer control:

- ID: `s0-byte-v1`
- Vocabulary: 256 ordinary byte IDs
- Special tokens: 0
- Config SHA256: `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`
- Vocab SHA256: `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`

Optimizer control:

- AdamW, LR `3e-4`
- betas `(0.9, 0.95)`
- eps `1e-8`
- weight decay `0`
- constant schedule, no warmup
- global gradient clip norm `1.0`
- FP32
- seed `1337`
- document-isolated packing

Requested optimized-target cap is 948,504, but the realized cap is strictly:

`min(948504, exact_terminal_one_pass_unique_optimized_train_targets)`

No sampling with replacement, document replication, alias inflation, padding-as-data, or replay is permitted.

## DATA-301 gate result

The exact DATA-301 terminal authority reports:

- status: `TERMINAL_BLOCKED`
- normalized unique candidate bytes prebuild: 183,061
- source families: UA 1, EN 1, code 2
- authorized balanced no-replay capacity: 0
- full five-source terminal loss ledger: unavailable
- cluster-safe split: not reached
- deterministic sharding: not reached
- two clean builds: not permitted because prebuild hard gates fail

Hard blockers include missing passing exact Wave-3 quality/privacy reruns, insufficient source-family diversity, zero-record terminal selection-validation authorities, missing full five-source unique-loss accounting, and absence of two byte-identical clean builds.

The DATA-300 v2 contract itself explicitly says `corpus_state=NOT_BUILT_NOT_FROZEN_NOT_TERMINAL` and forbids repairing the diversity deficit through replay.

## Selection and final-test isolation

Selection-validation may choose the best checkpoint only after a nonempty immutable selection authority is terminal and proven disjoint from training/final-test bytes.

Preregistered checkpoint fractions remain 0%, 25%, 50%, 75%, and 100% of the realized one-pass budget. Best checkpoint rule: minimum frozen selection-validation aggregate BPB, with deterministic tie-break to the earliest checkpoint.

The final-test authority was **not opened or evaluated by LEARN-317**. It remains prohibited until the selection checkpoint is frozen.

## Required evidence status

| Requirement | Status |
|---|---|
| Exact frozen external-real corpus | BLOCKED upstream |
| Canonical byte tokenizer | FROZEN control |
| No replay beyond unique capacity | PASS by zero-execution / gate enforcement |
| D05 | NOT RUN — blocked before training |
| Fresh-process resume | NOT RUN — no checkpoint exists |
| Aggregate BPB | NOT RUN |
| UA BPB | NOT RUN |
| EN BPB | NOT RUN |
| Code BPB | NOT RUN |
| Source-family BPB | NOT RUN |
| Gradient telemetry | NOT RUN — zero optimizer updates |
| Clip telemetry | NOT RUN — zero optimizer updates |
| Generation | NOT RUN — no checkpoint exists |
| Selection-validation use | NOT RUN |
| Final-test use | UNOPENED |
| LOCAL_FREE | PASS |

D05 checkpoint/resume mechanics exist in the repository, including the integrated resume/repro line, but this worker does not claim a D05 pass for a run that never had an authorized checkpoint. A fresh-process resume test is mandatory after the first authorized checkpoint is produced.

## Exact unblock conditions

LEARN-317 may train only after all of the following refer to one exact successor corpus identity:

1. A successor DATA-300 contract admits at least two independently authorized source families in each UA, EN, and code stratum without replay.
2. DATA-301 or its successor terminal build publishes non-null corpus/shard identities and nonzero exact balanced one-pass optimized-loss capacity.
3. Exact Wave-3 quality and privacy authorities pass on that same inventory.
4. Exact/near dedup, evaluation decontamination, cluster-safe split, and full post-split unique-loss accounting are terminal for that identity.
5. Required immutable nonempty selection-validation authority is terminal and disjoint from train/final-test bytes.
6. Two independent clean builds are byte-identical.

Only then: run D05 locally; train from scratch to at most the unique one-pass cap; retain scheduled checkpoints; collect gradient/clip telemetry; score aggregate/UA/EN/code/source-family BPB on selection-validation; choose and freeze the best checkpoint; verify fresh-process resume; generate samples; and only afterward open final-test for one final report.

Machine-readable authority evidence: `evidence/learn317/authority-gate.json`.
