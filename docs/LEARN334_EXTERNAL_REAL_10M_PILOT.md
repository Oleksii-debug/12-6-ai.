# LEARN-334 external-real 10M scratch Base pilot

`SWARM_WORKER_ID: LEARN-334-EXTERNAL-REAL-10M-PILOT`

## Decision

`BLOCKED_PRE_OPTIMIZER_NO_TERMINAL_FROZEN_CORPUS`

No optimizer update was executed. This is a fail-closed launch result, not a learned-model result and not a stage-promotion signal.

## Launch-time authorities

The worker branch was cut from DATA-301 head `8820ba1b255f6bb95c7db0531fd846078a1aae01` before any training action.

The current S3 scratch-Base architecture authority is `configs/stages/s3_10m.json`: target 10,000,000 parameters, expected 10,059,840, random initialization, vocabulary 8192, context 1024, d_model 320, 6 layers, 8 MHA / 8 KV heads, d_ff 864, SwiGLU, pre-RMSNorm and RoPE. Expected model identity is `3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a` and expected init identity is `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`.

The last preregistered external-real no-replay recipe available before launch is the LEARN-319 successor protocol. Its optimization contract is AdamW 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0, constant LR, no warmup, global-norm clip 1.0, FP32 and seed 1337. It requires deterministic 0/25/50/75/100% boundaries, a mandatory fresh-process resume at 50%, selection by aggregate immutable selection-validation BPB only, UA/EN/code/source-family BPB, pre/post-clip gradient telemetry, exact optimized-target accounting, retained-checkpoint fresh reload, raw Base generation and a first-party logits fingerprint. Its no-replay optimized-target bound is `min(131,938, terminal_one_pass_unique_train_optimized_targets)`.

No attempt was made to silently mutate that recipe or wait for sibling optimizer experiments.

## Corpus gate

Current DATA-301 terminal evidence states:

- `corpus_frozen = false`;
- `corpus_terminal = false`;
- `release_ready = false`;
- `corpus_identity = null`;
- `shard_identity = null`;
- `authorized_balanced_no_replay_capacity = 0`.

The exact hard blockers are G05 exact Wave-3 quality rerun, G06 exact Wave-3 privacy rerun, G09 source-family diversity (`uk=1,en=1,code=2` versus at least two independent families per stratum), G10 nonempty immutable selection-validation, G12 a full five-source unique-loss ledger, and G14 two independent clean byte-identical builds.

Because the requested run explicitly requires a frozen external-real Research Corpus V1 and a preregistered no-replay budget, starting optimizer step 1 with zero authorized balanced no-replay capacity would violate the mission. Replaying source positions, using validation material as training data, inventing a corpus identity or fabricating missing source families is prohibited.

## Required evidence status

- checkpoints: none; no learned checkpoint exists;
- fresh resume: not reached; no checkpoint exists to resume;
- aggregate BPB: none;
- UA BPB: none;
- EN BPB: none;
- code BPB: none;
- source-family BPB: none;
- gradient telemetry: none;
- generation: none;
- logits fingerprint: none;
- optimizer updates: 0;
- training executed: false.

## Unblock condition

Launch is permitted only after a successor terminal corpus authority provides a non-null immutable corpus/shard identity, at least two authorized training families in each of UA/EN/code, passing exact quality and privacy gates, nonempty immutable selection-validation, a full post-split unique-loss ledger, and two byte-identical clean builds. At that point the run must re-bind the then-current safe S3 architecture and the last terminal preregistered recipe available before that future launch; this blocked worker does not pre-authorize a later run.

`LOCAL_FREE` only. No foreign pretrained weights, SFT, RLHF or DPO.
