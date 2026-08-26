# LEARN-319 external-real ~3.2M short scale probe

`SWARM_WORKER_ID: LEARN-319-EXTERNAL-REAL-3M-SHORT`

## Decision

`BLOCKED_NO_TERMINAL_FROZEN_CORPUS`

No optimizer update was executed. This is a fail-closed scientific result, not a learned-model result and not a stage-promotion signal.

## Authority reconstruction

The exact DATA-300 v2 contract at head `8ea7f830e50a23754d189dd4134f4afad76a7ee9` binds contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5` and explicitly states corpus state `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`.

At this worker's execution cutoff, the published branch `data301/corpus-v03-terminal-build-20260826` compares byte-for-byte/commit-for-commit identical to DATA-300: ahead 0, behind 0. No terminal DATA-301 corpus release is therefore available to consume.

The DATA-300 candidate is five source objects / four independent families / 183,061 source bytes (UA 88,565; EN 84,793; code 9,703). The retained DATA-294 ledger authorizes only 173,355 document-isolated text causal targets and no full five-source optimized-target capacity. Terminal immutable selection-validation is empty.

Hard blockers retained from DATA-300 are G05 quality rerun, G06 privacy rerun, G09 family diversity, G10 nonempty selection-validation, G12 full five-source unique-loss accounting and G14 two clean byte-identical builds. Inventing a frozen corpus identity, reusing validation bytes as training, or replaying source positions would violate the mission.

## Preregistered successor execution

If and only if a terminal corpus release clears every hard gate, the short scale probe is frozen as follows before optimizer step 1:

- model: 3,213,120-parameter decoder-only Base, d_model 192, 7 layers, 12 MHA heads / 12 KV heads, d_ff 528, context 256, ModelSpec `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc`;
- tokenizer: canonical `s0-byte-v1`, vocabulary 256, no special tokens;
- optimizer: AdamW 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0, constant LR, no warmup, global-norm clip 1.0, FP32, seed 1337;
- optimized-target budget: `min(131,938, terminal_one_pass_unique_train_optimized_targets)`; no replay, no replacement sampling, no padding-as-data;
- selection trajectory: deterministic boundaries at 0/25/50/75/100% of realized one-pass budget, with a mandatory fresh-process resume at the 50% boundary;
- best checkpoint: minimum aggregate immutable selection-validation BPB only; chronological final retained separately; final-test never influences selection;
- evaluation: aggregate, UA, EN, code and source-family BPB with model-state non-mutation proof;
- optimization telemetry: pre/post clip gradient norms, clip activation, update/weight L2 ratio and exact optimized-target counters;
- inference evidence: raw Base generation plus first-party logits fingerprint from fresh retained-checkpoint reload.

The realized budget may be smaller than 131,938 if the terminal post-split unique-train ledger is smaller. It may never be enlarged through recycling.

## Truth boundary

Training executed: false. Optimizer updates: 0. Learned checkpoint: none. BPB trajectory: none. Gradient/clipping/update-ratio trajectory: none. Generation/logits: none. No automatic stage promotion is authorized.

LOCAL_FREE only. No foreign pretrained weights, SFT, RLHF or DPO.
