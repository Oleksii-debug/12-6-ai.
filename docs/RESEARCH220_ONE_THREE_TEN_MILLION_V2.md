# RESEARCH-220 — 1M → 3.2M → 10M fixed-recipe empirical comparison V2

RESEARCH-220 executes the terminal-green RESEARCH-212 contract without changing the frozen RESEARCH-192 scientific producer. The experiment remains a LOCAL_FREE CPU diagnostic over three model sizes and is not a universal scaling-law fit or a stage-promotion authority.

## Frozen experiment

The exact preregistered arms remain 1M seeds 1337/1338, 3.2M seeds 1337/1338, and 10M seed 1337. No seed is added after results are observed. Every arm uses DATA-25 identity `422f545d...`, canonical `s0-byte-v1`, M150 evaluation identity `7189e6df...`, document-isolated seq128 packing, batch 8, common InitSpec, AdamW LR 3e-4 / betas 0.9/0.95 / eps 1e-8 / weight decay 0, constant schedule, no warmup, accumulation 1, clip 1.0 and FP32 deterministic execution.

The common actual optimized-token boundaries are exactly 17,125 / 66,417 / 131,938. The 1M, 3.2M and 10M arms therefore receive identical optimized-token exposure at every comparison boundary.

The model family remains exact MHA with context-capable ModelSpec length 256 and 16-wide heads: 1,037,696 parameters, 3,213,120 parameters and 10,000,640 parameters. The experiment does not import the separate S3 GQA/runtime geometry, context-transfer changes, or nonzero weight decay.

## Execution and evidence

Each arm is prepared, trained through the midpoint, resumed in a distinct process, trained to the final boundary, then verified in another fresh process. D05 checkpoint integrity, step/token counters, source/model/tokenizer/corpus/run identities and held-out evaluation non-mutation must all pass.

The existing RESEARCH-192 comparison reports held-out/UA/EN/code BPB, training BPB, generalization gap, wall time, throughput, peak RSS, 6*N*T, BPB gain per added parameter and BPB gain per incremental 6*N*T compute. RESEARCH-220 adds gradient-norm and clip-rate summaries computed only from the already-recorded frozen train curve; this adds no optimizer updates and cannot change the trajectory.

The accepted M150 artifact is used as a reproducibility control. RESEARCH-220 requires the new 1M seed-1337 random-init state to match the exact M150 random-init state hash and rechecks the common model, InitSpec, tokenizer, corpus, evaluation, sequence, batch and non-duration AdamW recipe. M150's later 474,377/948,504-token checkpoints are not substituted at RESEARCH-220's shorter common boundaries.

## Claim boundary

This is descriptive evidence from three sizes and a preregistered five-arm seed matrix. It does not establish a universal scaling law, external-corpus representativeness, production readiness, intelligence, alignment or instruction following. No foreign pretrained weights, SFT, RLHF, DPO or paid compute are used.
