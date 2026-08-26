# COMPUTE-32 EUR10k training program

Status: `PLANNING_ONLY_NOT_COMPUTE_AUTHORIZATION`.

No paid compute is launched or authorized by this package. Materially paid execution remains forbidden unless an owner separately provides `COMPUTE_AUTHORIZED` after all technical gates pass.

## Live baseline

The branch is stacked on the current Product convergence PR #132 head `d07393f6f62b99c8106c0b72e6dd6ee53430e4dd`, not bootstrap-only `main`.

Current established facts:

- S0 has real training/repeatability/strict-evaluation evidence; it remains EXPERIMENTAL.
- S1/S2/S3 exact ModelSpecs are 107,856 / 1,066,112 / 10,059,840 parameters.
- S4/S5/S6/S7 exact engineering candidates are 100,384,512 / 400,598,016 / 999,106,560 / 2,998,029,312 parameters, but are not frozen.
- PR #73 now has successful CI and a successful real tokenizer evidence-capture workflow. It proves maintained-library BPE/Unigram mechanics on the tiny controlled corpus, not a representative tokenizer winner. Its retained evidence records Unigram exact-artifact repeatability as a blocker on the pinned backend.
- A representative large corpus, tokenizer winner and S6 architecture are not frozen.
- Distributed topology/checkpoint contracts exist, but canonical GPU/NCCL/FSDP multi-GPU training is not yet demonstrated.
- S0 CPU throughput is not admissible evidence for GPU duration or cost.

## Strategic choice

Use a staged campaign rather than spending EUR10k on the largest model that can theoretically fit the budget.

The primary main-run candidate is S6:

- 999,106,560 parameters;
- 32,768 vocabulary;
- 4,096 context;
- d_model 2,048;
- 18 layers;
- 32 query heads / 8 KV heads;
- head dimension 64;
- SwiGLU d_ff 6,720;
- pre-RMSNorm, RoPE, tied embeddings;
- 20,971,520,000 training tokens;
- 1,048,576 global tokens/update;
- 20,000 optimizer updates;
- BF16 AdamW, LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0.1;
- 400-step warmup and cosine decay;
- one H100 SXM 80GB for the primary plan.

Single-GPU is deliberate. It makes the first serious run independent of a distributed runtime that has not yet earned canonical GPU evidence. S7/3B remains a successor only after multi-GPU qualification.

## Scaling-law policy

The classic Chinchilla result is the baseline: fixed-compute model size and data should scale together. It is not treated as a universal 20-token-per-parameter law. DeepSeek reported materially different model/data exponents across datasets. Recent compute-data scaling work also shows diminishing value when fresh data is replaced by repeated/derived data. Recent tokenization scaling work shows that token count itself is tokenizer-dependent, so tokenizer experiments must compare identical raw bytes and bits-per-byte rather than perplexity alone.

Therefore the program first measures a local scaling curve on the actual corpus/tokenizer, then freezes S6 only if the measured curve predicts value.

## Corpus proposal

Build at least 24B unique post-filter tokens before main authorization:

- 75 weight units FineWeb-Edu English;
- 25 weight units FineWeb2 `ukr_Cyrl`.

Both source families are published under ODC-By 1.0, but exact revisions, file hashes, attribution, retrieval state and rights review must be recorded in the project manifest before freeze.

Required build controls: stable record IDs, deterministic train/validation/test split, exact and near-duplicate removal before split, contamination registry, restart-safe mixture cursor, byte counts, post-tokenization token counts and immutable source identities.

## Tokenizer decision

Target vocab size is exactly 32,768 because it is part of the S4-S7 parameter geometry.

Use PR #73's real maintained-library harness. ByteLevel BPE is eligible for representative-corpus experimentation. Unigram is not eligible for freeze until its repeated-build artifact-identity blocker is resolved or it is explicitly excluded.

For any eligible candidate, train on the same 8 GiB UTF-8 corpus sample and require:

- 100% UTF-8 round trip;
- zero OOV;
- exact 32,768 vocabulary;
- exact repeated artifact identity;
- held-out BPB overall, English and Ukrainian;
- bytes/token and end-to-end training throughput.

Choose the lowest held-out BPB. If candidates are within 0.5%, choose lower measured training cost and simpler operational behavior.

## Architecture pilots

S4 pilot: exact 100,384,512-param candidate, context 2,048, 2,097,152,000 tokens, BF16, one H100.

S5 pilot: exact 400,598,016-param candidate, context 4,096, 8,388,608,000 tokens, BF16, one H100.

Evaluate at 12.5%, 25%, 50%, 75% and 100%. Fit checkpoint curves using held-out BPB. Do not reuse S0 CPU throughput in this fit.

Advance S5 -> S6 only if S5 is stable and the local curve predicts at least 0.5% BPB improvement for S6 while measured-throughput projection remains <= EUR8k after reserve.

## Mandatory cheap S6 qualification

Before main authorization, run exactly 134,217,728 tokens on the exact S6 shape, exact frozen tokenizer/corpus, BF16, 4K context and one H100:

- global batch 1,048,576 tokens;
- 128 optimizer steps;
- microbatch one 4,096-token sequence;
- gradient accumulation 256;
- force checkpoint after step 64;
- terminate the training process;
- restore in a fresh process and complete.

Qualification gates:

- measured global throughput >= 8,000 tok/s;
- peak HBM <= 90%;
- data wait <= 10%;
- checkpoint overhead <= 5%;
- loss decreases;
- zero non-finite steps;
- checkpoint round-trip PASS;
- resumed trajectory meets the declared continuity tolerance;
- data cursor resume PASS.

This experiment is cheap, but still paid; it also requires separate `COMPUTE_AUTHORIZED` before execution.

## Main checkpoint/recovery plan

Checkpoint every 250 optimizer steps. Retain three rolling checkpoints and milestones at 2k, 5k, 10k, 15k and 20k steps. Require source/model/tokenizer/corpus/data-cursor identity, artifact self-hash, optimizer/scheduler state, RNG state and fresh-Trainer restore.

At 16 bytes/parameter for full Adam-style training state, S6 is approximately 15.99 GB/checkpoint before filesystem/metadata overhead. Eight retained full-state payloads are approximately 127.9 GB raw.

On failure, resume only from the latest fully verified committed checkpoint. Ambiguous optimizer transitions are never replayed.

## Cost model

Cost is recomputed only from the S6 GPU qualification. Current prices are assumptions, not authorization inputs.

As of 2026-08-25, the planning quote uses RunPod Secure Cloud H100 SXM at USD3.29/GPU-hour and observed USD/EUR 0.857284; the gate rounds upward to EUR2.83/GPU-hour. Tax, storage, egress and availability are excluded and require a re-quote before authorization.

For 20,971,520,000 tokens on one GPU:

- 8k tok/s floor -> about 728.18 hours and EUR2,061 GPU compute;
- 12k tok/s -> about 485.45 hours and EUR1,374;
- an external optimized OLMo-core 1B/4K/BF16 H100 reference reports about 44k tok/s/GPU, corresponding to about 132.4 hours and EUR375, but that is not 12-6 evidence and cannot authorize spend.

EUR10k is a ceiling, not a target. If measured S6 cost is EUR1-3k, do not waste the remainder merely to consume budget.

## Stop criteria

Stop the main run on non-finite loss/gradients; repeated OOM after one predeclared microbatch fallback; validation BPB worsening at two consecutive post-warmup evaluations; data wait >10% for three windows; checkpoint overhead >5% for three intervals; projected compute >EUR8k; or any source/model/tokenizer/corpus/data-cursor identity drift.

## EUR2k -> EUR10k evidence

Scaling from an EUR2k program to this envelope requires: frozen representative tokenizer/corpus identities; stable S4/S5 runs; a local BPB scaling curve predicting material S6 gain; real S6-shaped GPU throughput/memory evidence; durable checkpoint/recovery; and >=20% projected budget reserve.

A successful S0 CPU run, FLOP estimate or external benchmark is not sufficient.

## EUR10k -> higher evidence

The next candidate is S7 at 2,998,029,312 parameters and 8K context. It requires at least two S6 seeds with directionally consistent gains, evaluation gains matching pre-registered predictions, enough unique effective data, real canonical multi-GPU NCCL training, same-topology recovery plus declared reshard policy, >=70% measured scaling efficiency to the proposed GPU count, and a separately authorized budget.

Without that evidence, improve data/tokenization/training efficiency rather than buy a larger run.

## Implemented prerequisite

`src/twelve_six/training/scale_launch_gate.py` plus `tools/compute32_launch_gate.py` implement the missing fail-closed decision boundary. The gate rejects CPU/extrapolated throughput, unfrozen identities, excessive HBM/data/checkpoint overhead, missing checkpoint/resume/data-cursor evidence, over-budget projections and multi-GPU plans without real distributed evidence.

The owner authorization token is never stored in the plan. Even a technically qualified record remains `launch_allowed=false` until the external literal `COMPUTE_AUTHORIZED` is supplied.
