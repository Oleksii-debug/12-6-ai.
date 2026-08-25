# COMPUTE-32 EUR10k training program

Status: `PLANNING_ONLY_NOT_COMPUTE_AUTHORIZATION`.

This package does not launch or authorize paid compute. Materially paid execution remains forbidden unless an owner separately supplies the literal external authorization token `COMPUTE_AUTHORIZED` after all technical gates pass.

## Live baseline

The implementation base is Product convergence PR #132 at `86dbcc0b804da988a34367ff74c49ee00bc05818`, not bootstrap-only `main`. That convergence line has real S0 training, repeatability and strict evaluation evidence for the 10,140-parameter Base model. S1/S2/S3 engineering ModelSpecs exist at 107,856 / 1,066,112 / 10,059,840 parameters. S4-S7 have exact engineering candidate counts but are explicitly not frozen.

Important launch blockers are still real:

- the accepted S0 tokenizer is 256 raw bytes, while future stages require larger vocabularies;
- future BPE/Unigram experiment code exists, but its maintained tokenizer dependency is not yet in the canonical hash-locked environment;
- no frozen large corpus manifest exists with deterministic restart identity and contamination accounting;
- distributed contracts exist, but canonical GPU/NCCL/FSDP multi-GPU training has not been proven;
- current S0 CPU throughput is not evidence for H100 throughput;
- S6 is an exact engineering candidate, not a frozen architecture.

## Strategic choice

Do not spend EUR10k on one maximum-size run. Use a staged campaign and stop when evidence no longer supports scaling.

The default main target is the existing S6 engineering candidate:

- 999,106,560 parameters;
- vocab 32,768;
- context 4,096;
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
- 400-step warmup followed by cosine decay;
- one H100 SXM 80GB as the primary plan so success does not depend on unproven distributed execution.

The token count is approximately 21 tokens/parameter. This is a starting operating point, not a claim that a universal 20:1 rule is optimal for the project's data. The pilot campaign must fit the local scaling behavior before S6 is frozen.

## Why staged rather than largest

Hoffmann et al. established the classic compute-optimal result that model size and data should scale together. Later work shows why the project must not treat the ratio as universal: DeepSeek reports materially different model/data scaling exponents for different datasets, and 2026 work on compute-data scaling shows diminishing effectiveness when fresh data is replaced by repetition. Recent tokenization scaling work also shows that token count itself is not a stable cross-tokenizer unit; raw bytes and bits-per-byte are required for fair tokenizer pilots.

Therefore the campaign buys information first:

1. train/freeze the tokenizer on the intended corpus distribution;
2. run S4 and S5 checkpoint curves on the same data/tokenizer;
3. run an S6-shaped qualification that proves memory, throughput, checkpoint and resume behavior;
4. only then decide whether S6 is worth the main run;
5. do not promote to S7 merely because budget remains.

## Corpus proposal

Build at least 24B unique post-filter tokens before the main authorization decision. Proposed first mixture:

- 75 weight units FineWeb-Edu English;
- 25 weight units FineWeb2 `ukr_Cyrl`.

Both source families are currently published under ODC-By 1.0. Exact dataset revisions, source files, hashes and attribution must be pinned in the project manifest before the launch gate can pass.

Required controls: stable record IDs, deterministic split, exact/near duplicate removal before split, contamination registry, restart-safe mixture cursor, source revision hashes, byte counts and post-tokenization token counts.

## Tokenizer decision

Compare exactly two 32,768-vocabulary candidates on the same 8 GiB UTF-8 sample drawn from the frozen mixture:

- ByteLevel BPE;
- Unigram with byte fallback.

Hard gates: 100% UTF-8 round trip, zero OOV, exact 32,768 vocabulary. Select by held-out bits-per-byte overall and separately for English/Ukrainian. Perplexity may only be compared within one fixed tokenizer. If BPB differs by less than 0.5%, choose the tokenizer with lower measured end-to-end training cost and simpler operational behavior.

This directly resolves the current S1+ tokenizer gap rather than silently carrying the S0 byte tokenizer forward.

## Pilot campaign

### S4

Use exact candidate 100,384,512 parameters, context 2,048, 2,097,152,000 tokens, BF16, one H100. Evaluate at 12.5%, 25%, 50%, 75%, 100% of the run.

### S5

Use exact candidate 400,598,016 parameters, context 4,096, 8,388,608,000 tokens, BF16, one H100. Use the same tokenizer/corpus identities and evaluation fractions.

Fit checkpoint curves using BPB, not only token loss. S6 may proceed only when S5 is numerically stable and the measured curves predict at least 0.5% BPB improvement at S6 within the EUR8k post-reserve compute ceiling.

## Cheap S6 qualification before main authorization

Run exactly 134,217,728 tokens on the exact S6 shape, context, precision, tokenizer and corpus intended for main training:

- global batch: 1,048,576 tokens;
- 128 optimizer steps;
- microbatch: one 4,096-token sequence;
- gradient accumulation: 256;
- forced checkpoint after step 64;
- destroy the live process/trainer objects;
- restore from the verified checkpoint in a fresh process and finish.

All gates must pass:

- GPU-measured global throughput >= 8,000 tok/s;
- peak HBM <= 90%;
- data-wait fraction <= 10%;
- checkpoint overhead <= 5%;
- loss decreases;
- zero non-finite steps;
- checkpoint round-trip passes;
- resumed trajectory satisfies the declared continuity tolerance;
- data cursor resumes correctly.

Even this cheap paid qualification still requires a separate owner authorization. The program never interprets "cheap" as "authorized".

## Main run checkpoint/recovery contract

Checkpoint every 250 optimizer steps. Retain the latest three rolling checkpoints plus milestones at 2k, 5k, 10k, 15k and 20k steps. Require self-hashed artifacts, exact source/model/tokenizer/corpus identities, fresh-Trainer restore, optimizer/scheduler state, RNG state and data-cursor state.

Full Adam state is planned at about 15.99 GB for S6 before filesystem/metadata overhead. Eight retained full-state payloads are therefore about 127.9 GB raw. Storage and egress are outside the GPU quote and must be separately priced.

On failure, restart only from the latest verified committed checkpoint. Never replay an optimizer transition whose commit state is ambiguous.

## Cost and duration policy

Do not use S0 CPU throughput for any GPU cost claim. Main-run cost must be recomputed from the S6 GPU qualification result.

Current external anchors are only assumptions. RunPod listed H100 SXM Secure Cloud around USD3.29/GPU-hour on 2026-08-25. At the observed USD/EUR conversion used by the plan, the gate rounds upward to EUR2.83/GPU-hour. Requote before authorization.

For 20,971,520,000 tokens on one GPU:

- at the qualification floor 8k tok/s: about 728.18 hours, about EUR2,061 GPU compute at the rounded quote;
- at 12k tok/s: about 485.45 hours, about EUR1,374;
- an external optimized OLMo-core 1B/H100/4K BF16 report is about 44k tok/s/GPU, which would imply about 132.40 hours and about EUR375, but this is not 12-6 evidence and must never be used as the authorization projection.

The EUR10k amount is a ceiling, not a spending target. Unspent budget is preferable to low-information repetitions.

## Main stop criteria

Stop and investigate rather than continuing spend when any of the following occurs:

- non-finite loss/gradients;
- repeated OOM after one predeclared microbatch fallback;
- validation BPB worsens at two consecutive scheduled evaluations after warmup;
- data-wait >10% for three consecutive windows;
- checkpoint overhead >5% for three consecutive intervals;
- measured cost projection exceeds EUR8k after reserve;
- any source/model/tokenizer/corpus/data-cursor identity drift.

## Scale from EUR2k to EUR10k

An EUR2k-class program justifies the EUR10k envelope only if it produces all of the following: frozen tokenizer and corpus identities, stable S4/S5 training, a local loss/BPB scaling curve that predicts a material S6 gain, GPU-measured throughput/memory evidence, durable checkpoint/recovery, and a projected S6 run with at least 20% budget reserve.

A successful S0 CPU run or theoretical FLOP calculation is not sufficient.

## Scale above EUR10k

The next logical candidate is S7 at 2,998,029,312 parameters, 32K vocab and 8K context, but it is not the automatic successor. Scaling above EUR10k requires two S6 seeds with directionally consistent gains, capability/evaluation gains matching pre-registered predictions, enough unique effective data, real canonical multi-GPU NCCL evidence, same-topology recovery plus a declared reshard policy, and >=70% measured scaling efficiency to the proposed GPU count.

Without those facts, the program should keep improving data/tokenization/training efficiency rather than buying a larger model.

## Executable gate

`src/twelve_six/training/scale_launch_gate.py` and `tools/compute32_launch_gate.py` implement the launch decision. The gate deliberately refuses CPU/extrapolated throughput, unfrozen identities, missing recovery evidence, excessive HBM/data/checkpoint overhead, over-budget projections, and multi-GPU launches without real distributed evidence.

The owner authorization token is not stored in the plan. A technically qualified record remains `launch_allowed=false` until the literal external token `COMPUTE_AUTHORIZED` is supplied.
