# C01 EUR2000 training program

Status: `PREPARED_NOT_AUTHORIZED_BLOCKED_ON_STAGE_FREEZE_SCALE_DATA_TOKENIZER_AND_GPU_PILOT`.

This package does not launch or authorize paid compute. It converts the current project state into a bounded decision program and adds a fail-closed launch-control prerequisite without modifying the incumbent C01 files from PR #70.

## Current evidence boundary

Observed, not extrapolated:

- S0 is a 10,140-parameter random-init Base with `s0-byte-v1`, vocabulary 256 and context 128.
- W5 PR #142 measured one exact-head LOCAL_FREE GitHub-hosted CPU training observation at 62,238.35 optimized tokens/sec for the 40-step S0 run. That number is CPU/S0 evidence only.
- Current engineering ModelSpecs are S1 107,856 parameters / vocab 512 / context 256, S2 1,066,112 / vocab 2,048 / context 512, and S3 10,059,840 / vocab 8,192 / context 1,024.
- D01 S4 is 100,384,512 parameters / vocab 32,768 / context 2,048, but remains `engineering_candidate_not_frozen` and requires preceding-stage PASS.
- S1 numerical and checkpoint mechanics have bounded CPU evidence, but PR #130 correctly rejects the S0 256-token vocabulary as canonical input for S1's 512-row vocabulary.
- PR #73 now has real locked tokenizer experiments on `tokenizers==0.23.1`. Controlled ByteLevel BPE requested vocab 512 but realized 472 tokens and repeated exactly; Unigram realized 497 but did not repeat exactly. The retained decision is correctly `NO_FREEZE_REPEATABILITY_BLOCKED`, and the ten-record controlled corpus is not representative scale-corpus evidence.
- The product lineage inspected here still contains only controlled S0 corpus bytes; no representative S1+ train/validation corpus identity is frozen.
- D12 has topology/reshard contracts and bounded CPU/Gloo evidence, but no GPU/NCCL/multi-node canonical-model training. This is not a blocker for the recommended <=100M single-GPU program.

Therefore the current bottleneck is not HBM capacity. It is stage/model selection, representative scale data/tokenizer identity, and same-geometry GPU evidence.

## Budget architecture

The hard ceiling is EUR 2,000, not a spend target:

- EUR 50 systems smoke and GPU calibration;
- EUR 150 pilot/recipe selection;
- EUR 1,200 main training hard cap;
- EUR 100 storage/recovery;
- EUR 500 restart/price-variance reserve that cannot be consumed automatically.

The pricing snapshot is an assumption, not a quote. Runpod's official guide lists Secure Cloud rates verified 10 August 2026 of USD 1.39/GPU-hour for A100 PCIe 80 GB and USD 2.99/GPU-hour for H100 SXM 80 GB. Using the ECB 24 August 2026 reference of USD 1.1664 per EUR gives approximately EUR 1.1917/hour and EUR 2.5634/hour respectively. VAT, storage, egress, capacity premiums and future price movement are excluded, so pricing must be refreshed before authorization.

## Strategy comparison

### A — S3 smaller model, more tokens

- parameters: 10,059,840;
- tokens: 1.0B;
- tokens/parameter: 99.4;
- sequence: 1,024;
- target global batch: 131,072 tokens / 128 full sequences;
- BF16, AdamW, one A100 80 GB;
- planning FLOPs: `6*N*T = 6.035904e16`;
- raw BF16 weights: 20,119,680 bytes;
- conservative full-Adam planning state: 160,957,440 bytes.

This is the lowest systems-risk path and gives the most token exposure per parameter. It is useful if the eventual corpus is not large enough or S4 governance is not ready.

### B — S4 larger model, fewer tokens

- parameters: 100,384,512;
- tokens: 2.0B;
- tokens/parameter: 19.9;
- sequence: 2,048;
- target global batch: 262,144 tokens / 128 full sequences;
- BF16, AdamW, one A100 80 GB;
- planning FLOPs: `1.204614144e18`;
- raw BF16 weights: 200,769,024 bytes;
- conservative full-Adam planning state: 1,606,152,192 bytes.

This buys an earlier parameter-scale signal but has more architecture/tokenizer risk and a low token/parameter ratio. It cannot be a legitimate main run while the S4 ModelSpec remains non-frozen.

### C — pilots, then balanced S4 main — recommended

Pilot sequence:

1. S1: 107,856 parameters, 10M tokens, context 256.
2. S2: 1,066,112 parameters, 50M tokens, context 512.
3. S3: 10,059,840 parameters, 200M tokens, context 1,024.
4. Main only after healthy evidence, governing preceding-stage PASS, and explicit S4 freeze: S4 100,384,512 parameters, 5B tokens, context 2,048.

Each stage must use a tokenizer artifact whose actual vocabulary cardinality equals that selected ModelSpec; requested vocabulary size is not accepted as a substitute for realized vocabulary size. The pilots should share one frozen corpus lineage, split/provenance policy and comparable recipe/evaluation semantics so that cross-scale evidence remains interpretable.

The three pilots together are only about `1.24e16` planning FLOPs, less than one half of one percent of the S4 main's `3.01153536e18` planning FLOPs. They therefore buy disproportionate information about data, tokenizer, loss dynamics, batch sizing and throughput before the expensive run.

The S4 main has about 49.8 tokens/parameter. It checkpoints every 100M optimized tokens, evaluates held-out every 250M tokens and retains the last two checkpoints plus 10/25/50/75/100% milestones. At the deliberately conservative 16 bytes/parameter state-planning bound, seven retained payloads are about 11.24 GB before the required off-instance replica and filesystem overhead.

## Wall-time and cost envelope

No GPU throughput has been measured for 12-6 AI, so the launch configuration does not encode a fabricated tokens/sec value. Main wall time is always:

`training_tokens / observed_same_geometry_pilot_tokens_per_second`.

For engineering intuition only, a dense `6*N*T` FLOP approximation and hypothetical fractions of vendor BF16 Tensor-Core peak can bound the S4 5B-token run. These are assumptions, not measurements and not launch evidence.

Using A100 dense BF16 Tensor-Core peak 312 TFLOP/s:

- at 1% effective utilization: about 268 GPU-hours and about EUR 320 at the planning rate;
- at 2%: about 134 hours and EUR 160;
- at 5%: about 53.6 hours and EUR 64.

Using H100 SXM dense BF16 peak approximately 989.5 TFLOP/s (half of NVIDIA's sparsity-marked 1,979 TFLOP/s figure):

- at 1% effective utilization: about 84.6 GPU-hours and about EUR 217 at the planning rate;
- at 2%: about 42.3 hours and EUR 108;
- at 5%: about 16.9 hours and EUR 43.

A 100M model can realize very low accelerator utilization because launch/kernel/input overheads matter. This is exactly why the EUR 10 same-geometry smoke and measured pilot are mandatory. The program chooses GPU based on measured EUR per completed token, not theoretical peak.

## Launch sequence

1. Complete governing preceding-stage evidence and explicitly freeze the selected main ModelSpec. S4 is not launchable merely because a candidate file exists.
2. Freeze scale corpus identity, source/license/provenance, train/validation split, contamination controls, token counts and restart-safe sharding.
3. Freeze a representative-corpus tokenizer whose actual token IDs and actual vocabulary cardinality match the selected ModelSpec; bind tokenizer and packing hashes.
4. Bind the exact GPU dependency/runtime lock and exact source SHA. Floating installation is prohibited.
5. Integrate the current model-agnostic Trainer/data/tokenizer/checkpoint seams into a general scale runner rather than cloning a second Trainer implementation.
6. Run the same-geometry systems smoke with at most 2M optimized tokens, 0.5 GPU-hour and EUR 10 hard cap. It must perform forward/backward/update, checkpoint/reload/resume and held-out evaluation.
7. Record tokens/sec, step time, data wait, loss, LR, gradient norm, peak HBM, checkpoint/eval time and exact GPU/runtime identity.
8. Run S1/S2/S3 pilots under the EUR 150 pilot cap. Select learning rate from the predeclared `3e-4`, `6e-4`, `1e-3` set using held-out evidence rather than train loss alone.
9. Project S4 wall time from measured same-geometry GPU tokens/sec and refresh provider price.
10. Owner explicitly authorizes one strategy, one provider/SKU, one git SHA, exact model/data/tokenizer/packing/runtime identities and one EUR cap.
11. Launch main only if the projected compute cost plus protected recovery reserve remains inside the EUR 2,000 ceiling.

## Stop and recovery policy

Immediate abort: NaN/Inf loss or gradient norm, data/tokenizer identity drift, validation tokens reaching the optimizer, checkpoint checksum/identity failure, OOM after pilot-resolved batch sizing, or runtime/provider identity differing from the authorization record.

Investigate then abort if persistent: throughput below 50% of the same-geometry pilot for ten minutes, data wait above 20% of step time for ten minutes, held-out loss failing the predeclared improvement gate, or gradient norm above 10x its running median for 20 consecutive steps.

Recovery point objective is 100M optimized tokens. Blind retry is prohibited. Resume must rebind git/model/data/tokenizer/packing/optimizer identities before consuming further paid compute.

## What remains genuinely unready

P00 is real: S4 remains an engineering candidate, not a frozen stage ModelSpec, and its own governance requires the preceding stage PASS.

P01/P02 are the largest scientific blockers. The tokenizer lane has real executable evidence now, but no representative scale corpus/tokenizer is frozen. The controlled BPE result is 472 actual tokens, not S1's current 512 rows; it cannot simply be declared canonical. The larger S2/S3/S4 requested vocabularies also need representative-corpus sweeps and exact ModelSpec re-binding.

P04 is the largest compute-estimation blocker: there is no same-geometry GPU training measurement. S0 CPU throughput is explicitly barred from the GPU cost equation.

The current first-party executable evidence CLI is S0-specific. A general scale runner still has to compose the existing model-agnostic Trainer, data/packing, selected tokenizer and checkpoint seams before the same-geometry smoke. That work belongs at the existing training/data integration seam; it should not create a second Trainer. The existing Trainer already exposes AdamW, BF16/FP16/FP32 configuration, constant/linear-warmup/cosine scheduling, gradient accumulation and clipping.

## Owner decision

The recommended authorization, when P00-P05 are green, is:

`AUTHORIZE C_PILOTS_THEN_S4_BALANCED_MAIN WITH HARD TOTAL CAP EUR 2000, MAIN CAP EUR 1200, SINGLE GPU, NO AUTOMATIC OVERRUN, NO DISTRIBUTED SCALE-UP WITHOUT A NEW AUTHORIZATION.`

Until then the correct decision is `DO NOT LAUNCH PAID COMPUTE`.
