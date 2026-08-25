# TRAIN-128 effective-batch transfer

## Authority

TRAIN-128 is a LOCAL_FREE effective-batch scaling experiment. It does not claim an exact theoretical critical batch size, does not authorize paid compute, and does not change a canonical Trainer default.

The branch is stacked on TRAIN-53 so the exact rerun path composes its existing DATA-21/22 bounded real-source intake, byte tokenizer, packing, Trainer/AdamW, TrainingObserver, held-out evaluator, and non-mutating gradient-noise diagnostic. TRAIN-128 adds only model-size orchestration and cross-size reporting.

The latest TRAIN-53 exact-head workflow at `67dd276068c30d434d89497a306f69e1c50faa1b` is not accepted as executed evidence: run `32862639045` failed during repository-wide Ruff checks before the batch experiment step ran. TRAIN-128 therefore treats TRAIN-53 as a source/mechanics incumbent and supplies fresh LOCAL_FREE source-equivalent measurements.

## Fixed controls

- tokenizer: `s0-byte-v1`, V256;
- real training documents: bounded Ukrainian Verkhovna Rada law object plus pinned Standard Ebooks `8-typography.rst`;
- held-out document: pinned Standard Ebooks `9-metadata.rst`;
- microbatch: 4 examples x 64 tokens = 252 causal loss tokens;
- effective loss-token batches: 252 / 504 / 1008 / 2016 through accumulation 1 / 2 / 4 / 8;
- matched main budget: 128 microbatches = 32,256 optimized loss tokens for every 500K/1M arm;
- seed 1337;
- optimizer: fp32 AdamW, LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0, constant LR, no warmup, clip norm 1.0;
- no LR scaling with batch;
- exact RESEARCH41 geometries: 467,808 and 1,037,696 parameters.

The bounded three-object corpus is real external text and rights-reviewed for training, but it is not a representative broad pretraining corpus. That limitation is retained explicitly.

## Fresh LOCAL_FREE source-equivalent matrix

Runtime: CPython 3.13.5, PyTorch 2.10.0+cpu, Linux x86_64, CUDA unavailable, 2 torch threads. This differs from the repository locked Python 3.11.16/PyTorch 2.13 environment, so these numbers are diagnostic evidence rather than locked-head authority.

| Params | loss tokens/update | optimizer steps | final held-out BPB | clip rate | median update/weight | train wall s | tokens/s | peak RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 467,808 | 252 | 128 | 5.485320 | 1.0000 | 0.0021442 | 2.9202 | 11,045.8 | 403.5 |
| 467,808 | 504 | 64 | 6.297740 | 1.0000 | 0.0025036 | 4.7166 | 6,838.9 | 392.7 |
| 467,808 | 1,008 | 32 | 6.784905 | 1.0000 | 0.0026977 | 2.7438 | 11,755.9 | 401.5 |
| 467,808 | 2,016 | 16 | 7.091793 | 1.0000 | 0.0033054 | 17.0952 | 1,886.8 | 393.3 |
| 1,037,696 | 252 | 128 | 5.721156 | 0.9688 | 0.0023901 | 3.7403 | 8,623.9 | 430.7 |
| 1,037,696 | 504 | 64 | 6.239425 | 1.0000 | 0.0024755 | 4.1006 | 7,866.2 | 412.7 |
| 1,037,696 | 1,008 | 32 | 6.498706 | 1.0000 | 0.0028598 | 4.3486 | 7,417.5 | 432.0 |
| 1,037,696 | 2,016 | 16 | 6.872496 | 1.0000 | 0.0032723 | 3.6285 | 8,889.6 | 414.1 |

Initial held-out BPB was 7.980024 at 467,808 parameters and 8.022766 at 1,037,696. At identical optimized-token budget, 252 tokens/update is the unique held-out winner at both sizes. Relative to that winner, final BPB is worse by 14.81%, 23.69%, and 29.29% for 504/1008/2016 at ~500K, and by 9.06%, 13.59%, and 20.12% at ~1M.

Wall-time numbers are recorded but are not used to infer universal hardware efficiency. This shared CPU was not isolated from other work and shows scheduler noise. Statistical comparisons are controlled by fixed microbatch geometry and matched token traces.

## Non-mutating gradient diagnostics

Eight sampled microbatch gradients were measured at initialization and after training for accumulation 1 and 4. The probe records squared mean-gradient signal, unbiased covariance trace, and `trace(cov)/||mean(g)||^2`, expressed additionally in local loss-token units. It fingerprints model weights before and after every probe and requires equality.

- 467,808 init proxy: 189.67 loss tokens; final accumulation-1 trajectory: 350.77; final accumulation-4 trajectory: 173.40.
- 1,037,696 init proxy: 220.77 loss tokens; final accumulation-1 trajectory: 914.98; final accumulation-4 trajectory: 167.97.

All probe weight fingerprints were preserved. These are local gradient-noise proxies only. Clipping is saturated or nearly saturated, the corpus is tiny, and checkpoint trajectories differ, so the values are not treated as a theoretical critical batch size.

## Statistical batch versus microbatch hardware

A dedicated ~500K control held effective batch and total optimized tokens fixed at 1,008 and 32,256 respectively while changing only microbatch implementation:

- 4x64 with accumulation 4: final BPB 6.784905008603018;
- 8x64 with accumulation 2: final BPB 6.784905008603018.

Median update/weight ratio also agrees to numerical precision. The 8x64 control measured 15,420 tokens/s versus 11,756 tokens/s for the 4x64 run in those particular CPU executions. Therefore the quality result is statistical/effective-batch behavior; microbatch shape can independently alter hardware throughput without changing the optimizer trajectory for the same grouped examples.

## Practical heuristic

Inside the measured 467,808–1,037,696 parameter interval, the empirically selected batch is 252 loss tokens/update at both endpoints. A log-log fit therefore has exponent alpha = 0.0. The conservative rule for the current fixed optimizer is:

`B_eff(P) = 252 loss tokens/update for approximately 0.5M through 1M parameters.`

Do not increase LR merely because batch increases. Reprobe if clipping policy, optimizer schedule, tokenizer, corpus, or microbatch geometry changes materially.

For larger models, use a bounded sublinear transfer envelope rather than extrapolating a critical batch. A practical next-scale guide is `252 * sqrt(max(P, 1M)/1M)`, quantized to the already tested loss-token grid and capped at 1,008 until fresh evidence exists. This returns 252 around 1M and approximately 1,008 around 10M. It is an engineering heuristic fitted to these observations, not a theory statement.

## Free 10M transfer validation

A current free CPU environment existed, so TRAIN-128 did not mark 10M `NOT_RUN`. The current byte-compatible S3 geometry from SCALE-03 was constructed exactly: V256/context 1024, D256/L12/Hq8/Hkv2/head_dim32/FF864, 10,000,640 parameters.

Two preregistered candidates were run for the same 16,128 optimized loss tokens:

| Params | loss tokens/update | optimizer steps | initial BPB | final BPB | clip rate | median update/weight | wall s | tokens/s | peak RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000,640 | 504 | 32 | 8.083528 | 7.663770 | 1.0000 | 0.0036159 | 14.3441 | 1,124.4 | 701.2 |
| 10,000,640 | 1,008 | 16 | 8.083528 | 7.444925 | 1.0000 | 0.0038338 | 26.2395 | 614.6 | 707.7 |

The short 10M transfer smoke favors 1,008 over 504 on held-out BPB. It is a smaller token-budget validation than the 500K/1M matrix and must not be pooled as an equal-budget scaling-law point. It does show that the useful batch region can move upward by ~10M.

## Decision

For the next fixed-recipe ~500K–1M real-source run, use 252 loss tokens/update unless hardware requires a larger microbatch grouping, in which case preserve the effective 252 target where feasible. For ~10M, start the next substantive run at 1,008 loss tokens/update and retain 504 as the smaller control; do not test a larger batch until a matched-budget 10M probe confirms that 1,008 is not a grid edge.

The wider MILESTONE-100 learned-Base goal is not closed by TRAIN-128: the current rights-approved real corpus used here is intentionally non-representative, and this batch experiment does not supply retained canonical checkpoints/resume/generation proof for a representative corpus. Those gates remain explicit rather than being inferred from batch evidence.
