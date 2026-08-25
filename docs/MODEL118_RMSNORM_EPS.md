# MODEL-118 RMSNorm epsilon qualification

Decision: **KEEP_CURRENT (`norm_eps=1e-5`)**.

## Pinned source

- Repository: `Oleksii-debug/12-6-ai.`
- Experimental base: `milestone100/first-learned-base-20260826` at `015593b22a600184fb4c8001fe3d70893bfc51d5`.
- Incumbent model blob: `c3879fe0ba9193d5a8176c284e1942f058ef7885`.
- Incumbent Trainer blob: `8fb5e9ce4c5417986ad1f086ebc16cd7538a151e`.
- TRAIN-54 layer-health reference blob: `8b7772ca4020627cf02cc42c4f70f15683739735`.

The incumbent RMSNorm remains unchanged: variance is accumulated in fp32, epsilon is added before `rsqrt`, the inverse denominator is cast back to the activation dtype, and the result is multiplied by the learned weight. `norm_eps` is already part of `ModelSpec`, so changing it changes model identity.

## Experiment

Log grid: `1e-6`, `1e-5`, `1e-4`. Canonical S1 (107,856 params) and S2 (1,066,112 params) were trained for 64 optimizer steps with identical seed, byte-token data order, optimizer, initialization, batch size, sequence length, and geometry within each scale. FP32 is the reference. CPU BF16 was executed only after the local runtime successfully executed bf16 autocast + SDPA. TRAIN-54-style hooks were registered only at steps 0, 4, 16 and 64 and removed immediately after each diagnostic. A 10,059,840-parameter S3 initialization smoke covered all grid values in fp32 and bf16.

The committed S0 train/validation fixture was used because the larger corpus v0.1 manifest exists on the base branch but its shard payloads do not. These results are numerical-mechanics evidence, not corpus-quality evidence.

## Primary seed 1337

| Scale | Precision | eps | BPB 0 -> 64 (FP32 eval) | denom min @0 | grad norm @0 | update ratio @64 | finite events |
|---|---|---:|---:|---:|---:|---:|---:|
| s1_100k | fp32 | 1e-06 | 9.018588 -> 7.166526 | 0.016175 | 1.5944 | 0.002387 | 0 |
| s1_100k | fp32 | 1e-05 | 9.018193 -> 7.167187 | 0.016450 | 1.5656 | 0.002385 | 0 |
| s1_100k | fp32 | 1e-04 | 9.014980 -> 7.161987 | 0.019000 | 1.3385 | 0.002422 | 0 |
| s1_100k | bf16 | 1e-06 | 9.018588 -> 7.166429 | 0.016179 | 1.5969 | 0.002388 | 0 |
| s1_100k | bf16 | 1e-05 | 9.018193 -> 7.167116 | 0.016451 | 1.5679 | 0.002386 | 0 |
| s1_100k | bf16 | 1e-04 | 9.014980 -> 7.161931 | 0.019000 | 1.3407 | 0.002422 | 0 |
| s2_1m | fp32 | 1e-06 | 11.072571 -> 5.961855 | 0.016891 | 6.3712 | 0.003703 | 0 |
| s2_1m | fp32 | 1e-05 | 11.072727 -> 5.962578 | 0.017153 | 6.2754 | 0.003712 | 0 |
| s2_1m | fp32 | 1e-04 | 11.072316 -> 5.961731 | 0.019593 | 5.3781 | 0.003764 | 0 |
| s2_1m | bf16 | 1e-06 | 11.072571 -> 5.961600 | 0.016890 | 6.3723 | 0.003705 | 0 |
| s2_1m | bf16 | 1e-05 | 11.072727 -> 5.961887 | 0.017154 | 6.2759 | 0.003714 | 0 |
| s2_1m | bf16 | 1e-04 | 11.072316 -> 5.961194 | 0.019591 | 5.3795 | 0.003760 | 0 |

All diagnostic windows were finite. All held-out evaluations were non-mutating. The maximum absolute BF16-native vs FP32 reevaluation difference at step 64 was about `1.08e-4` BPB.

## Interpretation

At the S1 initialization median RMSNorm input scale, `1e-5` is about 2.5% of the median input variance; at S2 it is about 2.0%; at S3 about 0.36%. It therefore supplies a modest small-model denominator floor while becoming progressively less intrusive with scale. By contrast, `1e-4` is about 25% of S1 median variance and 21% of S2 median variance, reducing the epsilon-free normalization gain to roughly 0.89 and 0.91 respectively. That is a real semantic attenuation, not merely a numerical guard.

The BPB ranking is not consistent enough to justify a change. At S1, `1e-4` wins for seeds 42 and 1337 but loses to current `1e-5` for seed 2026. At S2 fp32, the winner is `1e-5` for seed 42, `1e-4` for seed 1337, and `1e-6` for seed 2026. BF16 likewise changes ordering. The differences are small relative to the fixture and short-run sensitivity.

`1e-6` preserves more pure RMS normalization and starts with slightly larger gradient norms, but it produced no finite-state, BPB, update-ratio, or BF16 advantage that was consistent across scales and seeds. `1e-4` dampens early gradients but over-regularizes the denominator at the smallest scales without a consistent held-out benefit.

## Checkpoint/resume and identity

For current `1e-5`, S1 and S2 checkpoints taken at optimizer step 16 were loaded by fresh Python processes and resumed to step 64. Both resumed final parameter hashes matched uninterrupted training exactly. The repository checkpoint-v1 identity already stores the complete model spec and its hash, so an epsilon change is fail-closed under the existing identity contract. No checkpoint format change is required.

## Decision

**KEEP_CURRENT.** Retain `norm_eps=1e-5` in canonical S1/S2/S3. Do not create an experimental successor from this evidence. Re-open only if a longer representative-corpus run or accelerator-specific low-precision campaign shows a consistent finite-state or quality failure attributable to the denominator floor.
