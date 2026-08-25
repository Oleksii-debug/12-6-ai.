# MODEL-120: residual-scale transfer at the first deeper ~10M geometry

Status: **CLOSED — retain InitSpec v1**.

This is a focused transfer test, not a new initialization sweep. It consumes MODEL-19/MODEL-34 and TRAIN-54 and uses the current SCALE-03 byte-compatible ~10M geometry. No paid compute, pretrained weights, instruction tuning, stage-promotion claim, or broad capability claim is involved.

## Exact source identities

Source convergence head used for reconstruction: `b9bc147e0a08181b91798c2515cac7a79c66791c`.

Current selected geometry: ModelSpec `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`, 10,000,640 trainable parameters, V256, context 1024, D256, L12, Q8/KV2, Dh32, SwiGLU F864, pre-RMSNorm/RoPE, tied embeddings.

Incumbent initialization identity: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5` = `Normal(0, 0.02)` for ordinary weights with `attn.out_proj` and `mlp.down_proj` initialized at `0.02/sqrt(2L)`.

Compared only against the still-live MODEL-19 width-reference alternative, `0.02*sqrt(48/D)` with the same residual depth scaling. One unscaled residual control was used at initialization only because the scaled candidates already supplied the training comparison.

## Execution truth boundary

The current LOCAL_FREE container had no DNS path to github.com, so an exact checkout could not be cloned. GitHub source/config bytes and identities were reconstructed through the authenticated repository connector; the local CPU run used a source-extracted implementation matching the exact model semantics and current ModelSpec. Therefore the measurements below are **LOCAL_FREE source-extracted semantic evidence, not an exact-checkout CI authority claim**.

Machine: Python 3.13.5, PyTorch 2.10.0+cpu, CUDA unavailable, 5 CPU cores exposed, AMD EPYC 9V74, fp32 deterministic algorithms, 5 Torch threads.

Training/validation data for this mechanics test were disjoint real local Python/PyTorch source files: 5,057,472 training bytes from 218 files and 505,260 validation bytes from 17 files, split by a deterministic hash of file path. This is real source-code/text trajectory evidence; it is **not** evidence that the repository training corpus is externally representative.

## Initialization seeds

Seeds: 1337, 1338, 1339 for incumbent and width-reference. Unscaled bad-control: seed 1337 only.

| Candidate | Initial held-out BPB | Global raw grad norm | block-output RMS across depth | block RMS depth ratio |
|---|---:|---:|---:|---:|
| incumbent 1337 | 7.5372 | 34.89 | 0.02230 → 0.05650 | 2.53 |
| incumbent 1338 | 7.4902 | 32.17 | 0.02204 → 0.05950 | 2.70 |
| incumbent 1339 | 7.6048 | 36.11 | 0.02190 → 0.06068 | 2.77 |
| width-ref 1337 | 7.4318 | 13.30 | 0.00889 → 0.01065 | 1.20 |
| width-ref 1338 | 7.4674 | 12.02 | 0.00876 → 0.01075 | 1.23 |
| width-ref 1339 | 7.4712 | 13.96 | 0.00858 → 0.01045 | 1.22 |
| unscaled 1337 | 7.9649 | 13.56 | 0.05188 → 0.31813 | 6.13 |

Incumbent attention-output RMS depth ratios were 2.41–2.80 and MLP-output ratios 1.10–1.16. Its per-layer attention-gradient depth ratios were 2.38–2.56 and MLP-gradient ratios 2.52–2.87. Width-reference is flatter at initialization, but that alone is not a promotion criterion. The unscaled control is clearly the bad control: much larger residual depth growth and worse BPB.

## Matched real training trajectories

Each scaled candidate ran 32 AdamW updates for each of seeds 1337/1338/1339: batch 2, sequence 128, lr 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0, global grad clip 1.0, fp32 deterministic execution.

| Candidate | mean held-out BPB before | mean held-out BPB after | mean BPB improvement | mean first train loss | mean last train loss |
|---|---:|---:|---:|---:|---:|
| incumbent | 7.6997 | **4.3175** | **43.92%** | 5.1763 | **2.7697** |
| width-reference | **7.6021** | 4.4973 | 40.84% | **5.1242** | 3.0088 |

Incumbent final BPB by seed: 4.3808, 4.2775, 4.2942. Width-reference: 4.4693, 4.5033, 4.5194. The incumbent wins final held-out BPB in all three matched seeds despite starting slightly worse.

The incumbent had no early loss excursion above its first update in any seed; maximum train loss equals the first-update loss (5.1645–5.1866). Width-reference reached 5.2286–5.2779 after starting around 5.1242 on average, a small but measurable early excursion.

All six scaled trajectories clipped on every update. That is a training-scale/optimizer observation, not evidence that the residual rule failed: changing to width-reference did not remove clipping and produced one much worse raw-gradient spike. Incumbent maximum raw gradient norms by seed were 64.91, 54.13, 70.33; width-reference maxima were 352.94, 36.90, 20.80.

Incumbent median parameter update ratios were 0.268%, 0.286%, 0.278%; maxima were approximately 0.990% for all three seeds. Width-reference medians were 0.321%, 0.319%, 0.280%; maxima approximately 1.13%.

At step 32, all activation and gradient measurements remained finite. Incumbent final block-output RMS maxima were 1.123, 1.412, 1.421 across the three seeds; attention/MLP branch-output RMS remained about 0.10–0.15. Residual depth ratios grow during learning, so TRAIN-54-style depth-ratio warnings should remain visible as diagnostics rather than be misread as proof of initialization failure.

## Checkpoint, resume, evaluation, generation

For incumbent seed 1337, checkpoints were retained at steps 16 and 32 during the local run.

- step 16: 120,150,809 bytes, SHA-256 `ec669f450a8ce731c3fcb873ed7164f1f66c8e21ac83df0c6924a9ae2af43182`
- step 32: 120,150,809 bytes, SHA-256 `652ccda3b39f6481ee8dd627cf95e4c2c0d5c880f8242c4c91f15fdc56d2477f`

A new Python process loaded the step-16 model+optimizer state, replayed updates 17–32, and produced model-state SHA-256 `743d39747857ceb4f177bfe79afaa7837c9b087c7c82e0ccb568ad96bb2a6bba`, exactly equal to the direct step-32 state hash.

Held-out evaluation was hash-checked as non-mutating for all six scaled trajectories.

A byte-greedy diagnostic from prompt `def stable_` changed after learning; it is retained only as evidence that generation uses learned weights, not as a quality claim. The output remains poor after 32 updates, as expected for this bounded mechanics trajectory.

## Decision

**Retain InitSpec v1 unchanged and close MODEL-120.**

The deeper L12 ~10M transfer does not show a failure that justifies changing initialization. The incumbent is trainable across all three seeds, decreases train loss strongly, improves held-out BPB by ~44%, remains finite through depth, beats the only previously justified alternative on final held-out BPB in every matched seed, and avoids the alternative's single ~353 raw-gradient spike. The unscaled control confirms that residual depth scaling remains necessary.

Do not promote the width-reference base std. Do not introduce DeepNorm, muP, or another initialization family from this result. Keep the 100% clipping observation open under optimizer/training-scale diagnostics if later longer trajectories reproduce it; it is not grounds for an InitSpec revision here.

## Reproduction contract

Versioned experiment config: `configs/experiments/model120_residual_scale_transfer_10m.json`.

For an authoritative rerun, execute the same candidate/seeds/optimizer/corpus protocol from an exact checkout of this MODEL-120 branch using the repository-native `TwelveSixDecoder`, TRAIN-54 layer-health instrumentation (or its ancestry-preserving successor), canonical checkpoint adapter, and held-out evaluator. The present local measurements must not be relabeled as exact-checkout authority merely because the branch records them.
