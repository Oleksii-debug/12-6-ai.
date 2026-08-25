# TRAIN-130 parameter-group learning-rate experiment

## Authority and boundary

Worker: `TRAIN-130-PARAM-GROUP-LR`

Experimental branch: `train130/param-group-lr-20260826`

Frozen convergence base: `b9bc147e0a08181b91798c2515cac7a79c66791c` (`milestone100/first-learned-base-20260826`).

Authority: `LOCAL_FREE_EXPERIMENTAL_OPTIMIZER_EVIDENCE_NOT_CANONICAL_PROMOTION`.

This experiment does not modify the canonical `Trainer`, `build_optimizer`, AdamW defaults, scheduler, clipping, model implementation, tokenizer implementation, checkpoint format, or evaluation stack. The retained runner injects an experimental AdamW instance through the incumbent `Trainer(..., optimizer=...)` interface.

## Research question

For small decoder-only Base models with a physically shared token-embedding/output-head tensor, test whether modest group-specific learning-rate reduction is better than one global learning rate.

The fixed control protocol is:

- AdamW
- base learning rate `3e-4`
- betas `(0.9, 0.95)`
- epsilon `1e-8`
- weight decay `0.0`
- constant schedule, no warmup
- global gradient clipping `1.0`
- fp32 deterministic CPU execution
- batch size `4`
- sequence length `64`
- `252` predicted byte tokens per optimizer step
- seeds `1337` and `1338`
- `96` optimizer steps per run, `24,192` optimized target bytes per run

Variants:

- A: one global LR, `3e-4` for every trainable tensor.
- B: tied embedding/output tensor at `1.5e-4`; all other tensors at `3e-4`.
- C: RMSNorm weights at `1.5e-4`; all other tensors at `3e-4`.
- D: both reductions, but only if B and C independently improve held-out BPB at both scales.

No other optimizer hyperparameter is changed.

## Models

Both controls use the incumbent fixed-vocabulary scaling family and random initialization.

| Label | Parameters | d_model | Layers | Heads | d_ff | Vocab | Tied embeddings |
|---|---:|---:|---:|---:|---:|---:|---|
| ~500K | 467,808 | 96 | 4 | 6 | 256 | 256 | yes |
| ~1M | 1,037,696 | 128 | 5 | 8 | 352 | 256 | yes |

The physical identity `token_embedding.weight is lm_head.weight` is required by the grouping code. The tied tensor is placed in exactly one optimizer group. RMSNorm weights are identified by module type. Remaining trainable tensors form the base group. Parameter names are sorted and the partition is asserted to be disjoint and exhaustive.

## Bounded real corpus

The experiment uses the same train and held-out corpus for every variant and seed. The committed excerpts are real source text, normalized only to remove source-format markup and isolate disjoint train/held-out excerpts; records are explicitly `synthetic:false`.

Training file: `data/experiments/train130/bounded-real-train.jsonl`

Held-out file: `data/experiments/train130/bounded-real-validation.jsonl`

Training file SHA-256: `695ddbfc96fec101c6fa7cde19a92e807b6422e1e45dc6a56fbead81ac623a7b`

Held-out file SHA-256: `592b06d2059a1617c83ac96749f350a141b784feb8fdd953314ad7f75575cad2`

Sources:

1. `en.standardebooks.manual`, pinned git revision `d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b`, CC0-1.0. Training excerpt comes from `8-typography.rst`; held-out excerpt comes from `9-metadata.rst`.
2. `ua.rada.open-data.laws-texts`, bounded official-law source/version reviewed by DATA-21 as approved for training under Rada open-data reuse and the official-acts copyright boundary.

This is legitimate bounded real-source experimental data. It is **not** a canonical corpus freeze and is **not** evidence of broad real-world representativeness. The MILESTONE-100 representative-corpus gate therefore remains fail-closed.

## Results

Two seeds agree on direction at both scales.

| Scale | Variant | Mean final held-out BPB | Mean final lexical BPB | Held-out change vs A | Decision |
|---|---|---:|---:|---:|---|
| 467,808 | A global | 4.772663 | 4.818798 | control | retain |
| 467,808 | B embedding/output 0.5x | 5.649560 | 5.684849 | 18.373% worse | reject |
| 467,808 | C norms 0.5x | 4.792435 | 4.838541 | 0.414% worse | reject |
| 1,037,696 | A global | 4.192806 | 4.244027 | control | retain |
| 1,037,696 | B embedding/output 0.5x | 4.954611 | 4.976277 | 18.169% worse | reject |
| 1,037,696 | C norms 0.5x | 4.213421 | 4.266202 | 0.492% worse | reject |

Seed-level final held-out BPB:

- 467,808 A: `4.813113`, `4.732214`
- 467,808 B: `5.697900`, `5.601220`
- 467,808 C: `4.832391`, `4.752479`
- 1,037,696 A: `4.266642`, `4.118970`
- 1,037,696 B: `4.935543`, `4.973680`
- 1,037,696 C: `4.287025`, `4.139817`

Train-loss direction is consistent with held-out quality. Mean first-32 to last-32 update loss:

- 467,808 A: `4.939859 -> 3.543688`
- 467,808 B: `5.060074 -> 4.129411`
- 467,808 C: `4.940825 -> 3.555028`
- 1,037,696 A: `4.734675 -> 3.177618`
- 1,037,696 B: `4.867362 -> 3.698329`
- 1,037,696 C: `4.735891 -> 3.186617`

## Group dynamics

The intended update-ratio intervention occurred, but did not improve quality.

At 467,808 parameters, the tied tensor mean update/weight ratio changes from `0.0100041` in A to `0.00550431` in B. Its mean gradient L2 rises from `2.44276` to `2.74194`, while final tied-tensor weight L2 grows only from `3.12902` to `3.50833`, versus `4.40163` in A.

At 1,037,696 parameters, the tied tensor mean update/weight ratio changes from `0.00926273` in A to `0.00515134` in B. Its mean gradient L2 rises from `2.71803` to `3.07524`, while final tied-tensor weight L2 grows only from `3.62723` to `4.00071`, versus `4.81662` in A.

This is direct evidence that reducing the shared embedding/output LR slows lexical/output learning. Because the same physical tensor serves both token lookup and the LM projection, the reduced step size constrains adaptation on both sides at once. The increase in gradient norm does not compensate for the reduced parameter motion. The result is materially worse train loss, lexical BPB, and overall held-out BPB at both sizes.

For norms, C approximately halves the intended norm update/weight ratio (`1.58e-4 -> 7.91e-5` at 500K; `1.50e-4 -> 7.50e-5` at 1M) without a held-out or stability benefit. This is a cosmetic optimizer-dynamics difference, not a quality win.

## Stability and resume

No run produced a non-finite loss or optimizer failure. The pre-clip global gradient norm exceeded the fixed `1.0` clip threshold on every measured update in A, B, and C, so the variants do not demonstrate a stability advantage under this protocol.

Deterministic resume probes used a committed split at optimizer step 24 followed by four replayed steps. For every scale/variant pair A/B/C:

- replay losses were exactly equal,
- model-state hashes were exactly equal,
- optimizer-state signatures were exactly equal.

Held-out evaluation hashes model state before and after evaluation and fails if evaluation mutates parameters or persistent state.

## D decision

D was **not run**. Its prerequisite was intentionally strict: both B and C had to independently improve held-out BPB at both scales. B is decisively worse and C is slightly but repeatably worse. Running D would add combinatorial surface without evidentiary justification.

## Canonical decision

**No canonical optimizer change. Retain A, the incumbent global learning rate. Reject B and C. Do not promote D.**

The evidence supports a negative result: modest per-group LR reduction is not beneficial for these tied byte-level Base controls under the fixed AdamW protocol.

## Retained evidence

Compact retained report:

`reports/train130/param_group_lr_local_20260826.json`

Incumbent-backed reproduction runner:

`tools/run_train130_param_group_lr.py`

The runner is intentionally thin. It imports the first-party model, tokenizer and Trainer and supplies only the experimental optimizer grouping plus read-only group telemetry.

## Reproduction command

From an exact checkout of the experiment branch:

```bash
git checkout train130/param-group-lr-20260826
HEAD_SHA=$(git rev-parse HEAD)
PYTHONPATH=src python tools/run_train130_param_group_lr.py \
  --expected-source-sha "$HEAD_SHA" \
  --steps 96 \
  --seeds 1337,1338 \
  --output reports/train130/param_group_lr_reproduction.json
```

The runner records its source SHA and refuses an explicit SHA mismatch.

## Machine/execution truth boundary

The completed LOCAL_FREE measurements reported above were executed in the available tool runtime with:

- Python `3.13.5`
- PyTorch `2.10.0+cpu`
- CUDA unavailable
- Linux `6.18.35-x86_64-with-glibc2.41`
- two PyTorch CPU threads
- no paid compute

The runtime used for the measurements did not have a git/network bridge capable of cloning or mounting the GitHub connector checkout. Therefore the completed measurements are protocol-equivalent local evidence against the frozen incumbent code/contracts, but are **not claimed as an executed exact Git checkout of the final branch head**. The exact-checkout first-party runner is retained on the branch for that final reproduction step.

A longer 65,536-token-class matrix was attempted first but exceeded the execution wall-time ceiling before all A/B/C/seed cells completed. Its partial outputs were discarded and are not evidence. The retained matrix is the fully completed 24,192-target-byte-per-run protocol above.

## MILESTONE-100 consequence

The existing MILESTONE-100 branch remains the strongest complete learned-artifact spine for random initialization, versioned tokenizer/packing, checkpointing, fresh-process resume, held-out BPB, evaluation non-mutation and first-party generation. Its own DATA-25 truth boundary states that the retained corpus is project-authored and has zero external training-eligible sources.

TRAIN-130 adds bounded real-source optimizer evidence at 467,808 and 1,037,696 parameters, but it does not convert that bounded sample into a representative canonical pretraining corpus and it does not replace the MILESTONE-100 retained learned checkpoint.

Accordingly, the broader requested milestone must remain **NOT FULLY PROVEN** until a versioned representative real-source corpus is frozen through the project data gates and the exact branch/head learned run is executed against it. This limitation is intentional and fail-closed.
