# TRAIN-325 — 10M Clipping Final V3

## Decision

`FINAL_RETAIN_CLIP_1P0_FOR_TESTED_LEARNED_10M_IDENTITY`

The final clipping policy for the exact learned-10M identity tested by TRAIN-243 is global gradient-norm clipping at **1.0**.

This is not a claim that 1.0 is universally optimal. It is the preregistered fallback because every less-intrusive candidate failed at least one frozen stability/quality gate.

## Evidence consumed

TRAIN-243 exact source `7ff794749a6c6f6f8f905314a29f64e6daaf2211`, Actions run `32960810083`, artifact `9604676694`, artifact digest `sha256:d99bcf3bdc96556bb4e1da87333e08bcfeb759a1cc27c94bdddd926faab4a526`.

The matched experiment used three paired seeds (`20260825`, `20260826`, `20260827`) with equal optimized-token exposure. Target exposure was 131,072; every arm executed 131,294 optimized tokens because batches crossed the target at the same boundary. LR and the rest of the optimizer recipe were fixed; TRAIN-325 does not co-tune LR and executes no additional optimizer updates.

## Candidate disposition

| Arm | Clip norm | Mean held-out BPB | Paired BPB delta vs 1.0 | Mean clip frequency | Frozen disposition |
|---|---:|---:|---:|---:|---|
| Unclipped | none | 0.613253 | +0.346710 | 0% | reject: quality false |
| q95 | 3.4 | 0.252270 | -0.014273 | 42.04% | reject: spike and tail-frequency gates fail |
| q90 | 2.1 | 0.250830 | -0.015713 | 73.92% | reject: quality and tail-frequency gates fail |
| Incumbent | 1.0 | 0.266543 | 0 | 93.78% in the matched short arms | retain fallback |

The apparent mean BPB advantage of q95/q90 is insufficient for promotion: their paired confidence intervals cross zero, and they fail preregistered non-quality gates. Selection follows the preregistered conjunction of gates rather than retrospective metric cherry-picking.

Unclipped is clearly worse on quality: paired BPB delta estimate `+0.346710`, 95% bootstrap interval `[+0.133791, +0.611357]`.

All tested arms were finite. TRAIN-243 separately proved NaN/Inf gradient failure occurs before clipping.

## Research Corpus V1 boundary

Latest DATA-301 terminal evidence at head `8820ba1b255f6bb95c7db0531fd846078a1aae01` does **not** publish a terminal frozen Research Corpus V1 identity. It fail-closes on a zero balanced no-replay budget, empty terminal selection-validation, missing exact Wave-3 quality/privacy reruns, missing full five-source unique-loss accounting, and missing two authorized clean byte-identical builds.

Therefore TRAIN-325 does not run a pseudo-transfer experiment on non-authoritative corpus bytes. `clip=1.0` is final for the exact tested learned-10M identity only. Once a terminal Research Corpus V1 exists, clipping must be revalidated under that corpus rather than silently assumed transferable.

## Scientific scope

- Model: exact 10,000,640-parameter learned 10M Base identity used by TRAIN-243.
- Tokenizer: canonical `s0-byte-v1`.
- Optimization: no LR co-tuning.
- Paired seeds: three.
- Paid compute: none.
- Foreign weights: none.
- New TRAIN-325 training: none, because terminal TRAIN-243 already supplies the required experiment and the requested new corpus authority is currently unavailable.
- Universal clipping optimum claim: forbidden.

Machine-readable authority: `evidence/train325/final_clipping_v3.json`.
