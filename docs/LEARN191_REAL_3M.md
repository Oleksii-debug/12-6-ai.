# LEARN-191 — exact learned 3M bridge

`LEARN-191-REAL-3M` turns the RESEARCH-138 bridge recommendation into one scratch-trained Base artifact on the current DATA-25 / `s0-byte-v1` research truth model. It is a learned run, not a mechanics smoke and not a stage-promotion claim.

## Geometry

The bridge stays inside the RESEARCH-41 fixed-control family: byte vocabulary 256, maximum context 256, MHA, RMSNorm, pre-norm, SwiGLU, RoPE, tied embeddings, no attention/MLP bias and no dropout.

Exact ModelSpec: `d_model=192`, `n_layers=7`, `n_heads=n_kv_heads=12`, `head_dim=16`, `d_ff=528`, 3,213,120 trainable parameters, ModelSpec SHA-256 `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc`.

This is 8,312 parameters below the RESEARCH-138 geometric target of 3,221,432 (about 0.26%) while preserving the 1M rung's 16-wide heads and 2.75x FFN ratio.

## Frozen training truth

DATA-25 corpus identity is `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`. Tokenization is canonical `s0-byte-v1`, vocab 256, no special tokens. Training retains document-isolated seq-128 packing, batch 8, the M150 UA/EN/code mixture cadence, random initialization seed 1337, default InitSpec, and AdamW at LR 3e-4, betas 0.9/0.95, epsilon 1e-8, weight decay 0, constant schedule, FP32, deterministic algorithms, global gradient clipping 1.0.

No pretrained weights, SFT, RLHF, DPO or paid compute are permitted.

## Token budget and exposure

RESEARCH-138 identified 16,632 / 65,772 / 131,292 optimized tokens as the informative trajectory and observed structural deterioration by roughly 262K tokens at the 468K scale. LEARN-191 therefore stops at 131,292 optimized tokens and does not extend the curve after observing results.

DATA-25 contains 20,000,775 train byte-tokens. The final target is about 0.656% of one corpus pass, below the preregistered 1% source-exposure ceiling. This run is not a recycling experiment.

The 65,772-token checkpoint is the mandatory process boundary. Phase 1 exits after writing it; a fresh Python process must restore it and complete the final segment.

## Immutable selection validation

Checkpoint selection uses a deterministic preregistered subset of DATA-25 validation: first 256 packed UA examples, first 192 EN examples and first 128 code examples, with canonical byte tokenization, seq 128 and document isolation. The subset identity is hash-bound before optimizer step 1. It is selection-validation, not final-test authority.

Evaluation occurs at random initialization and the first optimizer boundary reaching 16,632, 65,772 and 131,292 optimized tokens. Each evaluation records aggregate BPB, UA/EN/code BPB, DATA-25 source-family BPB, model-state hashes before/after and non-mutation proof.

## Diagnostics and retention

The trajectory records train BPB, held-out BPB, fixed train-probe BPB, train-minus-validation BPB gap as a bounded memorization/generalization diagnostic, raw pre-clip gradient norms, clip activations, sampled parameter update ratios, optimized tokens, throughput, wall time, memory telemetry and first-party greedy Base generation. The train-probe gap is not a privacy-leakage measurement.

All three trained checkpoints are retained. `best-checkpoint.json` is selected only by minimum preregistered selection-validation BPB. `final-checkpoint.json` always points to the 131,292-token checkpoint. A separate fresh process strictly reloads and generates from the final checkpoint.

## Execution integrity and truth boundary

The workflow uses the universal exact hash-locked bootstrap with toolchain, runtime and dev locks. CI-165's launch gate creates and verifies a source-SHA/config/profile/corpus/ModelSpec/tokenizer/budget/output-bound launch envelope before phase 1. Phase 1 and resume both refuse absent or stale envelopes.

DATA-25 is project-authored research data and contains no rights-approved external training source in this identity. Therefore this artifact supports controlled ladder comparison only. It makes no external-representativeness, intelligence, alignment, instruction-following, privacy-leakage or production-readiness claim.
