# RECOVER-176 — EVAL-133 English on LEARNED BASE LADDER V1

RECOVER-176 is a convergence recovery, not a new model family and not a new English benchmark.

The accepted learned Base ladder is MILESTONE-150 at exact source `1037439f65c48529904be170064bf69d0c75d18b`. Its comparable learned rungs are 95,568 parameters (~100K), 467,808 parameters (~500K), and 1,037,696 parameters (~1M), all trained from random initialization under one DATA-25 V0.1 / `s0-byte-v1` / document-isolated 128-byte / held-out BPB truth model. The ~10M rung remains `INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE`; RECOVER-176 does not promote it.

## Immutable EVAL-133 recovery

The suite, reservation registry, scorer, and original tests are restored byte-for-byte from the accepted EVAL-133 branch and guarded by their original Git blob object IDs. The suite identity remains `eval133-en-raw-v1` version `1.0.0`, SHA-256 `f9e713ff336e6189f7aa0ddbb21303431ab2041b6700ed38243eaf65865805cb`. The reservation identity remains `850e0c34fd6ab35d0829b3f78ff5e81fbcb8c1ee900f3e7f1b967ea23a8f2e40`.

The old EVAL-133 workflow is not reused because its exact run failed before testing with `No module named pytest`. RECOVER-176 installs the complete hash-locked Linux x86_64 toolchain, runtime, and dev/test locks before running the original EVAL-133 tests.

## Corpus exclusion gate

DATA-25 predates the EVAL-133 reservation registry, so its original corpus manifest alone is not accepted as proof of exclusion. RECOVER-176 rebuilds the exact M150 DATA-25 corpus, then runs the immutable EVAL-133 normalized decontamination logic over both DATA-25 train and validation text. It also checks the retained legacy S0 packaged train and validation material.

Any exact full-alternative collision or normalized reserved context/full-continuation substring collision fails the recovery.

## Checkpoints evaluated

For every comparable M150 scale, RECOVER-176 evaluates the step-0 random-initialization checkpoint, the M150 best held-out-BPB checkpoint, and the M150 final step-1000 checkpoint. When best and final are the same checkpoint, the checkpoint is evaluated once and carries both roles.

Before EVAL-133 scoring, each checkpoint is verified and loaded with M150 source, ModelSpec, tokenizer, DATA-25 corpus, run-manifest, and seed bindings. The original EVAL-133 scorer checks model-state and model-mode non-mutation. RECOVER-176 additionally hashes Trainer state before and after every scoring pass and fails if it changes.

The report includes pair accuracy, raw log-likelihood margins, token/UTF-8-byte-normalized margins, all eight per-phenomenon breakdowns, length diagnostics, item scores, and a scale trend based on each scale's M150 best checkpoint. No monotonic scaling improvement is assumed; the measured trend is reported as observed.

## Authoritative output

The machine-readable output is `recover176-evidence/learned-base-ladder-v1-eval133.json`. It embeds the complete validated M150 LEARNED BASE LADDER V1 report, including its quality/efficiency/scaling rankings, exact model/training/checkpoint/generation evidence, and 10M incomplete boundary, then adds immutable EVAL-133 identity, decontamination evidence, random/best/final English raw-LM results, non-mutation evidence, and scale trend.

This artifact is evidence for a learned scratch Base language-model ladder and a project-authored raw English minimal-pair diagnostic only. It makes no claim of instruction following, intelligence, alignment, production readiness, or broad English proficiency. It uses no foreign pretrained weights, SFT, RLHF, DPO, or paid compute.
