# Research Corpus V1 and pretraining budget gates

Status: `PLANNING_AND_FAIL_CLOSED_GATING_NOT_TRAINING_AUTHORIZATION`

This document closes a unit/claim ambiguity in the ~20M campaign. Parameter count, normalized source bytes, tokenizer tokens, optimized causal loss positions, epochs and FLOPs are different quantities. They must never be substituted for one another.

## Live project boundary

The primary MODEL-341 mechanics candidate has 20,613,440 randomly initialized parameters. The live ~20M readiness controller keeps the campaign at `BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING` because DATA-301 still has no terminal corpus identity, no terminal shard identity and zero authorized balanced no-replay loss positions.

This package does not change that decision and does not authorize paid compute.

## Three distinct training modes

### 1. MECHANICS_PILOT

Purpose: prove forward/backward, trainer behavior, checkpoint/save/load/resume, generation, inference and data plumbing.

A mechanics pilot may use LOCAL_FREE bounded fixtures where existing lane rules permit it. It cannot establish model quality, scientific learned-model admission or sufficient pretraining exposure.

### 2. RESEARCH_CAMPAIGN

Purpose: a project-local learned-model experiment with immutable external-real data, exact no-replay optimized-position accounting and preregistered evaluation/checkpoint boundaries.

For the primary ~20M candidate, existing LEARN-345/RESEARCH-313 planning keeps the meaningful internal envelope at 10M-40M unique optimized positions, with 20M positions as the preregistered requested campaign budget. This is an internal experimental envelope, not a universal scaling law and not a quality-optimal claim.

The campaign remains blocked until an exact terminal Research Corpus V1, shard identity and post-pack unique-loss ledger exist.

### 3. QUALITY_PRETRAIN_REFERENCE

Purpose: keep long-range data/compute planning from confusing the project-local short research run with quality-focused pretraining.

Google DeepMind's Chinchilla experiment trained a 70B model on 1.3T tokens. The derived ratio is about 18.57 training tokens per parameter. Applying that ratio only as a planning reference gives approximately:

- 20,613,440 parameters -> 382,821,029 tokens;
- 100,000,000 parameters -> 1,857,142,858 tokens;
- 1,000,000,000 parameters -> 18,571,428,572 tokens.

Source: https://deepmind.google/discover/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/

This is not a hard 12-6 gate. Chinchilla's result is compute-optimal under its study assumptions, not a universal quality optimum. Modern open models often deliberately train much longer. Ai2 reports Olmo 2 1B pretraining on 4T tokens, which is 4,000 nominal tokens per parameter at 1B scale.

Source: https://docs.allenai.org/release_notes/olmo-release-notes

The Olmo observation is also not a 12-6 requirement. It demonstrates why our architecture ladder and data ladder must be planned separately.

## Research Corpus V1 acquisition gap

The current planning snapshot from NEXT100-069 / PR #495 records dedup-certified capacity:

| Stratum | Target | Current | Gap | Families | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ukrainian text | 9,000,000 B | 90,044 B | 8,909,956 B | 2 | 2 |
| English text | 7,000,000 B | 84,793 B | 6,915,207 B | 1 | 2 |
| Code | 4,000,000 B | 69,133 B | 3,930,867 B | 4 | 2 |
| Total | 20,000,000 B | 243,970 B | 19,756,030 B | - | - |

Because English has only one independently credited family at this snapshot, the hard family gate fails and the current feasible fixed 45/35/20 mixture is zero. Adding a second English family removes the family-count blocker but does not solve the volume deficit.

The 20MB target is therefore a minimum Research Corpus V1 acquisition/pipeline milestone. It must not be described as sufficient quality-focused pretraining data for a 20M-parameter Base.

## Critical path

1. Converge terminal training-admitted source authorities into one successor source registry instead of accumulating isolated source PRs indefinitely.
2. Prioritize high-volume independent UA and EN text families plus high-quality code capacity. Tiny library-source PRs should not dominate engineering time when the remaining gap is measured in millions of bytes.
3. Freeze one exact pre-decontamination candidate record inventory and identity.
4. Run quality, privacy, global exact/near dedup and evaluation decontamination against that exact identity.
5. Create cluster-safe train/validation splits, tokenize/pack them and materialize an exact post-pack unique-loss ledger. Source bytes are not a substitute for this ledger.
6. Rebuild from two clean roots and require byte-identical corpus/shard/tree identities.
7. Requalify checkpoint and bounded trainer mechanics against the primary MODEL-341 candidate.
8. Only then consider a compute-authorization request for a preregistered learned campaign.

## Scale path after ~20M

For ~100M, preserve the same ModelSpec/tokenizer/data/checkpoint identities and move first to FSDP2-style sharded state when single-device memory or optimizer-state pressure justifies it. Do not introduce tensor/pipeline/context parallelism merely because the parameter count increased.

For ~1B, qualify multi-GPU topology before training: FSDP2/HSDP plus tensor parallelism where measured memory/throughput requires it, distributed checkpointing, deterministic data continuation and failure recovery. TorchTitan is a suitable maintained integration target because current PyTorch work composes FSDP2 with tensor parallelism and supports larger multi-dimensional parallel strategies; the project should reuse those primitives rather than create a bespoke distributed runtime.

Current PyTorch references:
- https://pytorch.org/blog/torchcomms/
- https://pytorch.org/blog/efficient-moe-pre-training-at-scale-with-torchtitan/
- https://pytorch.org/blog/6x-faster-async-checkpointing/

Sparse MoE remains a later research path for substantially larger scales. It is not needed to reach the immediate 20M/100M/1B dense milestones.

## Queue discipline

The repository currently has heavy Actions saturation and many overlapping remediation PRs. This package intentionally adds no dedicated workflow. It should be verified through focused local execution and later absorbed into a converged CI surface rather than creating another per-worker workflow queue entry.
