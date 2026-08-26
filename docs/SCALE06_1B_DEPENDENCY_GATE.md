# SCALE-06 ~1B dependency gate

## Decision boundary

The repository already contains an S6 engineering candidate at 999,761,920 random-init parameters. That fact alone is not evidence that a 1B learned run should start.

`scale_1b_readiness.py` adds a fail-closed evidence layer. Every material dependency is absent by default. A gate clears only when the caller binds it to a durable authority reference from the owning lane. Bare `--qualified` booleans and free-form prose are intentionally rejected because they would allow readiness to be self-attested without terminal evidence.

The report separates engineering/scientific readiness from compute authorization:

- `ready_for_authorization_request` becomes true only after all required system and scientific dependencies carry syntactically immutable authority references;
- `ready_for_material_compute` additionally requires an explicit authorization prefix plus a durable GitHub/artifact authority reference.

No stage promotion or compute authority is created by this module. The coordinator must still verify that each referenced authority is the live terminal authority before composing it.

## Current S6 geometry

- target: 1,000,000,000 parameters;
- exact current-tokenizer candidate: 999,761,920 (-0.023808%);
- vocabulary: 256 raw-byte IDs, tied embedding/output;
- context: 4096 byte tokens;
- width: 2048;
- depth: 18 blocks;
- attention: 32 query heads / 8 KV heads / head dimension 64;
- SwiGLU FFN: 7328;
- pre-RMSNorm, RoPE, random initialization.

The byte vocabulary is an execution-compatibility candidate, not a production-tokenizer decision. Replacing it with a learned vocabulary changes the embedding/output parameter surface and must produce a newly solved and newly hashed ModelSpec.

## Evidence-reference grammar

Engineering/scientific gates accept only one of these forms:

- `github:<scope>@<40-hex-commit>:success`
- `github:<scope>@<40-hex-commit>:pass`
- `github:<scope>@<40-hex-commit>:admitted`
- `github:<scope>@<40-hex-commit>:qualified`
- `artifact:<scope>@<64-hex-sha256>`

This is structural hardening, not remote verification. The coordinator must still check that a GitHub head is terminal and that an artifact hash belongs to the intended evidence object. A queued, running, stale, closed-unmerged or superseded authority must not be composed simply because its string matches the grammar.

Compute authorization is separate and uses the same immutable reference discipline. It must begin with `COMPUTE_AUTHORIZED:` or `TRAINING_AUTHORIZED:` and the suffix must itself be one of the GitHub/artifact authority forms above. For example, `COMPUTE_AUTHORIZED:artifact:owner-compute-approval@<64hex>` is structurally acceptable, while `COMPUTE_AUTHORIZED:yes`, `COMPUTE_AUTHORIZED:owner-message-id:test`, and a `github:...:queued` suffix are rejected. Green CI alone cannot create owner authorization; the coordinator must verify that the referenced terminal object is actually the intended approval authority.

## Required positive evidence

1. `preceding_stage_authority`: preceding scale stage admitted by its own evidence/promotion gate.
2. `production_tokenizer_authority`: versioned production tokenizer frozen from the eventual decontaminated corpus, with fertility/coverage/BPB and exact identity evidence.
3. `native_gqa_authority`: 32Q/8KV path integrated without materializing repeated K/V heads, with target-runtime numerical and accelerator evidence.
4. `distributed_checkpoint_authority`: DCP/FSDP2 save/load/resume and reshard composed on the target model/runtime with integrity evidence.
5. `data_pipeline_authority`: deterministic corpus ingestion, dedup/decontamination, split, packing, restart and accounting mechanics qualified.
6. `stage_data_budget_authority`: stage-specific authority that binds exact unique post-tokenization corpus-token and non-ignored causal-loss counts, a preregistered total training-token exposure budget, and the replay/epoch policy used to reach that exposure. The R01 20x/50x/100x ladder applies to total training-token exposure planning, not to a unique-data minimum.
7. `training_recipe_authority`: scale-qualified optimizer/LR/schedule/warmup/precision/gradient policy, initialization assumptions, seeds, training budget, checkpoint/evaluation cadence and stopping rule. A successful 20M recipe is not automatically transferable to 1B.
8. `evaluation_firewall_authority`: preregistered selection-validation/final-test identities plus training-data exclusion and decontamination authority before optimizer step 1.
9. `accelerator_runtime_authority`: exact CUDA/PyTorch/NCCL topology with finite forward/backward/update, measured memory/throughput and no silent distributed-objective drift.
10. `compute_authorization`: explicit owner authorization for materially paid compute, bound to an immutable authority object.

The first nine are prerequisites for even requesting a material-compute authorization. The tenth is the separate authorization gate.

## Why the three scientific gates are separate

### Data pipeline is not data budget

R01 policy PR #568 separates four quantities that must not be conflated: source-capacity bytes, unique post-tokenization corpus tokens, unique non-ignored causal-loss positions, and total training-token exposure after any permitted replay. The 20x / 50x / 100x ladder is an **exposure-budget planning ladder**. For a nominal 1B target those planning points are approximately 20B / 50B / 100B total training-token exposures; they are not requirements for 20B / 50B / 100B unique loss positions, are not hard minima or maxima, and are not quality guarantees.

The stage authority must therefore bind the immutable unique-data/loss ledger separately from the selected exposure budget and replay/epoch cap. Data-constrained scaling evidence makes this distinction operationally important because repeated exposure has diminishing value and cannot be relabeled as new unique data. A working pipeline or a source-byte total cannot stand in for either unique-data evidence or the preregistered exposure plan.

The runtime evaluator in stacked PR #611 is the intended kind of machine evidence this gate should eventually consume. Meeting an exposure tier still does not authorize training or paid compute by itself.

### Preceding-stage success is not optimizer transfer authority

The project currently uses ordinary model parameterization. Literature on Maximal Update Parametrization (`muP`) demonstrates that cross-scale hyperparameter transfer is a property obtained by deliberate parameterization and validation; it must not be assumed for a standard parameterization. Until this project adopts and validates such a transfer scheme, the 1B campaign needs scale-specific evidence for LR, warmup, optimizer moments, precision, clipping/update ratios and stopping rules.

The live learned-20M/100M control issue #548 already requires the launch packet to bind optimizer/scheduler/precision recipe and budget/stop rules. SCALE-06 now makes the same requirement explicit at 1B.

### Evaluation must exist before the run

A large run must not begin first and decide later what checkpoint, validation split or final test counts. The 1B gate therefore requires an immutable evaluation/decontamination authority before authorization readiness. This composes with the repository's existing checkpoint-selection and contamination controls rather than creating a second evaluator.

## Runtime architecture direction

The project should continue with native PyTorch composable distributed primitives rather than introduce a second framework solely for 1B. The existing SCALE-05 path already uses meta initialization, FSDP2 and activation checkpointing. TorchTitan demonstrates the same family of mechanisms at larger scale: FSDP2, tensor/pipeline/context parallelism, meta initialization, activation checkpointing and distributed checkpointing. Treat it as an upstream design reference, not a source of pretrained weights.

The current canonical model still repeats K/V heads for GQA before SDPA. Maintained PyTorch exposes `scaled_dot_product_attention(..., enable_gqa=True)`, so the open PERF-21 integration is the correct immediate attention dependency rather than accepting the repeated-K/V path at 1B.

## Data/compute scaling boundary

Parameter count is not the target by itself. Compute-optimal scaling shows that model and data scale together; data-constrained studies show that repeated data has diminishing value; deployment-aware and over-training studies show that useful token/parameter ratios can exceed the simple compute-optimal point. The correct project action is therefore to measure scaling curves and freeze both the unique-data requirements and the total-exposure/replay policy explicitly, not to equate parameter count, source bytes, unique positions or replayed exposures with one another.

## Allocation-safe validation

With no authority references, the report must remain blocked:

```bash
python tools/assess_scale_1b_dependencies.py --meta-probe
```

To compose evidence, pass exact durable references from the owning lanes, for example:

```bash
python tools/assess_scale_1b_dependencies.py \
  --stage-data-budget-authority github:r01-1b-data-budget@<40hex>:qualified \
  --training-recipe-authority github:r01-1b-training-recipe@<40hex>:qualified \
  --evaluation-firewall-authority github:eval-1b-firewall@<40hex>:qualified
```

These flags alone are not sufficient; all other engineering authorities must also be supplied. Compute remains blocked unless `--compute-authorization` contains a separately granted, immutable owner authority with the required prefix, for example `COMPUTE_AUTHORIZED:artifact:owner-compute-approval@<64hex>`.

The CLI records the supplied authority map in its JSON output so a later reviewer can see exactly what evidence was used to clear each gate.

Resource estimates are analytical planning values under the current project semantics. They are not measured CUDA peaks, throughput, MFU, checkpoint pause time or budget quotes.

## External primary references

- PyTorch scaled dot product attention / `enable_gqa`: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- TorchTitan: https://github.com/pytorch/torchtitan
- Compute-optimal scaling, Hoffmann et al. 2022: https://arxiv.org/abs/2203.15556
- Data-constrained scaling, Muennighoff et al. 2023: https://arxiv.org/abs/2305.16264
- muTransfer / Maximal Update Parametrization, Yang et al. 2022: https://arxiv.org/abs/2203.03466
- Unit-scaled muP, Blake et al. 2024: https://arxiv.org/abs/2407.17465
- Over-training scaling, Gadre et al. 2024: https://arxiv.org/abs/2403.08540
- Llama 3 scaling discussion: https://ai.meta.com/blog/meta-llama-3/
