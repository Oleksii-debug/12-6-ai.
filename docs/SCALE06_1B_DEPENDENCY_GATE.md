# SCALE-06 ~1B dependency gate

## Decision boundary

The repository already contains an S6 engineering candidate at 999,761,920 random-init parameters. That fact alone is not evidence that a 1B run should start.

`scale_1b_readiness.py` adds a fail-closed evidence layer. Every material dependency is absent by default. A gate clears only when the caller binds it to a non-empty durable authority reference from the owning lane. Bare `--qualified` booleans are intentionally not accepted because they would allow readiness to be self-attested without terminal evidence.

The report separates engineering readiness from compute authorization:

- `ready_for_authorization_request` becomes true only after all engineering dependencies carry authority references;
- `ready_for_material_compute` additionally requires an explicit authorization reference beginning with `COMPUTE_AUTHORIZED:` or `TRAINING_AUTHORIZED:`.

No stage promotion or compute authority is created by this module.

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

## Required positive evidence

Each engineering input below must be a durable authority reference such as an exact GitHub head with terminal CI evidence or an immutable evidence artifact identity:

1. `preceding_stage_authority`: preceding scale stage admitted by its own evidence/promotion gate.
2. `production_tokenizer_authority`: versioned tokenizer frozen from a decontaminated representative corpus with fertility/coverage/BPB evidence.
3. `native_gqa_authority`: 32Q/8KV path integrated without materializing repeated K/V heads, with target-runtime numerical and accelerator evidence.
4. `distributed_checkpoint_authority`: DCP/FSDP2 save/load/resume and reshard composed on the target model/runtime with integrity evidence.
5. `data_pipeline_authority`: representative corpus, decontamination, deduplication, split, packing, unique-loss accounting and held-out boundaries terminally bound.
6. `accelerator_runtime_authority`: exact CUDA/PyTorch/NCCL topology with finite forward/backward/update, measured memory/throughput and no silent distributed-objective drift.
7. `compute_authorization`: explicit owner authorization for materially paid compute, prefixed `COMPUTE_AUTHORIZED:` or `TRAINING_AUTHORIZED:`.

The first six are engineering gates. The seventh is an authorization gate and cannot be inferred from green tests or from the presence of an S6 candidate.

## Runtime architecture direction

The project should continue with native PyTorch composable distributed primitives rather than introduce a second framework solely for 1B. The existing SCALE-05 path already uses meta initialization, FSDP2 and activation checkpointing. TorchTitan demonstrates the same family of mechanisms at larger scale: FSDP2, tensor/pipeline/context parallelism, meta initialization, activation checkpointing and distributed checkpointing. Treat it as an upstream design reference, not a source of pretrained weights.

The current canonical model still repeats K/V heads for GQA before SDPA. Maintained PyTorch exposes `scaled_dot_product_attention(..., enable_gqa=True)`, so the open PERF-21 integration is the correct immediate attention dependency rather than accepting the repeated-K/V path at 1B.

## Data/compute scaling boundary

Parameter count is not the target by itself. Scaling-law evidence shows that increasing model size without scaling training data wastes compute. The final S6 token budget must therefore be selected from measured held-out curves and the actual tokenizer's tokenization behavior. Raw byte-token counts must not be equated mechanically with BPE-token budgets.

## Allocation-safe validation

With no authority references, the report must remain blocked:

```bash
python tools/assess_scale_1b_dependencies.py --meta-probe
```

To compose evidence, pass the exact durable references from the owning lanes, for example `--native-gqa-authority <exact-authority>` rather than a boolean readiness flag. Compute remains blocked unless `--compute-authorization` contains a separately granted authorization reference with the required prefix.

The CLI records the supplied authority map in its JSON output so a later reviewer can see exactly what evidence was used to clear each gate. Supplying an authority string does not itself validate another lane's CI; the coordinator must only compose terminal evidence.

Resource estimates are analytical planning values under the current project semantics. They are not measured CUDA peaks, throughput, MFU, checkpoint pause time or budget quotes.

## External primary references

- PyTorch scaled dot product attention / `enable_gqa`: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- TorchTitan: https://github.com/pytorch/torchtitan
- Compute-optimal scaling (Hoffmann et al., 2022): https://arxiv.org/abs/2203.15556
