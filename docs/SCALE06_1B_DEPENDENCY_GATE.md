# SCALE-06 ~1B dependency gate

## Decision boundary

The repository already contains an S6 engineering candidate at 999,761,920 random-init parameters. That fact alone is not evidence that a 1B run should start.

`scale_1b_readiness.py` adds a fail-closed evidence layer. Every material dependency is `False` by default and must be supplied positively by its owning lane. The report separates engineering readiness from compute authorization:

- `ready_for_authorization_request` becomes true only after all engineering dependencies are qualified;
- `ready_for_material_compute` additionally requires explicit compute authorization.

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

1. `preceding_stage_admitted`: the preceding scale stage has passed its own evidence/promotion gate.
2. `production_tokenizer_qualified`: a versioned tokenizer is frozen from a decontaminated representative corpus with fertility/coverage/BPB evidence.
3. `native_gqa_qualified`: the 32Q/8KV path is integrated without materializing repeated K/V heads and has target-runtime numerical + accelerator evidence.
4. `distributed_checkpoint_qualified`: DCP/FSDP2 save/load/resume and reshard are composed on the target model/runtime with integrity evidence.
5. `data_pipeline_qualified`: representative corpus, decontamination, deduplication, split, packing, unique-loss accounting and held-out boundaries are terminally bound.
6. `accelerator_runtime_qualified`: exact CUDA/PyTorch/NCCL topology has finite forward/backward/update, measured memory and throughput, and no silent distributed objective drift.
7. `compute_authorized`: explicit owner authorization for materially paid compute.

The first six are engineering gates. The seventh is an authorization gate and cannot be inferred from green tests.

## Runtime architecture direction

The project should continue with native PyTorch composable distributed primitives rather than introduce a second framework solely for 1B. The existing SCALE-05 path already uses meta initialization, FSDP2 and activation checkpointing. TorchTitan demonstrates the same family of mechanisms at larger scale: FSDP2, tensor/pipeline/context parallelism, meta initialization, activation checkpointing and distributed checkpointing. Treat it as an upstream design reference, not a source of pretrained weights.

The current canonical model still repeats K/V heads for GQA before SDPA. Maintained PyTorch exposes `scaled_dot_product_attention(..., enable_gqa=True)`, so the open PERF-21 integration is the correct immediate attention dependency rather than accepting the repeated-K/V path at 1B.

## Data/compute scaling boundary

Parameter count is not the target by itself. Scaling-law evidence shows that increasing model size without scaling training data wastes compute. The final S6 token budget must therefore be selected from measured held-out curves and the actual tokenizer's tokenization behavior. Raw byte-token counts must not be equated mechanically with BPE-token budgets.

## Allocation-safe local validation

Run:

```bash
python tools/assess_scale_1b_dependencies.py --meta-probe
```

With no evidence flags, the report must remain blocked. Individual flags are intended only for composing already-terminal evidence from the owning lanes; they are not self-attestations.

Resource estimates are analytical planning values under the current project semantics. They are not measured CUDA peaks, throughput, MFU, checkpoint pause time or budget quotes.

## External primary references

- PyTorch scaled dot product attention / `enable_gqa`: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- TorchTitan: https://github.com/pytorch/torchtitan
- Compute-optimal scaling (Hoffmann et al., 2022): https://arxiv.org/abs/2203.15556
