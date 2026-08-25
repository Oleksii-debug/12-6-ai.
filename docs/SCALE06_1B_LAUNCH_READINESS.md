# SCALE-06: S6 ~1B launch readiness

Status: allocation-safe engineering readiness package; not a stage freeze, launch authorization, promotion, capability claim, or audit verdict.

## Lineage and ownership

SCALE-06 is stacked on SCALE-05 PR #164 and reuses its scale-runtime seam instead of forking it. The S6 geometry is a selective intake of the existing D01 PR #37 non-frozen candidate. Tensor-parallel work remains owned by D12 PR #151, distributed/checkpoint persistence remains owned by the active D05/D12 lineages, and paid-compute planning remains owned by D13 PR #70.

This package does not edit `model.py`, the base `Trainer`, SCALE-05 `scale_runtime.py`, TP implementation files, checkpoint core, distributed core, tokenizer/data code, dependency locks, or compute-plan files.

## Exact S6 engineering candidate

The candidate is the existing D01 S6 identity, carried onto the SCALE-05 execution lineage without changing its architecture:

- target: 1,000,000,000 parameters;
- exact trainable parameters: 999,106,560 (-0.089344% from target);
- vocabulary geometry budget: 32,768, tied embedding/output;
- context: 4,096;
- residual width: 2,048;
- blocks: 18;
- query/KV heads: 32/8 GQA;
- head dimension: 64;
- SwiGLU width: 6,720;
- pre-RMSNorm, RoPE, no attention/MLP/output bias or dropout;
- ModelSpec: `cc64cbe94a461c364f063652098e55bdcf640d4be756ee1e743a23dda3de7261`;
- InitSpec: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`;
- Base: first-party random initialization only.

The 32K vocabulary is only an architecture cost budget. No S6 tokenizer or corpus is frozen.

## Allocation-safe real architecture execution

`src/twelve_six/training/s6_readiness.py` constructs the complete 999,106,560-parameter `TwelveSixDecoder` on the PyTorch `meta` device through the inherited SCALE-05 `build_meta_decoder` seam. The workflow requires the actual module parameter total to equal the ModelSpec formula exactly and requires every parameter to remain on `meta`, so the full architecture is instantiated and structurally validated without allocating roughly 4 GB of fp32 weight storage.

A separate bounded LOCAL_FREE CPU analogue executes the inherited `ActivationCheckpointedDecoder` and `ExternallyPlacedTrainer` path with gradient accumulation. It requires a real finite forward/loss/backward/gradient-norm path, a committed AdamW optimizer update, checkpoint-safe Trainer state, and a non-zero weight delta. This is execution-seam evidence only; it is not a proxy for 1B convergence or accelerator performance.

## Current-runtime resource algebra

The estimates deliberately match SCALE-05 semantics: fp32 persistent parameters, fp32 gradients, two fp32 AdamW moments, bf16/fp16 saved-activation planning, blockwise activation checkpointing, and current expanded-GQA attention intermediates.

At sequence length 4,096 and microbatch 1:

| Quantity | Exact bytes | Approx. GiB |
| --- | ---: | ---: |
| FP32 parameters | 3,996,426,240 | 3.722 |
| FP32 gradients | 3,996,426,240 | 3.722 |
| two FP32 Adam moments | 7,992,852,480 | 7.444 |
| replicated persistent total | 15,985,704,960 | 14.888 |
| FSDP2 persistent / rank, world 4 | 3,996,426,240 | 3.722 |
| FSDP2 persistent / rank, world 8 | 1,998,213,120 | 1.861 |
| weight-only checkpoint | 3,996,426,240 | 3.722 |
| model + two Adam moments checkpoint | 11,989,278,720 | 11.166 |
| checkpointed activation lower-bound, microbatch 1 | 797,966,336 | 0.743 |

Unexpanded bf16/fp16 K/V cache is 36,864 bytes per token per sequence, or 150,994,944 bytes / 144 MiB at the full 4,096-token context.

The checkpoint-recompute-aware planning estimate is 10,408,771,584 FLOP/token. The prepared 64-update pilot contains 4,194,304 optimized tokens, therefore its first-order estimate is 43,657,552,289,857,536 FLOPs (~43.66 PFLOP).

These values are resource algebra, not measured CUDA peaks. They exclude or lower-bound material effects such as FSDP all-gather windows, CUDA allocator fragmentation, NCCL buffers, fused-kernel workspaces, temporary logits/loss storage beyond the estimator, DCP staging, dataloader/pinned-memory costs, and framework/runtime overhead. Therefore the arithmetic alone is not a GPU-fit claim.

## Prepared launch profile

`configs/runs/s6_1b.scale06_launch.json` is intentionally `PREPARED_NOT_LAUNCHED` and describes one conservative first accelerator qualification topology:

- one node, eight GPUs, world size 8;
- minimum 24 GiB CUDA memory per GPU;
- FSDP2 full-shard;
- tensor-parallel degree 1 in SCALE-06 v1;
- bf16 autocast with fp32 persistent state;
- sequence length 4,096, microbatch 1, gradient accumulation 16;
- 64 pilot optimizer updates / 4,194,304 optimized tokens;
- blockwise non-reentrant activation checkpointing;
- Flash SDPA required;
- AdamW fp32 moments.

D12 PR #151 already supplies an independently owned TP seam and proves that the historical 32Q/8KV S6 geometry is head-aligned for TP2/TP4/TP8. SCALE-06 deliberately does not claim FSDP2+TP composition because that combined runtime has not been validated on an exact accelerator head.

## Checkpoint boundary

The ordinary single-process checkpoint-v1 path is not accepted for a paid FSDP2 S6 launch. The launch profile requires `torch.distributed.checkpoint`-style sharded persistence integrated with the project-owned semantic identity and committed-step recovery contracts. SCALE-06 records that requirement but does not create a competing checkpoint format.

The full model + two Adam moments are about 11.166 GiB before metadata/scalars. A scale run must prove actual sharded save, verify, fresh-process reload, and committed-step resume with host/device memory telemetry before the checkpoint gate can become true.

## Data and tokenizer boundary

The current S0 byte tokenizer/data are useful controlled mechanics fixtures only. They are explicitly forbidden as evidence of S6 capability training readiness. A launch requires:

- a versioned first-party tokenizer artifact selected from measured fertility/coverage evidence;
- a versioned real-scale corpus manifest and data lineage;
- a held-out split whose tokens are proven to have zero optimization exposure;
- tokenizer/model-vocabulary compatibility bound into the run identity.

Until those exist, `real_scale_data_tokenizer_ready=false`.

## Five launch blockers

S6 remains fail-closed until every current launch gate is independently proven:

1. `preceding_s5_stage_pass`;
2. `real_scale_data_tokenizer_ready`;
3. `dcp_sharded_checkpoint_validated`;
4. `cuda_nccl_fsdp2_smoke_validated`;
5. `compute_authorized`.

All five are `false` in the checked-in launch profile. No code in this package can silently reinterpret them as launch approval.

## Compute boundary

No materially paid compute is authorized or launched. The package makes no CUDA/NCCL throughput, peak-memory, MFU, convergence, quality, capability, production serving, promotion, or audit claim. No foreign pretrained Base weights and no instruction/alignment/refusal/personality layer are introduced.

## Next system leap

The next legitimate step is not a cosmetic parameter increment. First close the five S6 blockers, especially real tokenizer/data, sharded checkpoint recovery, and an exact-head CUDA/NCCL/FSDP2 training smoke with measured memory and throughput. If those gates pass, the existing D01 ~3B S7 candidate can become the next allocation-safe systems target, with TP/FSDP composition and checkpoint/storage scaling driven by measured S6 coefficients rather than paper estimates.
