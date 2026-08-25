# D12 distributed training and scale-systems contract

Status: **EXPERIMENTAL / LOCAL_FREE**. This package does not authorize paid compute, promote an S0 candidate, or change Base behavior.

Authority base: `d08/dependency-lock-packaging-20260824@3d5d2332577d1ccb2b6ecbb5197b1d95a4baba6f`; exact-head CI run `32742220948` completed successfully.

## D08 topology audit

The inherited D08 contract is internally consistent for the current 12-6 project profile:

- physical world size = `DP * TP * PP * CP`;
- project EP is a subgroup inside DP and does not add a physical rank dimension;
- `EP` divides `DP`; `EDP = DP / EP`;
- generic dense memory estimation rejects `EP > 1`;
- S0 remains `DP=TP=PP=CP=EP=1`.

D12 adds a deterministic fake-rank mapping with TP as the smallest stride, then CP, DP and PP. EP/EDP coordinates are derived from the DP rank. Fake groups cover DP, TP, PP, CP, EP and EDP without initializing `torch.distributed`.

### Current Megatron compatibility finding

The project EP-as-DP-subgroup profile must **not** be described as a universal current Megatron Core invariant. Current Megatron Core documentation exposes EP as a parallel dimension, documents total GPU count as `TP * PP * CP * EP * DP`, and newer MoE Parallel Folding can decouple expert-side and attention-side data distributions. Current `core.parallel_state` also supports explicit `tp-cp-ep-dp-pp` ordering.

Therefore a future Megatron adapter needs its own topology translation. It may not silently pass this project's `ParallelPlan.data_parallel` and `expert_parallel` into Megatron flags and assume identical world-size algebra. The project profile remains useful for dense S0 and a restricted EP-inside-DP compatibility mode; folded/orthogonal EP needs a richer topology object.

Primary references:
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html

## CPU/fake topology contract

`mesh.py` defines logical coordinates and process groups for data, tensor, pipeline, context, expert-subgroup and expert-data-parallel axes. This validates topology algebra without GPU/NCCL/paid compute.

Rank identity is a logical coordinate such as `pp=0/dp=3/cp=1/tp=0/ep=1/edp=1`. Hostnames, PIDs, CUDA local ranks and scheduler task IDs are execution facts, not checkpoint identity.

## PyTorch-native seam without S0 infection

`runtime.py` imports no Torch at module import time. `build_torch_native_plan()` is pure Python and keeps tiny S0 free of distributed initialization. The explicit capability probe lazily checks:

- FSDP2 `torch.distributed.fsdp.fully_shard`;
- `DTensor`;
- tensor-parallel `parallelize_module`;
- PyTorch Distributed Checkpoint save/load.

FSDP2 is the preferred first maintained-library state-sharding seam because current PyTorch documents it as DTensor-based per-parameter sharding. Higher-level distributed checkpoint APIs are the appropriate path for full-state or resharded checkpoint workflows.

Primary references:
- https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- https://docs.pytorch.org/docs/stable/distributed.tensor.html
- https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
- https://docs.pytorch.org/docs/stable/distributed.checkpoint.html

No function in this package calls `fully_shard()`, `parallelize_module()`, or initializes `torch.distributed`. Canonical model/trainer wrapping remains a later explicit runtime integration.

## D05 identity to distributed-checkpoint identity

D05 checkpoint v1 computes `checkpoint_id = SHA256(canonical_json({"identity": identity_record, "files": files}))`. The D05 checkpoint ID therefore binds both logical training lineage and the exact unsharded artifact set. A reshard changes physical files and cannot truthfully retain the same artifact-bound ID.

D12 layers a layout identity on top:

1. verify source D05 format/version and recompute its checkpoint ID;
2. compute `d05.identity_sha256 = SHA256(canonical_json(D05 identity record))`;
3. preserve both the source D05 checkpoint ID and logical identity SHA;
4. bind saved topology, backend format, reshardability flags and every shard record;
5. compute order-independent `artifact_set_sha256` from sorted shard records;
6. compute a topology-specific `layout_id`.

Two identities now have distinct purposes:

- **logical training identity**: D05 identity SHA-256, preserved across legitimate reshard;
- **physical layout identity**: D12 layout ID, changed when topology, rank ownership, backend format or shard bytes change.

A future distributed writer should use this as a sidecar contract or introduce a reviewed D05 format version with equivalent semantics; it should not mutate D05 v1 semantics in place.

## Reshard, resume and checksum rules

Direct resume is valid when saved and target topology snapshots are identical. Topology-changing resume is valid only when:

- the backend format explicitly supports resharding;
- optimizer state is also reshardable when optimizer resume is required;
- the D05 logical identity is preserved;
- target topology separately passes world-size/divisibility validation;
- rank-local artifacts receive new checksums and a new layout ID after the next save.

Rank-local filenames are not semantic identities. After resharding, writer rank 3 may own different fragments than writer rank 3 in the source job.

Current primary evidence:
- TorchTitan documents DCP resharding, including a one-CPU seed checkpoint loadable with arbitrary GPU count.
- Megatron Core `torch_dist` supports topology-changing loads and distinguishes optimizer formats with different reshardability.
- OLMo-core exposes maintained distributed checkpoint infrastructure and current checkpoint maintenance.

References:
- https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md
- https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html
- https://github.com/allenai/OLMo-core/blob/main/AGENTS.md
- https://github.com/allenai/OLMo-core/blob/main/CHANGELOG.md

## Memory-estimator algebra and local measurement

The inherited estimator is first-order planning only. State arithmetic is `ceil(total_parameters / state_shard_factor) * bytes_per_state`; activation arithmetic assumes ideal division by `TP * PP * CP`.

D12 locally measured a real CPU `torch.nn.Linear(3, 2, bias=True)`:

- 8 fp32 trainable parameters;
- 32 materialized parameter bytes;
- 32 materialized gradient bytes after backward.

With estimator coefficients 4 bytes/parameter, 4 bytes/gradient and all other terms zero, the estimator returns the same 64 total bytes. This validates coefficient arithmetic only.

It does **not** validate transformer capacity predictions. The estimator still omits allocator fragmentation, communication buckets, all-gather buffers, optimizer implementation details, kernel workspaces, checkpoint staging, pipeline imbalance, embedding/output imbalance, recomputation policy and backend overhead. PP/TP state division remains idealized because `ModelScaleSpec.total_parameters` lacks per-layer/per-tensor decomposition.

Before materially paid compute, replace planning coefficients with measurements from the exact candidate SHA and exact hardware.

## Float8 and observability

Float8 is not an S0 requirement and is not enabled here. TorchTitan's current guidance targets GPUs with Float8 tensor cores and notes that benefit depends on GEMM dimensions being large enough to amortize dynamic quantization overhead. The trigger is measured kernel shape/performance on candidate hardware, not stage number alone.

TorchTitan already exposes memory snapshots, CPU/GPU profiling, Flight Recorder and structured logging. Prefer maintained seams before writing project-specific profiler infrastructure. Megatron Core adds mature communication overlap, distributed optimizer and FP8 paths when GPU-scale measurements justify the integration cost.

References:
- https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/quantization/float8.md
- https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md
- https://github.com/pytorch/torchtitan
- https://docs.nvidia.com/megatron-core/developer-guide/latest/

## Stage-triggered backend adoption

Backend choice is trigger-based, not a hard parameter-count ladder.

### Stay custom / single-device

Keep the existing 12-6 model, tokenizer, trainer and D05 path while one device fits and measured throughput is acceptable. This is mandatory for S0 and remains the default for later small stages until measurements show a reason to distribute.

### PyTorch native FSDP2 / DTensor / TP / DCP

Adopt first when a real multi-device dense run is needed but the project should keep its own model/trainer loop. Triggers include model/optimizer state no longer fitting one device, a specific layer width requiring TP, or topology-independent DCP resharding. Do not introduce PP/CP merely because APIs exist.

### TorchTitan

Evaluate when multiple PyTorch-native scale features must be composed and local glue becomes the larger engineering risk: FSDP2 plus TP/PP/CP, async DCP, MFU/throughput/memory profiling, Flight Recorder, or Float8 experiments on eligible hardware. Model registration must preserve 12-6 random-init lineage and must not import foreign pretrained weights.

### OLMo-core

Evaluate when its integrated training/data/checkpoint infrastructure is a better fit than maintaining equivalent orchestration locally. Current OLMo-core exposes combinable FSDP/HSDP/DDP, TP, PP, CP and EP implementations plus distributed checkpointing. Adoption requires an interface audit against 12-6 tokenizer/data/checkpoint identities.

### Megatron Core

Evaluate when measured large-GPU performance needs justify heavier integration: aggressive TP, PP for deep models, CP for long contexts, real MoE EP/EDP/folding, distributed optimizer, communication overlap, Transformer Engine or FP8. A dedicated Megatron topology adapter is mandatory because current Megatron EP algebra is richer than the project's EP-as-DP-subgroup `ParallelPlan`.

## Evidence boundary

This package does not prove NCCL behavior, real multiprocess collectives, FSDP2 wrapping of D01, TP/PP/CP canonical training correctness, MoE runtime, GPU memory capacity, throughput/MFU, Float8 numerical equivalence, distributed D05 I/O, or node-loss recovery. Those remain future evidence gates. No CANDIDATE/STABLE or independent audit PASS follows from this package.
