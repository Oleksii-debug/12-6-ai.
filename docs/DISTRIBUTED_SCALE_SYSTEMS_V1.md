# Distributed Scale Systems v1

Status: EXPERIMENTAL scale-readiness design and LOCAL_FREE validation only. This document does not authorize paid compute, promote an S0 candidate, or replace AUDIT-A/AUDIT-B.

Authority base for this package is PR #58, branch `d08/dependency-lock-packaging-20260824`, exact SHA `3d5d2332577d1ccb2b6ecbb5197b1d95a4baba6f`. PR #58 exact-head CI `32742220948` is recorded by the active integration lane as completed SUCCESS. The underlying D08 distributed source is `b8e01a38bfba18176e36c34b1530839ccaea4b21` / PR #20 / CI `32646152647` SUCCESS.

## 1. Existing D08 topology contract and naming boundary

The existing project contract intentionally defines physical world size as:

`project_world = project_DP_total * TP * PP * CP`

`EP` is a subgroup of `project_DP_total`, so:

`expert_DP = project_DP_total / EP`

This is internally consistent but the word `DP` is not identical to the current Megatron Core CLI/grid definition. Current Megatron rank generation uses separate `EP` and `DP` axes. Translation is therefore:

`Megatron_EP = project_EP`

`Megatron_DP = project_DP_total / project_EP`

and:

`Megatron_world = TP * PP * CP * EP * Megatron_DP = project_world`

This translation prevents double-counting EP while preserving D08's existing API. The project-local `RankLayout` is an identity contract, not a claim that rank numbers equal TorchTitan, OLMo-core, or Megatron rank numbers.

The canonical project-local physical rank axis order is `dp, pp, cp, tp`. The layout records a SHA-256 identity over axis order and the complete `ParallelPlan`. EP and expert-DP coordinates are deterministically decomposed inside the project DP axis.

CP partitions activations/sequence while shared weights remain duplicated. `RankLayout.dense_gradient_sync_group()` therefore spans project DP and CP while holding TP/PP coordinates fixed. Backend adapters may realize that synchronization differently.

## 2. LOCAL_FREE topology validation

Two layers are required before any paid or multi-node run:

1. Fake topology algebra must cover rank<->coordinate round trips, DP/TP/PP/CP groups, EP groups, expert-DP groups, divisibility, world size and backend translation without initializing a process group.
2. A bounded CPU/Gloo probe must exercise real `torch.distributed` initialization and a collective. `run_cpu_gloo_probe()` is capped at world size 8 and never uses GPU/cloud resources.

The committed test uses four local Gloo ranks with project `DP=2, TP=2` and verifies all ranks agree on logical layout identity and on an all-reduced rank sum.

## 3. FSDP2 / DTensor / TP seam

Single-device S0 remains unaffected. Importing `twelve_six.distributed` does not initialize `torch.distributed`, construct a `DeviceMesh`, or wrap the model.

`build_torch_mesh_spec()` translates the existing `ParallelPlan` into a lazy five-dimensional mesh contract:

`dp_replicate, dp_shard, cp, pp, tp`

The two DP dimensions multiply back to existing `project_DP_total`. `fsdp_shard_degree` must divide project DP. This supports an HSDP-style split without changing the current public `ParallelPlan`.

`TorchMeshSpec.create_device_mesh()` is deliberately lazy. The caller must initialize the process group. `TorchMeshSpec.fsdp2_kwargs()` returns arguments for `torch.distributed.fsdp.fully_shard`; it never mutates a model itself. TP/PP/CP helpers only select their mesh dimensions.

Generic native FSDP2 binding fails closed for `EP>1`. Dense/shared parameters and expert parameters do not have the same replication/sharding domain in MoE. A MoE-aware backend must supply explicit expert groups instead of applying a dense generic wrapper to the whole model.

Current PyTorch primary documentation states that FSDP2 shards parameters, gradients and optimizer states, represents sharded parameters as DTensors, and can consume a full SPMD mesh with named data-parallel shard/replicate dimensions. Current DCP state-dict APIs normalize FQNs across FSDP2/DDP/TP combinations and can be passed directly to DCP.

Primary references, checked 2026-08-24:

- https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- https://docs.pytorch.org/docs/main/distributed.checkpoint.html
- https://docs.pytorch.org/docs/stable/distributed.tensor.html
- https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
- https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

## 4. D05 identity -> distributed checkpoint contract

D05 checkpoint v1 already binds exact Git SHA, ModelSpec hash, tokenizer hashes, dataset manifest, run manifest, training config, optimizer/scheduler identity, step/tokens, environment and optional environment-lock hash. D05 `checkpoint_id` additionally hashes its physical SafeTensors/JSON files.

Distributed resharding cannot use physical shard bytes as the semantic training-state identity because physical bytes and filenames can change when topology changes. The D12 envelope therefore separates:

- `D05CheckpointRef.identity_sha256`: semantic parent identity from the verified D05 manifest identity record;
- `D05CheckpointRef.checkpoint_id` and `manifest_sha256`: migration/source artifact proof;
- `state_dict_schema_sha256`: canonical global state/FQN schema expected by the distributed loader;
- `save_layout_sha256` and `save_world_size`: topology at save time;
- one SHA-256/size/writer-rank record per physical DCP file;
- one RNG digest per logical save rank;
- `artifact_set_sha256`: sorted topology-dependent physical shard identity;
- `envelope_sha256`: aggregate binding of semantic parent, topology, state schema, RNG digests and physical artifact set.

A DCP implementation must obtain model/optimizer state through PyTorch distributed state-dict APIs rather than serialize rank-local parameter IDs as canonical identity. PyTorch's current DCP tutorial explicitly supports loading with a different world size by allocating the target sharded state first and loading into that target topology.

## 5. Resume and reshard semantics

Two resume modes are distinct and must never be conflated.

`EXACT_TOPOLOGY` requires the same logical rank-layout hash, same world size and same canonical state-dict schema. Only this path may restore rank-local RNG by logical rank and claim exact-trajectory eligibility, subject to all other determinism requirements.

`RESHARD` may change world size/topology when the state-dict schema is unchanged and the selected backend/optimizer checkpoint format supports load-time resharding. Model/optimizer/trainer counters may be resumed, but changed rank cardinality has no one-to-one mapping for rank-local RNG streams. The default policy is therefore deterministic reseeding of new rank streams from stable global seed/step inputs and an explicit **no bitwise trajectory claim**.

A changed canonical FQN/state schema fails closed in both modes. Offline checkpoint conversion, if introduced later, must create a new attested artifact rather than silently rewriting identity.

Megatron Core's current distributed checkpoint documentation similarly distinguishes a fast `dp_reshardable` optimizer format from a slower `fully_reshardable` format that supports arbitrary model-parallel changes. That distinction is evidence that optimizer reshardability must be an explicit capability, not inferred from model-weight reshardability.

Primary references:

- https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html

## 6. Memory estimator algebra and measurements

`estimate_training_memory()` remains a first-order planning estimator. It is not allocator telemetry and cannot justify a paid run by itself.

The D12 measurement seam computes exact bytes from local tensor `numel() * element_size()` while deduplicating identical tensor objects. The LOCAL_FREE test builds a 40-parameter fp32 model, measures 160 parameter bytes and checks that the estimator returns exactly 160 bytes when all non-parameter coefficients are set to zero.

Before any GPU capacity claim, add measured evidence for:

- allocated/reserved device memory at steady state;
- peak activation memory at the real sequence/microbatch size;
- optimizer-state bytes after at least one optimizer step;
- FSDP2/TP shard-local parameter bytes;
- communication buffers and fragmentation;
- activation checkpointing/recompute effects.

The existing generic estimator continues to reject EP>1 because total parameter count alone cannot separate dense and expert parameter populations.

## 7. Float8 adoption seam

Float8 is **not** enabled for S0. It becomes eligible only after a bf16/fp32 baseline is numerically stable, the target accelerator supports the recipe, GEMMs are large enough for quantization overhead to amortize, and measured profiling shows matmul throughput is a limiting factor.

Current torchao 0.17 documentation marks float8 training for `torch.nn.Linear` stable and reports large-scale throughput gains, while MXFP8/MoE variants remain prototype. TorchTitan's current float8 guidance targets H100-class hardware, recommends selecting only profitable linear layers, and requires `torch.compile` for competitive performance. These are reasons to gate float8 behind measurement, not to enable it speculatively.

Primary references, checked 2026-08-24:

- https://docs.pytorch.org/ao/stable/workflows/training.html
- https://docs.pytorch.org/ao/stable/api_reference/api_ref_float8.html
- https://docs.pytorch.org/ao/stable/eager_tutorials/pretraining.html
- https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/quantization/float8.md

## 8. Large-scale debugging/profiling seam

Do not treat a distributed timeout as a generic training failure. Capture enough rank-scoped evidence to classify collective mismatch, straggler, OOM, numerical failure or process failure.

For later GPU stages, the preferred seam is:

- per-rank structured logs with run/candidate/layout identities;
- PyTorch profiler traces around selected warm/active windows, not continuous full-run tracing;
- allocator memory snapshots around OOM or selected iterations;
- communication timeout dumps / Flight Recorder where supported;
- periodic throughput, tokens/sec, TFLOPs/MFU and peak memory metrics;
- explicit deterministic/debug modes separate from production throughput mode;
- artifact retention tied to exact source SHA and run manifest.

TorchTitan currently exposes CPU/GPU profiling, memory snapshots, Flight Recorder, deterministic/debug controls, fake communication and `local_tensor` simulation for multi-dimensional numerics. These are evaluation targets for later adoption; D12 does not vendor or depend on TorchTitan today.

Primary references:

- https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md
- https://github.com/pytorch/torchtitan/blob/main/torchtitan/config/configs.py
- https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py

## 9. Backend adoption gates

### Gate 0 — current S0 / single process

Stay custom and single-device. Use D12 contracts/tests only. No FSDP2, TP, PP, CP, EP, DCP or float8 runtime is required.

### Gate 1 — first dense multi-GPU need

Adopt native PyTorch first: DeviceMesh/DTensor + FSDP2 and DCP, adding TP only when layer memory or measured throughput requires it. This minimizes framework lock-in and preserves existing 12-6 model/trainer semantics.

### Gate 2 — repeated 2D/3D composition and operational complexity

Evaluate TorchTitan against the native adapter when the project repeatedly needs maintained DP/TP/PP/CP composition, DCP integration, profiler/Flight Recorder configuration, compile and float8 seams. Current TorchTitan is actively moving toward full-DTensor/config-based sharding, so integration must be pinned/tested rather than assumed stable.

Current reference:

- https://docs.pytorch.org/devlogs/distributed/2026-08-17-config-based-sharding-full-dtensor/

### Gate 3 — integrated research trainer/data/checkpoint stack becomes valuable

Evaluate OLMo-core when its trainer/data/callback abstractions and composable FSDP/HSDP/DDP, TP, PP, CP, EP and checkpoint components remove more code than they add. Keep 12-6 model/checkpoint identity above the framework. Current main documentation describes these components as composable; the changelog records current CP/TP eval and checkpoint/debug work.

References:

- https://github.com/allenai/OLMo-core/blob/main/AGENTS.md
- https://github.com/allenai/OLMo-core/blob/main/CHANGELOG.md

### Gate 4 — deep NVIDIA scale / PP-CP-EP-MoE / maximum throughput

Evaluate Megatron Core when the target is NVIDIA-only large scale and requires mature PP/CP/EP/MoE, distributed optimizer, Transformer Engine/FP8 and scale-specific communication overlap. Its current guide recommends starting with DP and adding TP/PP/CP only as model/layer/depth/context constraints demand.

References:

- https://docs.nvidia.com/megatron-core/developer-guide/latest/
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html

### Backend selection rule

Do not maintain four production backends indefinitely. Keep the project semantic contracts framework-neutral, benchmark candidate backends on the same exact model/data/run identity, then select one canonical large-scale execution backend for each materially different scale regime. Any backend change must prove checkpoint migration/reshard behavior and numerical equivalence before it becomes canonical.

## 10. Truth boundary and next execution

This package proves topology/checkpoint/runtime contracts plus LOCAL_FREE CPU evidence only. It does not prove GPU throughput, MFU, NCCL stability, multi-node behavior, FSDP2 training correctness, TP/PP/CP/EP model numerics, float8 convergence, or paid-run readiness.

The next runtime milestone after the current S0 composition is stable is a tiny exact-head two/four-rank native PyTorch experiment using a real 12-6 model, FSDP2/DCP behind these adapters, save->reload on the same topology, then save->reshard-load on a different LOCAL_FREE topology where practical. Only after that should framework bakeoffs or GPU profiling be considered.
