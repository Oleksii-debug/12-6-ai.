# D12 real FSDP2 model execution

## Incumbent and scope

This lane is stacked on D12 PR #76, which is itself stacked on the primary D12 runtime surface in PR #74. It does not replace the existing topology, rank-layout, checkpoint, or CPU/Gloo contracts. It adds the missing real-model execution seam.

The LOCAL_FREE execution target is the committed S1-shaped `configs/stages/s1_100k.json` ModelSpec: 107,856 random-initialized parameters. This is engineering evidence for distributed execution only. It is not S1 training-quality evidence, a scale promotion, or a corpus claim.

## Blockers fixed

1. Canonical 12-6 parameters are plain `Tensor` objects before FSDP2. The previous runtime passed the full five-dimensional mesh plus `DataParallelMeshDims`, a PyTorch path intended for parameters that are already DTensors on a full SPMD mesh. The canonical path now slices the existing mesh to the 1D FSDP or 2D HSDP data-parallel submesh. A separate `fsdp2_spmd_kwargs()` method retains the pre-existing-DTensor case.
2. The D02 Trainer computes and normalizes gradients by iterating ordinary local tensors. FSDP2 exposes sharded parameter gradients as DTensors. `FSDP2Trainer` is a D12-only adapter that keeps the D02 loop but uses PyTorch's DTensor-aware `torch.nn.utils.clip_grad_norm_` for the global norm after the existing token normalization.
3. FSDP2 is applied bottom-up to every `TransformerBlock` and then to the root `TwelveSixDecoder`. The optimizer is created only after `fully_shard`, so AdamW owns the DTensor parameters.
4. A failed FSDP forward can leave per-iteration sharding state undefined. The execution path exercises `FSDPModule.reset_iter_state()` with an over-context forward and proves that a valid forward succeeds afterward.

## LOCAL_FREE execution

The integration test and CLI execute all of the following on two local CPU processes with Gloo:

- default process group initialization;
- the existing D12 `ParallelPlan` and five-dimensional `DeviceMesh`;
- slicing to the FSDP data-parallel submesh;
- `fully_shard` on real 12-6 transformer blocks and the root model;
- DTensor parameter verification;
- deterministic synthetic token data;
- `DistributedSampler` with exact rank partition accounting;
- real forward and backward;
- finite global gradient norm;
- AdamW optimizer update and non-zero sharded parameter delta;
- reduced mean loss and global token count;
- FSDP iteration-state recovery after an aborted forward;
- explicit post-step rank failure propagation;
- bounded join/timeout handling and `destroy_process_group()` in `finally`.

Run from an installed locked environment:

```bash
python tools/run_fsdp2_model_execution.py local-cpu \
  --stage-config configs/stages/s1_100k.json \
  --world-size 2 \
  --samples-per-rank 1 \
  --sequence-length 8 \
  --source-sha "$(git rev-parse HEAD)" \
  --output fsdp2-local-cpu-evidence.json
```

## Exact GPU/NCCL extension path

No paid or external GPU resource is launched by this change. On an authorized machine with at least two visible NVIDIA GPUs and the repository's locked CUDA/NCCL PyTorch environment, the minimal pilot is:

```bash
torchrun --standalone --nproc-per-node=2 \
  tools/run_fsdp2_model_execution.py torchrun \
  --backend nccl \
  --device-type cuda \
  --stage-config configs/stages/s3_10m.json \
  --samples-per-rank 1 \
  --sequence-length 128
```

The process-local CUDA device is selected from `LOCAL_RANK` before DeviceMesh creation. The same native FSDP2/ModelSpec/Trainer adapter then runs under NCCL. This command is launch-ready but is **NOT TESTED ON CUDA** in this lane until an authorized GPU run is actually recorded.

For the cheapest first GPU proof, `configs/stages/s1_100k.json` and `--sequence-length 32` may be substituted before advancing to the 10M stage. That cheaper check validates transport/device wiring but does not meaningfully test FSDP memory scaling.

## Precision boundary

This D12 execution lane uses FP32 deliberately. Active D02 precision work owns BF16/FP16 and GradScaler semantics. After D02's current precision surface lands, it can be passed through FSDP2 using PyTorch `MixedPrecisionPolicy` without duplicating precision contracts here.

## Remaining boundaries

CPU/Gloo proves process-group, DeviceMesh, FSDP2 state transitions, DTensor optimizer integration, sampling, reductions, and failure cleanup. It does not prove NCCL behavior, CUDA memory savings, GPU throughput, multi-node rendezvous, topology-dependent numerical equivalence, distributed checkpoint durability, or resharded resume. Those claims require their own executed evidence.
