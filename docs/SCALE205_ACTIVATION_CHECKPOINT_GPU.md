# SCALE-205 activation-checkpoint GPU qualification

Status: **NOT_RUN_NO_GPU in the connected worker environment**. No GPU PASS is claimed by this branch until the device-bound evidence file reports `status: PASS` from an explicitly free/authorized CUDA runner.

SCALE-205 consumes SCALE-143 at exact source SHA `30f85fdfb930bb38d45d116fee9c2e82f8241b56`. It does not add a recomputation implementation. The only activation-checkpoint implementation remains SCALE-143's maintained PyTorch `checkpoint_wrapper` with `CheckpointImpl.NO_REENTRANT`, RNG preservation, and per-block wrapping.

## Target identities

- S3 / ~10M: `configs/stages/s3_10m.json`, 10,059,840 parameters, maximum context 1024.
- S4 / ~100M: `configs/stages/s4_100m_accelerator.candidate.json`, 99,897,600 parameters, maximum context 4096, preferred bf16 qualification.
- Exact CUDA purpose environment: `linux-x86_64-cuda-training`, Python 3.11.16, PyTorch 2.13.0 CUDA 13.0 closure as recorded by D08.

S4 remains an engineering candidate; this qualification cannot freeze or promote it.

## Measurement protocol

`tools/run_scale205_activation_checkpoint_gpu.py` is a bounded evidence harness around the existing SCALE-143 benchmark and checkpoint wrapper.

For 10M it measures the native maximum context, batch 1, for both `none` and `per_block`.

For 100M it searches a bounded set of batch/context pairs in fresh subprocesses, largest valid tokens per step first with longer context as the tie-break. A CUDA OOM is evidence, not a process to recover and continue in the same allocator state. The first per-block-feasible setup becomes the largest feasible 100M setup for the final paired comparison. The same setup is then measured with `none` and `per_block`.

Each successful benchmark reports SCALE-143's existing CUDA metrics:

- peak allocated VRAM;
- peak reserved VRAM;
- forward time;
- backward time;
- optimizer and total step time;
- tokens/second.

The qualification also records physical HBM using `torch.cuda.mem_get_info()` before model allocation. `usable_hbm_bytes` is the smaller of device total HBM and free HBM at preflight, so an occupied device cannot receive an optimistic headroom denominator.

## 80% rule validation

The SCALE-143 policy is evaluated from measured target-hardware values, not CPU RSS or a linear GPU estimate:

- if uncheckpointed execution OOMs, select `per_block`;
- otherwise compute `peak_reserved_vram / usable_hbm`;
- at or below 0.80, select `none`;
- above 0.80, select `per_block`.

Peak **reserved** VRAM is the policy input because it represents allocator pressure more conservatively than allocated tensors alone. Both allocated and reserved values remain in evidence.

A selected policy is not considered executable unless the corresponding measured run succeeds at that exact scale/setup.

## Numerical parity

Parity is pairwise `none` versus `per_block` only. The harness keeps one GPU model resident at a time: initial state and the control logits/gradients are moved to host memory before the checkpointed model is built. This avoids turning a parity check into an artificial two-model VRAM requirement.

Tolerance is inherited from SCALE-143:

- fp32: rtol 1e-6, atol 1e-7;
- bf16: rtol 2e-2, atol 2e-2;
- fp16: rtol 5e-3, atol 5e-3.

Both logits and all canonicalized parameter gradients must be within tolerance.

## DCP compatibility

For both model scales, the CUDA qualification runs a fresh single-rank NCCL DCP round trip with per-block checkpointing:

1. construct the exact stage model;
2. apply SCALE-143 per-block checkpoint wrapping;
3. materialize optimizer state with one bounded step;
4. save through the repository `save_scale_checkpoint` path;
5. independently verify the committed DCP payload;
6. rebuild model + wrapper + optimizer;
7. exact-topology load through `load_scale_checkpoint`;
8. compare post-load logits to the pre-save reference;
9. retain only compact identities/results and delete the transient tensor payload.

This is compatibility evidence, not a learned-training checkpoint or promotion artifact.

## Compute authorization and workflow

`.github/workflows/scale205-activation-checkpoint-gpu.yml` has two paths:

- an ordinary hosted CPU job proves the harness returns `NOT_RUN_NO_GPU` and cannot manufacture a GPU PASS;
- the measurement job is eligible only when repository variable `SCALE205_FREE_GPU_AUTHORIZED` is exactly `true` and a self-hosted runner matches `[self-hosted, linux, x64, gpu, twelve-six-free-authorized]`.

The GPU job checks out the exact SHA, verifies the D08 CUDA purpose environment, builds the persistent environment from exact hash-locked toolchain/runtime files, records GPU/driver/HBM identity, executes only bounded qualification probes, and uploads compact JSON/text evidence. No managed GPU runner, provider provisioning, credentials, or paid-compute authorization is introduced.

## Current result

The connected execution environment used to assemble this branch reports PyTorch `2.10.0+cpu`, `torch.cuda.is_available() == false`, zero CUDA devices, and no `nvidia-smi`. Therefore:

- 10M GPU VRAM/timing/parity/DCP: **NOT RUN**;
- 100M GPU VRAM/timing/parity/DCP: **NOT RUN**;
- 80% HBM rule: **NOT VALIDATED ON GPU**;
- CPU SCALE-143 memory results remain CPU evidence only.

Machine-readable local preflight: `evidence/swarm_exp_01/scale205_connected_worker_no_gpu_20260826.json`.

A future device-bound result is authoritative only if `scale205-activation-checkpoint-gpu.json` records `status: PASS`; any `NOT_RUN_*`, OOM-only, or `GPU_EXECUTED_WITH_BLOCKERS` result must not be promoted to a GPU PASS.
