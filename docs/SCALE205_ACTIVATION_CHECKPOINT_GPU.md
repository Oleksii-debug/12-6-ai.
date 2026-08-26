# SCALE-205 target-GPU activation-checkpoint validation

## Scope

SCALE-205 validates the existing SCALE-143 activation-checkpointing decision rule on real CUDA hardware when and only when such hardware is already visible and authorized. It does not introduce another recomputation implementation.

The branch is rooted at exact SCALE-143 head `30f85fdfb930bb38d45d116fee9c2e82f8241b56`. The only checkpoint implementation consumed by the GPU probe is the accepted PyTorch `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` path with `CheckpointImpl.NO_REENTRANT` and `preserve_rng_state=True`.

No paid compute is authorized.

## Truth boundary

A CPU result, a CUDA build without a visible device, an HBM estimate, or the SCALE-143 CPU extrapolation can never produce `PASS_GPU_MEASURED`.

The GPU runner must satisfy the committed D08 `linux-x86_64-cuda-training` runtime identity:

- CPython 3.11.16;
- PyTorch 2.13.0;
- PyTorch CUDA runtime 13.0;
- an actually visible CUDA device.

If `torch.cuda.is_available()` is false, the result is exactly `NOT_RUN_NO_GPU` and `gpu_pass=false`. If a CUDA device is visible but the accepted D08 Torch/CUDA identity is not present, the result is `BLOCKED_CUDA_ENVIRONMENT_MISMATCH`.

## Compared policies

Only the SCALE-143 incumbent choices relevant to the decision rule are compared:

1. `none`;
2. `per_block`.

`every_other_block` is intentionally excluded because SCALE-143 did not select it as the default pressure response.

Each matched setup uses the same model specification, initialization seed, batch size, sequence length, precision, optimizer geometry, and synthetic token trace. The benchmark consumes `tools/benchmark_activation_checkpointing.py` from SCALE-143 rather than duplicating its timing/memory implementation.

## Model targets

### ~10M

The GPU probe consumes `configs/stages/s3_10m.json` from SCALE-143:

- expected parameters: 10,059,840;
- ModelSpec SHA-256: `3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a`;
- maximum context: 1024.

The comparison is attempted at context 1024.

### ~100M

The GPU probe consumes `configs/stages/s4_100m_accelerator.candidate.json`:

- expected parameters: 99,897,600;
- ModelSpec SHA-256: `6103d0d457e25206c11871f09aef1f2e23860329c060379c9f956b3851740170`;
- maximum context: 4096;
- preferred precision: bf16;
- status remains `engineering_candidate_not_frozen` with promotion disabled.

The probe attempts 4096 first and falls back through smaller contexts only on genuine CUDA OOM. The retained comparison is the largest context at which both `none` and `per_block` complete, so memory and throughput deltas remain matched.

## Measurements

For each successful policy run the retained SCALE-143 benchmark reports:

- peak CUDA allocated bytes;
- peak CUDA reserved bytes;
- CPU RSS;
- median forward seconds;
- median backward seconds;
- median optimizer seconds;
- median total step seconds;
- median optimized tokens/second;
- loss.

SCALE-205 additionally records:

- exact GPU name and compute capability;
- total HBM;
- free HBM at preflight;
- memory saved by per-block checkpointing;
- per-block/no-checkpoint step-time ratio;
- per-block/no-checkpoint throughput ratio;
- deterministic first-party logit/gradient parity deltas under an explicit numeric tolerance;
- a GPU PyTorch DCP save/load round-trip after a real optimizer update.

## 80% memory-headroom rule

SCALE-143 says to retain `none` while measured/calibrated uncheckpointed peak is at most 80% of usable device memory and to switch directly to `per_block` above that boundary or on OOM.

SCALE-205 tests that rule against actual target hardware. `usable_hbm_bytes` is the free device memory measured immediately at GPU preflight, before the model comparisons. The decision metric is:

`uncheckpointed_peak_reserved_bytes / usable_hbm_bytes`

- ratio <= 0.80: rule selects `none`;
- ratio > 0.80: rule selects `per_block`.

The report retains the measured ratio and decision. It does not substitute total-board HBM for contemporaneously usable free HBM.

## Numerical and checkpoint compatibility

For each retained matched setup, `none` is the reference trajectory and `per_block` is reconstructed from the identical initial state and token trace. The probe records maximum absolute logit and gradient deltas. The current conservative pass gates are:

- bf16: absolute delta <= 2e-2;
- fp32: absolute delta <= 1e-7.

The report also records the conventional relative tolerance associated with the SCALE-143 precision envelope, but the pass gate is the stricter absolute-delta test above.

DCP compatibility is tested on the real CUDA device by applying per-block checkpointing, executing forward/backward plus AdamW update, saving model and optimizer state with PyTorch Distributed Checkpoint, deliberately mutating a parameter, loading the checkpoint, and requiring exact restoration of the probe parameter. SCALE-143 separately retains the repository FSDP2+DCP exact-resume integration evidence; this GPU check is a device-bound DCP smoke and does not replace that distributed trajectory test.

## Workflow behavior

`.github/workflows/scale205-activation-checkpoint-gpu.yml` is fail-closed.

On an ordinary GitHub-hosted CPU runner it validates the committed D08 CUDA-purpose profile, records `NOT_RUN_NO_GPU`, uploads that evidence, and exits successfully without simulating CUDA.

If an already-authorized runner exposes a real NVIDIA CUDA device, the workflow installs the exact hash-locked canonical runtime behind the D08 CUDA-purpose profile and runs the device-bound probe. No credential, provider, server-provisioning, or paid-compute path is present in this work.

## Status interpretation

- `NOT_RUN_NO_GPU`: valid negative preflight evidence; not a GPU PASS.
- `BLOCKED_CUDA_ENVIRONMENT_MISMATCH`: a device was visible but runtime identity was not authoritative.
- `NO_COMMON_FEASIBLE_SETUP`: the compared policies did not share a feasible context for the requested model.
- `FAIL_GPU_MEASURED`: CUDA execution occurred but at least one required parity/DCP/model comparison failed.
- `PASS_GPU_MEASURED`: and only this status is a target-GPU validation PASS.

Until a terminal artifact reports `PASS_GPU_MEASURED`, SCALE-143 remains CPU-calibrated and GPU validation remains incomplete.
