# D02 S0 precision runtime contract

Status: experimental engineering hardening. Canonical S0 training remains deterministic `fp32` CPU unless a separately bound run config selects another mode.

## Gap closed

`TrainerConfig.precision` already admitted `fp32`, `bf16`, and `fp16`, but the previous Trainer inferred runtime behavior inline. Unsupported precision/device combinations could fail after Trainer had already started model/device or RNG/optimizer initialization, and downstream evidence had no small machine-readable description of the resolved autocast/scaler policy.

D02 now resolves a `PrecisionRuntime` before Trainer performs model `.to(...)`, deterministic RNG setup, optimizer/scheduler construction, or scaler construction. Unsupported combinations fail closed at that boundary.

## Current verified policy

- `fp32`: no autocast and no GradScaler.
- `bf16` on CPU: bfloat16 autocast, no GradScaler.
- `bf16` on CUDA: requires an available CUDA runtime and a positive `torch.cuda.is_bf16_supported()` capability probe.
- `fp16`: requires an available CUDA runtime, float16 autocast, GradScaler enabled.
- unverified bf16 device types fail closed instead of silently approximating support.

The runtime contract is serializable through `PrecisionRuntime.to_dict()` and is intentionally separate from D05 checkpoint format/version ownership.

## LOCAL_FREE regression evidence

The precision regression suite exercises:

1. fp16-on-CPU rejection before model `.to(...)` and before Python/Torch RNG mutation;
2. exact machine-readable fp32 and CPU-bf16 runtime policies;
3. fail-closed simulated CUDA-bf16 capability rejection;
4. a real canonical D01 10,140-parameter random-init model performing two CPU-bf16 optimizer steps with finite loss, finite gradient norm, exact token accounting, and actual parameter change.

The normal D02 exact-head workflow still runs the canonical 40-step fp32 S0 evidence path. This package does not replace that evidence or silently change S0 defaults.

## Truth boundary

This work does not claim fp32-vs-bf16 numerical equivalence, cross-hardware bitwise reproducibility, CUDA/GPU execution, fp16 execution, throughput/MFU improvement, production mixed-precision readiness, paid compute authorization, or stage promotion. Those require explicit device-bound evidence. Canonical Base remains random-initialized and pretraining-only, with no foreign pretrained weights or instruction/alignment/refusal/personality/domain-specialization behavior.
