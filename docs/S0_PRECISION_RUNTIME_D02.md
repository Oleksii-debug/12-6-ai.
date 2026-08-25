# D02 precision runtime contract

Status: experimental engineering hardening. Canonical S0 training remains deterministic `fp32` CPU unless a separately bound run config selects another mode.

## Runtime semantics

`TrainerConfig.precision` is a requested mode, not proof that the current device can execute it. D02 resolves a machine-readable `PrecisionRuntime` before model device transfer, RNG mutation, optimizer/scheduler construction, or scaler construction.

The verified policy is fail-closed:

- `fp32` is accepted only on an explicitly available CPU/CUDA runtime; no autocast or GradScaler is used.
- `bf16` CPU uses bfloat16 autocast with FP32 parameters/master weights and no GradScaler.
- `bf16` CUDA additionally requires `torch.cuda.is_available()` and a positive `torch.cuda.is_bf16_supported()` probe.
- `fp16` requires available CUDA, float16 autocast, FP32 parameters/master weights, and CUDA GradScaler.
- unverified device types and unavailable requested CUDA modes are rejected before Trainer side effects.
- an already down-cast model is rejected: mixed precision in this Trainer means FP32 model/optimizer master weights plus lower-precision eligible compute, not half-precision optimizer state.

The runtime identity records requested precision, device type, parameter dtype, optimizer master dtype, autocast behavior, and GradScaler behavior. Trainer checkpoints already preserve the complete `TrainerConfig`, including requested precision, and scaler state; resume rejects a different Trainer config.

## Measured CPU evidence

The current exact S1 numerical-preflight incumbent (#106, source `8894b44eb8627971b50a734fecd36e74293c1093`) executed the same 107,856-parameter random-init model for six real optimizer steps on a locked Ubuntu 24.04 / Python 3.11.16 / PyTorch 2.13.0 hosted CPU.

| Metric | fp32 CPU | bf16 CPU |
| --- | ---: | ---: |
| initial train loss | 6.237287 | 6.237287 |
| final train loss | 4.738896 | 4.727155 |
| final validation loss | 4.720620 | 4.707920 |
| gradient norm range | 1.597052–3.216873 | 1.597257–3.237895 |
| changed parameter elements | 107,856 / 107,856 | 107,856 / 107,856 |
| weight delta L2 | 9.988740 | 10.060440 |
| six-step wall time | 0.083782 s | 0.386868 s |

On that specific hosted CPU, bf16 was numerically healthy but about 4.62x slower in wall time than fp32. This is hardware-specific evidence, not a universal CPU bf16 performance claim. Precision comparisons use semantic tolerances and trajectory/finite-update checks; bitwise equality across precisions is intentionally not required.

## Larger-model policy

The repository already contains an executable S3 engineering ModelSpec at 10,059,840 parameters. The next measured precision step should exercise that actual model rather than expand the S0 contract-test matrix.

For future ~10M/~100M GPU runs, selection is explicit rather than silently downgraded:

1. prefer CUDA bf16 when the device reports bf16 support and a device-bound numerical probe passes;
2. otherwise use CUDA fp16 with GradScaler when a device-bound fp16 probe passes;
3. use CUDA fp32 when reduced precision is unsupported or numerically unsafe;
4. use CPU fp32 as the correctness fallback; CPU bf16 remains opt-in because current performance evidence is unfavorable on the hosted CPU.

No requested precision silently changes into another mode. A launcher must record the mode it actually selected in the run/training identity.

## Inference and checkpoint boundary

D05 checkpoint identity already binds `training.precision`; D02 Trainer state also includes the full precision-bearing config and scaler state. The first-party D07 loader reconstructs a fresh FP32 `TwelveSixDecoder` and loads the checkpoint into that model without inference autocast, so current canonical first-party inference is FP32 even when training used bf16 autocast. D07 owns any future inference-dtype/quantization policy; this D02 work does not create a competing inference mode.

## Truth boundary

CUDA libraries being installed is not CUDA execution evidence. No GPU was available in the current hosted S1 preflight, so this package does not claim measured CUDA bf16/fp16 loss, speed, memory, GradScaler dynamics, MFU, or production GPU readiness. It also does not claim cross-hardware bitwise reproducibility, paid compute authorization, stage promotion, foreign pretrained weights, or instruction/alignment behavior.
