# D02 precision runtime contract

Status: experimental engineering hardening. Canonical S0 training remains deterministic `fp32` CPU unless a separately bound run config selects another mode.

## Runtime semantics

`TrainerConfig.precision` is a requested mode, not proof that the current device can execute it. D02 resolves a machine-readable `PrecisionRuntime` before model device transfer, RNG mutation, optimizer/scheduler construction, or scaler construction.

The verified policy is fail-closed:

- `fp32` is accepted only on CPU or an actually available CUDA runtime; no autocast or enabled GradScaler is used.
- an explicit `cuda:N` must refer to a currently visible CUDA device before Trainer calls `model.to(...)`.
- `bf16` CPU uses bfloat16 autocast with FP32 parameters/master weights and no enabled GradScaler.
- `bf16` CUDA requires available CUDA and native BF16 support on the requested device. Where PyTorch exposes `including_emulation`, D02 calls the capability probe with `including_emulation=False`; emulated BF16 is not accepted as the large-training path.
- `fp16` requires available CUDA, float16 autocast, FP32 parameters/master weights, and CUDA GradScaler.
- unverified device types and unavailable requested CUDA modes are rejected before Trainer model/device or optimizer mutation.
- an already down-cast model is rejected: mixed precision in this Trainer means FP32 model/optimizer master weights plus lower-precision eligible compute, not half-precision optimizer state.

The runtime identity records requested precision, device type, parameter dtype, optimizer master dtype, autocast behavior, and GradScaler behavior. Trainer checkpoints preserve the complete `TrainerConfig`, including requested precision, plus scaler state; resume rejects a different Trainer config.

## Measured CPU evidence

The S1 numerical-preflight incumbent (#106, source `8894b44eb8627971b50a734fecd36e74293c1093`) executed the same 107,856-parameter random-init model for six real optimizer steps on a locked Ubuntu 24.04 / Python 3.11.16 / PyTorch 2.13.0 hosted CPU.

| Metric | fp32 CPU | bf16 CPU |
| --- | ---: | ---: |
| initial train loss | 6.237287 | 6.237287 |
| final train loss | 4.738896 | 4.727155 |
| final validation loss | 4.720620 | 4.707920 |
| gradient norm range | 1.597052–3.216873 | 1.597257–3.237895 |
| changed parameter elements | 107,856 / 107,856 | 107,856 / 107,856 |
| weight delta L2 | 9.988740 | 10.060440 |
| six-step wall time | 0.083782 s | 0.386868 s |

On that hosted CPU, bf16 was numerically healthy but about 4.62x slower in wall time than fp32. That is hardware-specific evidence, not a universal CPU bf16 performance claim.

Downstream TRAIN-15 PR #153 is stacked on this precision incumbent and consumes it rather than forking precision logic. Its LOCAL_FREE architecture-matched canonical S3 10,059,840-parameter one-step CPU probe at sequence length 64 reported finite updates and FP32 model/AdamW storage in both modes:

| Mode | loss | grad norm | shifted training tokens/s | fresh-process peak RSS |
| --- | ---: | ---: | ---: | ---: |
| fp32 CPU | 9.043866 | 15.458591 | ~180 | ~579 MiB |
| bf16 CPU autocast | 9.043782 | 15.458709 | ~219 | ~578 MiB |

The opposite CPU throughput result on that PyTorch 2.10.0+cpu host is exactly why performance selection must be device-bound. Precision comparisons use finite trajectory/update evidence and semantic tolerances; bitwise equality across precisions is intentionally not required.

## Larger-model policy

The canonical S3 config is 10,059,840 parameters. TRAIN-15 #153 provides its downstream single-GPU mechanics path, including synchronized step timing, transfer truth, CUDA memory telemetry, OOM failure policy, checkpoint/reload, continuation, and post-training inference. Its default older-GPU-compatible pilot uses CUDA fp16 + GradScaler; bf16 is selectable only after the D02 native-BF16 gate passes.

SCALE-04 PR #152 owns the current S4 accelerator candidate at 99,897,600 parameters. Its current first-order BF16-autocast serious-profile estimate is ~3.926 GiB total training memory: ~0.372 GiB FP32 parameters, ~0.372 GiB FP32 gradients, ~0.744 GiB two FP32 Adam moments, and ~2.438 GiB coarse BF16 activations. This is estimator evidence, not measured VRAM. Because #152 is currently based on the pre-precision #89 lineage and its profile validator is BF16-only, a real S4 accelerator run must integrate this D02 precision incumbent first. Any fp16 fallback must be an explicit run profile, never a silent downgrade.

For future ~10M/~100M GPU runs, selection is explicit:

1. prefer CUDA bf16 only when native BF16 is proven on the requested visible device and a device-bound numerical probe passes;
2. otherwise use explicit CUDA fp16 with GradScaler when a device-bound fp16 probe passes;
3. use explicit CUDA fp32 when reduced precision is unsupported or numerically unsafe;
4. use CPU fp32 as the correctness fallback; CPU bf16 remains an opt-in measured mode rather than a performance assumption.

No requested precision silently changes into another mode. A launcher must record the mode actually selected in the run/training identity.

## Inference and checkpoint boundary

D05 checkpoint identity already binds `training.precision`; D02 Trainer state also includes the full precision-bearing config and scaler state. The first-party D07 loader reconstructs a fresh FP32 `TwelveSixDecoder` and loads the checkpoint into that model without inference autocast, so current canonical first-party inference is FP32 even when training used bf16 autocast. D07 owns any future inference-dtype/quantization policy; this D02 work does not create a competing inference mode.

## Retained scale evidence

The D02 precision-scale probe exercises the existing S3 ~10M ModelSpec in separate fp32 and CPU-bf16 processes and records loss trajectory, gradient norms, weight deltas, parameter/optimizer tensor bytes and dtypes, process RSS, step/token throughput, scaler trajectory, checkpoint precision identity, native post-training inference dtype, and CUDA memory only when CUDA actually executes. Its workflow is path-scoped so unrelated downstream PRs do not repeatedly run the scale probe.

## Truth boundary

CUDA libraries being installed is not CUDA execution evidence. No GPU was available in the current hosted/LOCAL_FREE evidence paths, so this package does not claim measured CUDA bf16/fp16 loss, speed, memory, GradScaler dynamics, MFU, or production GPU readiness. It also does not claim cross-hardware bitwise reproducibility, paid compute authorization, stage promotion, foreign pretrained weights, or instruction/alignment behavior.
