# GPU-201 10M CUDA benchmark

`GPU-201-10M-CUDA-BENCHMARK` is a target-device performance campaign for the current scratch-Base S3 10M training line. It is deliberately separate from model-quality promotion and from CPU performance evidence.

## Scientific binding

The campaign is bound to the current SCALE-141 geometry and optimizer seam: `S3-SCALE02-BYTE-GQA-v1`, 10,000,640 parameters, ModelSpec `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`, InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`, canonical `s0-byte-v1`, DATA-25 corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`, sequence length 256, and the live SCALE-141 AdamW configuration. The timed batches are deterministic fixed tensors used only for device timing; the report binds the corpus identity but does not mislabel synthetic timing input as corpus-I/O throughput.

The accepted precision runtime is composed directly into the Trainer. The accepted PERF-148 instrumentation blob is reused exactly and its Git blob identity is verified before CUDA execution. `torch.compile` remains disabled because no separate accepted compile decision is assumed by this worker.

## Fail-closed execution

An ordinary CPU runner produces only `NOT_RUN_NO_GPU`. It does not execute a 10M CPU benchmark and does not extrapolate CPU numbers to CUDA.

A CUDA run additionally requires both of these exact-device inputs before model construction:

- successful `linux-x86_64-cuda-training` purpose-environment evidence for the exact source SHA; and
- executed `GPU-199-CUDA-PRECISION-PILOT` evidence bound to the same CUDA device, including an explicit selected precision.

A missing, `NOT_RUN_NO_GPU`, stale, differently bound, or unsupported GPU-199 result aborts the CUDA campaign. GPU-201 does not independently invent the precision decision.

## Measurements

Default timing uses 8 warmup optimizer steps and 32 measured optimizer steps, with a hard minimum of 4 warmups and 16 measured steps. It reports end-to-end optimized tokens/s, trainer-step tokens/s, data-wait time, forward/loss time, backward time, update-component time, peak allocated/reserved VRAM, and device identity. FP32 is always measured. Native BF16 is measured when the exact device proves support. FP16 is measured only when GPU-199 selected it as the justified fallback.

The selected precision also receives a fresh checkpoint save/verify/load roundtrip measured with synchronized host wall time. A geometric microbatch headroom probe starts at 1 and doubles until the configured ceiling or the first CUDA OOM; OOM is recorded and the failed in-memory attempt is discarded.

PERF-148 profiling is checked against an unprofiled canonical Trainer replay over the exact same initialization and batch trace. The final parameter fingerprint must match exactly and all parameters must remain finite. This is the semantic guard: performance instrumentation cannot silently change the update trajectory.

## Workflow policy

The pull-request job runs on the ordinary GitHub-hosted CPU runner and proves the exact `NOT_RUN_NO_GPU` contract plus the CUDA purpose-profile bootstrap. The real CUDA job is manual-only and targets generic self-hosted labels `[self-hosted, linux, x64, cuda, gpu]`. It does not provision hardware, purchase compute, or contain provider credentials.

No foreign pretrained weights, SFT, RLHF, or DPO are used. The benchmark makes no claim of intelligence, alignment, instruction following, production readiness, or model promotion.
