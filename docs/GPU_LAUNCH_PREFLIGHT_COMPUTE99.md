# COMPUTE-99 GPU launch preflight

This is a fail-closed qualification surface for future 10M and 100M accelerator work. It provisions nothing and authorizes nothing. A visible CUDA device may be consumed only by the existing bounded 10M TRAIN-15 mechanics pilot; if CUDA is absent the retained result is `NOT_RUN_NO_GPU` with VRAM and throughput fields left null.

The current source baseline is Product/convergence head `fb9c6d9b73ce436d637077892d73edf136fcaeac`. The preflight binds its D08 CUDA-training environment profile, checkpoint-v1 surface, observability surface and merged single-GPU runner. It also records the current D02 precision incumbent separately because that hardened precision implementation is not composed onto this Product baseline.

## Exact current candidates

The 10M execution binding is the run manifest that is actually merged today: `configs/runs/s3_10m.single_gpu_pilot.experimental.json` -> `configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json`, 10,000,640 parameters, ModelSpec `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`. It uses controlled synthetic integer IDs and therefore does not constitute a tokenizer/corpus freeze.

The 100M binding is `configs/runs/s4_100m_serious.json` -> `configs/stages/s4_100m_accelerator.candidate.json`, 99,897,600 parameters, ModelSpec `6103d0d457e25206c11871f09aef1f2e23860329c060379c9f956b3851740170`. Its current byte-tokenizer engineering contract is `s0-byte-v1`, config SHA-256 `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`, vocab SHA-256 `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`. The stage itself is explicitly an engineering candidate and its serious run remains blocked on scaled D03 corpus production.

The D08 runtime lock is `linux-x86_64-cuda-training`, purpose-index identity `561f070c1792ddde5cf7a6b8df6beacfe93622b201d791939bf77b5c0b3f29c0`, profile identity `e2880fb2c9dbd56f84b558d8705d5248cbf079401da4ba83cc6d382e5a6cdbd6`, resolved profile SHA-256 `d2598da73301c8bbb995bb97894223bf4a65f053dd98b958d63a08b074591822`.

## Current qualification result

The current state is `BLOCKED`, intentionally. A legitimate accelerator campaign still requires all of the following to change with exact identities, not prose assertions: the hardened precision incumbent must be composed and native target-device precision proved; tokenizer, corpus and evaluation must each be `FROZEN` with exact SHA-256 identities; the stale CAMPAIGN-47 package must be regenerated against these exact 10M/100M ModelSpecs; a durable checkpoint target must be selected; an approved last-known-good/fresh-process recovery policy must be composed; the 10M memory lower bound must be replaced by a complete capacity estimate or measured CUDA evidence; the 10M CUDA smoke must pass on an already-free/current device; and the launch manifest must contain `COMPUTE_AUTHORIZED=true` plus a non-empty authorization ID before any authorized campaign launch.

For 100M, the retained SCALE-04 serious-profile memory figure is approximately 3.926 GiB before the required safety headroom. It is estimator evidence only, not measured VRAM. The 10M figure in the current manifest is only the FP32 parameter/gradient/two-Adam-moment persistent-storage floor, 160,010,240 bytes; activations and allocator behavior are deliberately excluded, so it is not accepted as a complete capacity estimate.

The current worker environment was actually probed and has PyTorch `2.10.0+cpu`, no CUDA device and no `nvidia-smi`, so the 10M smoke is retained as `NOT_RUN_NO_GPU`; tokens/sec and CUDA peak memory remain null.

## Running the preflight

From an exact repository checkout:

```bash
python tools/run_gpu_launch_preflight.py \
  --output evidence/compute99/gpu_launch_preflight.json
```

If exactly one CUDA device is already visible, the command delegates the bounded smoke to the existing TRAIN-15 runner and retains its synchronized tokens/sec and CUDA peak allocated/reserved bytes. It never contains a provisioning or purchase path. Use `--skip-smoke` only when intentionally validating source/config gates without consuming an already-visible accelerator.

To make CI or an operator command fail unless the campaign is fully qualified:

```bash
python tools/run_gpu_launch_preflight.py \
  --output evidence/compute99/gpu_launch_preflight.json \
  --assert-launch-ready
```

The current manifest is expected to exit nonzero with `--assert-launch-ready` because authorization and multiple technical gates are intentionally unresolved.

## Cost truth boundary

Do not convert CPU throughput into euro cost. Cost projection is permitted only after measured accelerator throughput exists on the exact bound geometry and an explicit current accelerator rate is supplied. No paid compute was launched by COMPUTE-99.
