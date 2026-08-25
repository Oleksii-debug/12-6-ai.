# TRAIN-129 CUDA precision qualification

Status: **NOT_RUN_NO_CUDA_PASS**.

This branch is a fail-closed device qualification surface. It does not provision, rent, purchase, or authorize compute. A CUDA result is valid only on an accelerator that was already attached to the process and explicitly attested by the operator as free or otherwise authorized.

## Exact live incumbents

Repository identity is `Oleksii-debug/12-6-ai.` including the trailing period.

- Learned Base provenance: `milestone100/first-learned-base-20260826` at `b9bc147e0a08181b91798c2515cac7a79c66791c`. The retained experiment code binds the canonical byte tokenizer by changing the S1 source vocabulary from 512 to 256, yielding exactly 95,568 parameters. It specifies 1,000 fp32 optimizer steps, checkpoints at 0/250/500/750/1000, fresh-process resume at 500, held-out UK/EN/code evaluation with model-state hashing, and first-party checkpoint inference.
- Single-GPU runtime: `train15/single-gpu-pilot-20260825` at `d6b71a7a18a6ac8cf9ec47fa181264b9df55b7eb`. It owns one-visible-device resolution, seed-before-model construction, pinned/non-blocking H2D truth, synchronized throughput, CUDA allocated/reserved peaks, poison-on-OOM policy, fresh-object checkpoint reload/resume, and post-reload inference.
- Precision policy/evidence: `train60/precision-learned-20260825` at `30c2f56a86177ce9fa3f25167389c86fa396d9a8`. Exact required blobs are `src/twelve_six/training/precision.py` = `63b35f695ab4b7d4bf35a777c884d0233f9cc24e` and `src/twelve_six/training/trainer.py` = `0ec579154521a9f11b2167f9f2611a2a05064c52`.

The learned MILESTONE-100 head contains TRAIN-15 ancestry but does **not** contain the TRAIN-60 precision resolver/trainer integration. TRAIN-129 therefore treats repository integration as a hard gate rather than silently using the generic autocast path.

## LOCAL_FREE device result

The available execution process reported PyTorch `2.10.0+cpu`, `torch.version.cuda == None`, `torch.cuda.is_available() == False`, CUDA device count 0, and no `nvidia-smi` executable. No CUDA kernel, VRAM, throughput, bf16, fp16, or GradScaler qualification was executed. This is not a CUDA failure of the model; it is an explicit **NOT_RUN_NO_CUDA_PASS** hardware outcome.

## Learned-scale truth boundary

The strongest substantial learned artifact currently identified is the 95,568-parameter MILESTONE-100 Base. The existing ~1M and ~10M surfaces are executable mechanics/scale candidates, not an already-qualified substantial learned checkpoint under the requested real-data precision comparison. TRAIN-129 does not relabel those mechanics probes as learned precision evidence.

The DATA-25 v0.1 corpus is versioned and split cleanly across UK/EN/code, but its manifest declares zero external training-eligible sources and project-authored data. Therefore it is useful for local model mechanics and the first learned Base proof but does not satisfy a claim of real-world external corpus representativeness.

## Pre-execution acceptance contract

The comparison must use the same learned configuration/data order/evaluation set for all eligible precision modes. fp32 is mandatory as reference. CUDA bf16 is admitted only after a positive native-device probe; emulation is not accepted. CUDA fp16 is admitted only after a finite autocast + enabled GradScaler probe. Model parameters and AdamW master weights remain fp32 under AMP.

Cross-precision trajectories need not be bit-identical. Before seeing results, acceptance is fixed at: all states finite; final train and held-out BPB absolute difference versus fp32 <= 0.08; maximum held-out curve BPB absolute difference <= 0.12; gradient-norm median/p95 relative difference <= 10%/25%; update-norm median/p95 relative difference <= 10%/25%; same-precision checkpoint/reload logits max-abs <= 1e-6. Every accepted mode must also record synchronized throughput, peak CUDA allocated and reserved memory, checkpoint/reload, and post-reload inference. fp16 must record positive scaler state/behavior; bf16 must not use GradScaler. Faster throughput alone cannot promote a lower precision.

## Executable preflight

Run from an exact local checkout:

```bash
python tools/train129_cuda_precision_preflight.py \
  --repo-root . \
  --output reports/train129/preflight.json
```

The default authorization is deliberately `UNAUTHORIZED`, so this command cannot authorize an accelerator launch. On a host where exactly one already-attached GPU is genuinely free/authorized, the operator may explicitly attest that fact:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train129_cuda_precision_preflight.py \
  --repo-root . \
  --output reports/train129/preflight.json \
  --authorization AUTHORIZED_PREPROVISIONED_FREE
```

A zero exit means only that preflight hard gates are ready for a device-bound comparison. It is **not** a CUDA PASS. A CUDA PASS can be issued only after fp32 and every device-eligible lower-precision run complete the full learned comparison and satisfy the acceptance contract.

## Exact blockers before legitimate execution

1. Integrate the exact TRAIN-60 precision resolver/trainer into the learned convergence lineage with ancestry preserved and rerun exact-head regression evidence.
2. Produce or select one substantial learned ~1M or ~10M Base checkpoint/config with frozen tokenizer, corpus/eval identity, data order, checkpoint identity, and no foreign pretrained weights or instruction tuning. Current mechanics-only ~1M/~10M candidates do not pass this gate.
3. Execute on exactly one already-free/authorized CUDA device. Record GPU name, UUID, driver, PCI bus, compute capability, memory, PyTorch/CUDA/cuDNN runtime, and positive native-precision probes.
4. Run fp32 reference first, then only device-qualified bf16/fp16 modes under the same learned experiment. Retain per-mode trajectories and final checkpoints.
5. Recommend a precision only for that exact hardware/runtime scope after quality/numerical gates pass; otherwise keep fp32 or leave the device unqualified.
