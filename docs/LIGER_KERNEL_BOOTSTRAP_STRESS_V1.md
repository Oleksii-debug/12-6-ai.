# Liger Kernel Bootstrap Stress V1

Status: RETEST_RUNTIME_REQUIRED

Worker: `OPEN-SOURCE-BOOTSTRAP-STRESS-V1`
Lane: `D08|LIGER-KERNEL|OPEN-SOURCE-REUSE-RESEARCH|BOOTSTRAP-STRESS-V1`
Claim issue: #778
Parent authority: #720 and D08/#9
Project base: `main@5020afd671a3885c1b738c8b4eafe7525f630546`

## Exact upstream identity

- Repository: `linkedin/Liger-Kernel`
- Release: `v0.8.2`
- Tag commit: `000be60929938fd1358e03524c6ab398b6d421bd`
- Package: `liger-kernel==0.8.2`
- Release sdist SHA-256: `387673ed6bf64fc8150cc8315fed578d2fc717ec3450f53489f480880223c1b8`
- License: BSD-2-Clause
- LICENSE blob: `d2fcc2b1c4384d0bcd1424b7f83db8e48fa753f6`
- NOTICE blob: `ea2881754f5b3e0eb9926dd9dc6c9d772f962911`

The upstream package describes itself as Triton kernels for LLM training and declares `torch>=2.1.2` and `triton>=2.3.1` for its normal CPU/CUDA installation path. The selected qualification surface is RMSNorm, RoPE, SwiGLU and cross-entropy. Real parity requires actual upstream execution; CPU reference calculations are not Liger runtime evidence.

## Rights boundary

The software license was checked from the exact release tag. The upstream NOTICE records third-party material from Unsloth, Triton, Efficient Cross Entropy, Flash Attention, AutoAWQ and llm.c, plus a Tiny Shakespeare test-data reference. This worker does not ingest datasets and does not use model weights. No dataset or model-weight rights are inferred from the software license.

## Environment / bootstrap

A dedicated temporary Python environment was created at `/tmp/12-6-liger-bootstrap-v1` using Python 3.13.5. Host facts recorded in machine evidence:

- Debian 13 / Linux x86_64
- 5 CPU cores
- no NVIDIA or ROCm GPU detected
- `torch==2.10.0+cpu` present globally, with CUDA unavailable
- `liger-kernel` absent
- `triton` absent
- pip 25.1.1, uv 0.10.0 and git 2.47.3 available
- poetry, pdm and conda unavailable
- PyPI/GitHub DNS unavailable

Install attempts were made rather than merely declaring the package missing. `pip install liger-kernel==0.8.2` failed because the package index could not be resolved. A second `uv` attempt did not install anything; the locally installed uv CLI rejected the retry argument used by the generic bootstrap command. No fallback version was substituted.

## Tests and adversarial controls

The qualification suite contains a positive contract test plus fail-closed tests for:

- mutable/floating upstream identity;
- upstream commit and artifact drift;
- paid-compute escalation;
- foreign pretrained weights/tokenizer or canonical Base mutation;
- fabricated `PARITY_PROVEN` state;
- fabricated benchmark execution;
- evidence/environment identity tampering.

Local result before publication: `10 passed`; Python compilation: PASS; manifest validator: PASS.

## Runtime / benchmark / parity truth

Real Liger runtime was not executed. No GPU Triton environment existed and the exact package was not installable from the available network/cache state. Therefore:

- benchmark runs: `0/2`;
- forward/backward parity: not proven;
- latency/throughput/RSS/GPU-utilization measurements: not executed;
- adoption: not authorized;
- no upstream speed claim is transferred into 12-6 evidence.

The machine evidence remains `RETEST_RUNTIME_REQUIRED`, not PASS.

## Resume procedure

In a purpose environment with exact package availability:

1. Create a fresh isolated environment and install `liger-kernel==0.8.2` from the recorded immutable release artifact, retaining the artifact SHA-256.
2. Resolve an exact Triton/PyTorch environment whose versions are recorded rather than floating.
3. Run the existing harness against project-owned numeric fixtures for RMSNorm, RoPE, SwiGLU and cross-entropy, including forward and backward checks against independent PyTorch references.
4. Run each benchmark arm twice from clean processes and record latency, throughput, peak RSS/allocation counters and GPU utilization where available.
5. Require identical inputs, dtypes, shapes, seeds and tolerances between reference and upstream-derived arms.
6. Rebuild evidence and require the validator to accept the new runtime identities before any `PARITY_PROVEN` consideration.
7. Review the NOTICE/third-party obligations again if source files are copied rather than used as an external dependency.

## Canonical Base safety

No canonical Base weights, tokenizer, checkpoint, optimizer, training data, instruction/alignment behavior, or training run was modified. No paid compute was used. Liger remains an optional infrastructure candidate only.

## Durable artifacts

- `configs/research/liger_kernel_bootstrap_stress_v1.json`
- `tools/liger_bootstrap_stress.py`
- `tools/validate_liger_kernel_bootstrap_stress_v1.py`
- `tests/test_liger_kernel_bootstrap_stress_v1.py`
- `evidence/research/liger_kernel_bootstrap_stress_v1.json`
- this document

Next safe action: run the exact package in a purpose GPU/Triton environment and requalify from the current live main SHA before any promotion beyond candidate/retest state.
