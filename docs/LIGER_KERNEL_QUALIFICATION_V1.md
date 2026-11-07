# Liger Kernel Qualification V1

## Verdict
Current state: `RETEST_RUNTIME_REQUIRED`. Real backend parity is not proven.

## Exact upstream identity
Repository: `https://github.com/linkedin/Liger-Kernel`

Stable release: `v0.8.2`

Tag commit: `000be60929938fd1358e03524c6ab398b6d421bd`

PyPI package: `liger-kernel==0.8.2`

Source distribution: `liger_kernel-0.8.2.tar.gz`

Source distribution SHA-256: `387673ed6bf64fc8150cc8315fed578d2fc717ec3450f53489f480880223c1b8`

Release date: 2026-08-18

## Rights
Primary software license: BSD-2-Clause.

Pinned upstream `LICENSE` blob: `d2fcc2b1c4384d0bcd1424b7f83db8e48fa753f6`.

The upstream project also references third-party components under separate licenses. This qualification does not silently treat those transitive components as covered by the primary BSD-2-Clause license; a production dependency/SBOM intake must enumerate the installed closure and notices before adoption.

No dataset rights, model-weight rights, or pretrained-model rights are being granted by this package.

## Scope
Only these Liger surfaces are candidates for later parity work:

- RMSNorm
- RoPE
- SwiGLU
- cross-entropy

Liger remains an optional training-kernel backend, not part of canonical Base lineage.

## Environment bootstrap result
Worker environment: Linux x86-64, Python 3.13.5, Git 2.47.3, uv 0.10.0, no CUDA/ROCm accelerator detected.

Exact `liger-kernel==0.8.2` installation was attempted in a fresh temporary virtual environment with `pip install --only-binary=:all: --no-deps liger-kernel==0.8.2`.

The installation did not complete because external DNS/network access to PyPI was unavailable. No alternate version was installed and no global environment was changed.

Therefore:

- exact dependency execution: **NOT_EXECUTED_DEPENDENCY_ABSENT**
- GPU kernel parity: **NOT_EXECUTED**
- numerical parity: **NOT_PROVEN**
- throughput/memory benefit: **NOT_MEASURED**
- promotion: **RETEST_RUNTIME_REQUIRED**

## Local-free evidence

Focused/adversarial pytest: **14 passed**. The qualification validator executed twice with the same identity: `fa797a322c24d9d8d78879f2408ee629d93e328ebf051849867fd05405f84d33`. Python `compileall`: **PASS**.

The exact dependency install was attempted in a dedicated virtual environment and failed only because the local runtime could not resolve PyPI. No substitute version was installed.

## Truth boundary
The upstream README contains performance claims, but those claims are not 12-6 evidence. Only a real execution against the exact pinned dependency and compatible accelerator can establish project-local runtime, numerical parity, latency, throughput, or memory evidence.

The package contains no foreign model weights, no tokenizer replacement, no corpus mutation, no training run, no final-test access, and no paid compute.

## Required successor runtime gate
A later worker with the exact package available must execute each supported operator against a project-owned reference implementation using identical inputs, dtypes and shapes; compare forward outputs and gradients within preregistered tolerances; repeat each test twice in clean processes; record peak memory and latency on the actual hardware; and retain the exact installed dependency closure and notice/SBOM identity.

The adoption boundary remains fail-closed until those measurements exist.

## Handoff state

Final worker disposition: `RETEST_RUNTIME_REQUIRED`. A future exact-runtime worker must re-run the pinned package with a compatible GPU, then perform real forward/gradient parity plus two clean-process repeats and latency/memory measurements before any `PARITY_PROVEN` or `ADOPTED` state.
