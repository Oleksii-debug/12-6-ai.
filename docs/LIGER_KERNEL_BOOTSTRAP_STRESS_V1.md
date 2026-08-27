# Liger Kernel bootstrap stress V1

This package is an independent audit of `linkedin/Liger-Kernel` as an optional open-source training-kernel candidate. It does not adopt Liger into canonical 12-6 and does not modify the canonical Base lineage.

## Exact upstream identity

Repository: `https://github.com/linkedin/Liger-Kernel`
Release: `v0.8.2`
Immutable commit: `000be60929938fd1358e03524c6ab398b6d421bd`
Package: `liger_kernel==0.8.2`
Software license: BSD-2-Clause
LICENSE blob: `d2fcc2b1c4384d0bcd1424b7f83db8e48fa753f6`
NOTICE blob: `ea2881754f5b3e0eb9926dd9dc6c9d772f962911`

PyPI source artifact: `liger_kernel-0.8.2.tar.gz`
SHA-256: `387673ed6bf64fc8150cc8315fed578d2fc717ec3450f53489f480880223c1b8`

The upstream `setup.py` declares `torch>=2.1.2` and `triton>=2.3.1` for CUDA/CPU platform paths. Real kernel benchmarking therefore requires a genuinely installable dependency closure and a supported accelerator; upstream speed claims are not reused as 12-6 evidence.

## Rights audit

The immutable upstream LICENSE is BSD-2-Clause. The immutable NOTICE identifies derived material from Unsloth, Triton, Efficient Cross Entropy, Flash Attention, AutoAWQ and llm.c, each with separate upstream licensing obligations. The NOTICE also names Tiny Shakespeare test data. No upstream datasets or model weights were used by this worker. Software license is therefore not treated as blanket dataset/model rights.

## Environment and installation

Host: CPython 3.13.5, Debian GNU/Linux 13 x86_64, 5 visible CPU cores, no NVIDIA/ROCm/XPU/Ascend accelerator. Available package managers: pip and uv; git available. Host package inventory contained `torch 2.10.0+cpu`, `numpy 2.3.5`, `safetensors 0.7.0`, and `pytest 9.0.2`; `liger-kernel` and `triton` were absent. No exact Liger package was present in local cache and Python 3.11 was unavailable.

A fresh venv was created. Exact installation was attempted with:

`python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: liger-kernel==0.8.2`

The installation returned code 1 after DNS failures resolving `pypi.org`. No artifact was downloaded. An exact upstream tag fetch was also attempted and failed because `github.com` could not be resolved. No global Python package mutation was detected.

## Runtime, benchmark and parity boundary

Real Liger import/runtime: `NOT_EXECUTED`.
Real GPU benchmark: `NOT_EXECUTED`.
Numerical parity for RMSNorm, RoPE, SwiGLU and cross-entropy: `NOT_EXECUTED`.
Mocks were not used as runtime evidence.

The correct retest requires a network-enabled host with a supported accelerator and exact dependency closure. Run two clean attempts, execute real forward/backward probes, compare identical inputs/outputs against a project-owned reference, and measure latency, throughput, RSS/allocation behavior and device utilization. Promotion requires real runtime plus parity and measured benefit.

## Adversarial validation

The fail-closed validator rejects version drift, upstream commit drift, license drift, false runtime success, benchmark success without visible GPU hardware, mocked benchmark evidence and a forged evidence hash. Eight focused validator tests pass locally. These tests validate evidence mechanics only; they do not constitute evidence that Liger itself executed.

## Canonical safety

No canonical Base, tokenizer, corpus, checkpoint, model training, final-test payload or alignment behavior was changed or loaded. No paid compute was launched.

## Result

Primary verdict: `RETEST_RUNTIME_REQUIRED`.

The result is intentionally not `ADOPTABLE_COMPONENT` because installation, real runtime, GPU benchmark and numerical parity could not be executed in the available environment.
