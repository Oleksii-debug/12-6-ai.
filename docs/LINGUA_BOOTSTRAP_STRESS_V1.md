# LINGUA bootstrap stress qualification V1

Status: `RETEST_RUNTIME_REQUIRED`

This package qualifies the open-source Lingua language-identification library as an optional D03 LID cross-check. It does not change the canonical Base, tokenizer, corpus authority, checkpoint lineage, training, evaluation payloads, or paid-compute policy.

## Exact upstream identity

Repository: `https://github.com/pemistahl/lingua-py`  
Release tag: `v2.2.0`  
Annotated tag object: `f43375cba13a2ee209d632febfe9e074f3ff1d91`  
Resolved commit: `754ce21122c083a7200763015fdaf7cda8d85453`  
Package: `lingua-language-detector==2.2.0`  
Python range declared upstream: `>=3.12,<3.15`  
Release date: `2026-03-09`

Upstream release notes describe a switch in 2.2.0 to finite-state-transducer language-model storage, reducing memory consumption while retaining offline use. This project does not copy upstream performance claims into 12-6 evidence.

## Rights

The immutable `LICENSE.txt` at the pinned commit is Apache License 2.0. License blob SHA-1: `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`.

No separate `NOTICE` or `COPYING` file was found by repository search at the pinned revision. This records absence of those filenames, not a claim about third-party rights outside the repository.

The package is software infrastructure. Dataset rights are not inferred from its software license. This qualification makes no claim to any neural model-weight license: Lingua 2.x uses compiled bindings and statistical language-model assets rather than a foreign pretrained neural checkpoint.

## Local bootstrap result

Observed host:
- CPython 3.13.5
- Debian 13 / Linux 6.18.35 x86_64
- AMD EPYC 9V74 80-Core Processor
- 5 visible logical CPUs
- no NVIDIA GPU / `nvidia-smi` unavailable
- pip 25.1.1
- uv 0.10.0
- git 2.47.3
- Poetry, PDM and Conda unavailable
- `torch==2.10.0+cpu` installed globally
- `lingua-language-detector` absent
- pip cache disabled/unavailable

An isolated Python venv was created. A real `pip install lingua-language-detector==2.2.0` attempt was started in that venv. The host could not resolve `pypi.org` or `github.com`; no matching wheel/source existed in the local pip cache. The attempt timed out rather than silently selecting another version.

Therefore runtime execution, benchmark and parity are **NOT EXECUTED**. No mock runtime is presented as evidence.

## Deterministic retry

Use CPython 3.13 and a new isolated venv. Install exactly:

`lingua-language-detector==2.2.0`

For this host the intended wheel is:

`lingua_language_detector-2.2.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

Before installation, obtain the published wheel SHA-256 from PyPI immutable file metadata and add it to the manifest. Then install with hash enforcement, record installed distribution metadata, and run the project probe.

Do not substitute 2.1.x, a source checkout at floating `main`, the slim package, or another interpreter build.

## Promotion gate

Promotion to `ADOPTABLE_COMPONENT` requires exact installation, real import, deterministic UA/EN/code/mixed/noise probes, focused/adversarial tests, repeatability, and meaningful comparison against the current project LID contract. Until those are executed on the exact dependency, this package remains `RETEST_RUNTIME_REQUIRED`.
