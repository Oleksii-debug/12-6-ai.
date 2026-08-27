# Lingua 2.2.0 Bootstrap Stress V1

This package independently qualifies the open-source `lingua-language-detector` 2.2.0 release as an optional D08 execution candidate. It does not modify ENV-151 and does not enter canonical Base, tokenizer, corpus, checkpoint, training, or evaluation paths.

## Immutable upstream identity

Repository: `pemistahl/lingua-py`
Tag: `v2.2.0`
Commit: `754ce21122c083a7200763015fdaf7cda8d85453`
License: Apache-2.0; verified from upstream `LICENSE.txt` blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`.
Package: `lingua-language-detector==2.2.0`.

The upstream project declares Python `>=3.12,<3.15`; the v2.2.0 release notes explicitly state that Python 3.11 support was dropped. The exact CPython 3.13 Linux x86-64 wheel is `lingua_language_detector-2.2.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` with SHA-256 `4fbf936b47ef4fdd7043ebb4159d4a5f1c3648028e19d6e3c60464abc5f5e195`.

## Bootstrap compatibility finding

ENV-151 currently pins CPython 3.11.16. That is incompatible with Lingua 2.2.0. This worker did not alter ENV-151 or weaken its exact Python rule. Instead it used a fresh independent CPython 3.13 venv solely to attempt the exact third-party runtime installation.

## Installation evidence

The worker created a dedicated `/tmp` virtual environment. No global package was modified. The exact hashed package install was attempted first against local cache and then against PyPI. The local-cache attempt failed with no matching distribution. The network attempt failed because DNS could not resolve PyPI. No substitute version was installed.

Because the exact wheel was not present locally and external package access was unavailable, real Lingua import/execution, benchmark execution and runtime parity were not performed. This is a hard runtime evidence gap, not a PASS.

## Adversarial contract

The validator rejects upstream commit drift, artifact-hash drift, fabricated runtime execution, adoption without real benchmark/parity evidence, evidence self-hash tampering, canonical Base mutation and foreign-weight use. Negative-path mechanics pass locally under the available Python runtime.

## Retry

Use CPython 3.13 on Linux x86-64 and install exactly the wheel matching SHA-256 `4fbf936b47ef4fdd7043ebb4159d4a5f1c3648028e19d6e3c60464abc5f5e195`. After installation, rerun the planned two-repeat benchmark and fixed-fixture parity suite before considering promotion.

## Verdict

`RETEST_RUNTIME_REQUIRED`.
