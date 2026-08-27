# DataTrove Bootstrap Stress V1

Status: `RETEST_RUNTIME_REQUIRED`

Worker: `SWARM-774`
Lane: `D03|DATATROVE|OPEN-SOURCE-REUSE-RESEARCH|BOOTSTRAP-STRESS-V1`
Base: `main@5020afd671a3885c1b738c8b4eafe7525f630546`
Branch: `swarm/774-datatrove-bootstrap-stress`

## Exact upstream identity

Repository: `https://github.com/huggingface/datatrove`

Release/tag: `v0.10.0`

Commit: `7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664`

PyPI release date: `2026-08-13`

Wheel SHA-256: `c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d`

Source distribution SHA-256: `e31f89bdccb30ef0796854f5842ff52b4b224c28b2d5b110088e84071ea05c40`

PyPI provenance binds the release artifact to the same upstream source commit. The release declares Python `>=3.10`, including Python 3.13.

## Runtime qualification

The worker created a dedicated temporary environment using the available local Python 3.13.5 interpreter and `uv 0.10.0`. No global Python package set was modified.

The exact installation was attempted with:

`uv pip install --python <isolated-python> --no-cache --only-binary=:all: datatrove==0.10.0`

The package index could not be resolved. The exact artifact was not found in the local cache. A second path through the isolated interpreter confirmed that no pip module had been injected into the temporary environment. No alternate DataTrove version was substituted.

Because the exact package was not installed, the following are intentionally `NOT_EXECUTED`:

- real `datatrove` import
- real `LocalPipelineExecutor` execution
- two-repeat benchmark
- output-hash repeatability
- project-vs-upstream runtime parity

## Runtime contract prepared for retest

The runner uses the first-party DataTrove API documented by the pinned release: `LocalPipelineExecutor`, `JsonlReader`, and `JsonlWriter`. The fixture contains English and Ukrainian documents plus a repeated text case. Two independent clean temporary pipeline outputs are hashed and compared. The benchmark is CPU/local-only and does not access model weights, benchmark/final-test data, or training infrastructure.

## Adversarial controls

The validator rejects upstream commit drift, tag drift, wheel/sdist hash drift, incorrect installed version, missing dependency-freeze evidence after installation, mock execution, rights-boundary drift, canonical Base contamination, tokenizer mutation, and false adoption when runtime is `NOT_EXECUTED`.

The focused suite contains seven tests covering valid blocker evidence, upstream identity tamper, wrong installed version, missing lock, mock runtime, Base contamination, and attempted adoption without runtime.

## Rights

The actual upstream `LICENSE` file at `v0.10.0` is Apache-2.0. No `NOTICE` file was found at that tag. This is a software-code license record only. It does not grant rights to datasets processed by DataTrove. Dataset rights remain source-specific and project-controlled.

No dataset payload was downloaded or admitted. No model weight, foreign tokenizer, foreign instruction/alignment behavior, or benchmark/final-test payload entered the project.

## Decision

`RETEST_RUNTIME_REQUIRED`

This is not an upstream quality rejection. It is an environment-bound runtime qualification result. Promotion cannot occur until a networked or equivalently provisioned LOCAL_FREE environment installs the exact pinned artifact, freezes the resolved dependency set, executes the real two-run pipeline, records deterministic output hashes, and completes parity checks.

The complete machine-readable result is stored in `configs/research/datatrove_bootstrap_stress_v1.json`.
