# DataTrove bootstrap stress V1 — SWARM-779

## Verdict

`RETEST_RUNTIME_REQUIRED`

This independent audit establishes immutable upstream identity, license truth, environment capability, an actual isolated installation attempt, and fail-closed runtime evidence. It does not promote DataTrove.

## Ownership boundary

The primary DataTrove qualification package is already owned by SWARM-774 under `D03|DATATROVE|OPEN-SOURCE-REUSE-RESEARCH|BOOTSTRAP-STRESS-V1`. This audit deliberately does not modify that Product surface, the D03 production pipeline, or any workflow.

## Upstream identity

- repository: `huggingface/datatrove`
- stable release: `v0.10.0`
- tag commit: `7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664`
- release date: `2026-08-13T16:08:07Z`
- license: Apache-2.0
- exact LICENSE Git blob: `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`
- `NOTICE` at the pinned commit: absent
- `pyproject.toml` Git blob: `fad8b1f20a2d48d4abbf27f6544d498b38981419`
- declared Python floor: `>=3.10.0`
- GitHub release assets: none; release artifact SHA-256 therefore remains unavailable from this source

The upstream release metadata also records a compatibility fix pinning `xxhash<4` because `xxhash 4.0.0` breaks DataTrove dedup pipelines. That is upstream release information, not a 12-6 performance claim.

## Environment

- Python 3.13.5
- pip 25.1.1
- uv available at `/opt/pyvenv/bin/uv`
- git 2.47.3
- Linux x86_64, kernel 6.18.35
- Intel Xeon Platinum 8573C, 5 logical CPUs
- no NVIDIA GPU detected
- Poetry/PDM/Conda unavailable
- DataTrove absent from the global interpreter
- no DataTrove-specific uv/pip/wheel cache was found under `/home/oai/.cache`

## Installation attempt

An isolated environment was created with `uv venv /tmp/swarm779-datatrove-venv --python python`.

The exact request was:

`uv pip install --python /tmp/swarm779-datatrove-venv/bin/python datatrove==0.10.0 --index-url https://pypi.org/simple --no-cache`

The package index request was attempted and retried three times. It failed because the runtime could not resolve `pypi.org` (`Temporary failure in name resolution`). The exact distribution was therefore not installed and the real package import could not be executed. The temporary virtual environment was removed afterward. No global package mutation occurred.

## Runtime / benchmark truth

`real_datatrove_import_executed=false`

`bounded_fixture_pipeline_executed=false`

`benchmark_executed=false`

`parity_proven=false`

No test double is used as evidence of runtime quality. No upstream speed claim is adopted.

## Rights boundary

The software code is Apache-2.0 under the exact pinned LICENSE. Absence of a NOTICE file was checked at the pinned commit. This software license does not establish rights for any dataset payload that DataTrove might later process; those remain source-specific and downstream.

No model weights, tokenizer, corpus, benchmark/final-test payload, checkpoint, training run, or paid compute were touched.

## Re-test contract

The next runtime worker should use a network-capable, isolated environment and install exactly `datatrove==0.10.0` from the pinned release lineage. It must record the resolved wheel/sdist SHA-256, dependency versions, import version, and real bounded fixture execution before any `PARITY_PROVEN` or adoption decision. If the exact dependency still cannot be fetched, retain `RETEST_RUNTIME_REQUIRED` and do not substitute another version.

Machine evidence: `evidence/audit/datatrove_bootstrap_stress_v1.json`.
Validator: `tools/validate_datatrove_bootstrap_stress_v1.py`.
Tests: `tests/test_datatrove_bootstrap_stress_v1.py`.
