# Inspect AI Bootstrap Stress V1

Status: `RETEST_RUNTIME_REQUIRED`

Worker: `SWARM-784`
Lane: `D06|INSPECT_AI|PERFORMANCE_RUNTIME|BOOTSTRAP-STRESS-V1`
Base: `5020afd671a3885c1b738c8b4eafe7525f630546`

## Scope

This is an optional evaluation-runtime qualification only. It does not modify the canonical Base, tokenizer, trainer, checkpoint lineage, corpus, benchmark payloads or training policy.

## Upstream binding

Repository: `https://github.com/UKGovernmentBEIS/inspect_ai`

Package: `inspect-ai==0.3.260`

Immutable tag commit: `3f294e61b823d6bad5fc16706fc5825ea980c8ee`

License: MIT; exact `LICENSE` blob: `72fc87742ef8a944fab4f28fe6231696c62f2fa4`.

PyPI wheel: `inspect_ai-0.3.260-py3-none-any.whl`; SHA-256: `3da1dd4e4cbaec248b507799beb71eb9917eee1062eab1d9aeb6a8b5a03a386a`.

PyPI source distribution: `inspect_ai-0.3.260.tar.gz`; SHA-256: `5f6fbd7bc1fae0a770dc04e208daa9275de71f6d6b85b5fb68c162a1c4e0496f`.

The qualification fixture is project-authored synthetic data. No dataset or model rights are inferred from the software license.

## Environment-first result

Host: Debian GNU/Linux 13.3 (trixie), Linux 6.18.35, x86-64, Python 3.13.5, 5 visible CPUs, AMD EPYC 9V74, no visible NVIDIA GPU.

Available: python, pip, pip3, uv, git. Unavailable: Poetry, PDM, Conda.

Exact `inspect-ai==0.3.260` was not preinstalled and no matching local package-cache artifact was found.

PyPI access and direct Git access failed with DNS resolution errors. GitHub API access through the repository connector remained available, so upstream source identity and license were pinned independently.

## Installation

A dedicated environment `/tmp/inspect-bootstrap-env` was created.

Exact command executed:

`python -m pip install --disable-pip-version-check --no-input --no-cache-dir inspect-ai==0.3.260`

Result: `FAILED_EXACT_ARTIFACT_UNAVAILABLE`; exit code `1`; cause `NETWORK_DNS_UNAVAILABLE`.

No alternate version was installed. Global Python was not modified.

## Runtime, parity and benchmark

Real Inspect AI runtime execution was not possible because the exact package artifact was unavailable. Therefore this worker records no runtime PASS, benchmark PASS or parity PASS.

`tools/run_inspect_ai_bootstrap_stress_v1.py` is a real runtime runner. When exact 0.3.260 is installed, it exercises Inspect AI's public Dataset/Sample, TaskState, ModelOutput, solver and match-scorer APIs using a project-owned deterministic synthetic solver. It makes no external model-provider call. The returned score is compared to an independent exact-string oracle.

The intended runtime benchmark is two independent fresh-process executions with identical fixture/config identity. Record wall time and RSS only as local machine observations.

## Adversarial controls

The committed validator rejects upstream release/tag/commit/license drift, base-SHA drift, canonical Base contamination, foreign-weight claims, tokenizer mutation, wheel-hash drift, and false `ADOPTABLE_COMPONENT` status without actual installation/runtime/benchmark/parity evidence.

Offline adversarial tests cover false adoption and evidence-state integrity. Runtime-dependent negative cases remain unexecuted until the exact dependency is installed.

## Verdict

`RETEST_RUNTIME_REQUIRED`

This is not an adoption claim. The durable handoff is the exact version/hash contract, environment record, runner, fail-closed validator, focused tests and deterministic retest procedure.

## Retest procedure

1. Provision or preseed the exact `inspect_ai-0.3.260-py3-none-any.whl` and verify SHA-256 `3da1dd4e4cbaec248b507799beb71eb9917eee1062eab1d9aeb6a8b5a03a386a`.
2. Install only inside a dedicated environment and capture the resulting dependency freeze/SBOM.
3. Run the validator.
4. Execute the real runner twice in clean processes without external model/provider calls.
5. Require exact score parity against the project-owned oracle on identical fixture/task/config identities.
6. Add runtime negative tests for tampered identity/config and malformed task state.
7. Re-read live main, issue #784, current ownership and newest Inspect evidence before any promotion decision.
