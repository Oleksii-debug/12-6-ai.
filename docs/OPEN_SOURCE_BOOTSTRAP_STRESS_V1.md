# OPEN-SOURCE-BOOTSTRAP-STRESS-V1

## Scope

This package independently qualifies the existing ENV-151 universal exact execution bootstrap. It does not reimplement or modify ENV-151. The incumbent implementation remains PR #313 (`env151/universal-execution-bootstrap-20260826`) at the scout head `bbca2101ea9409b47d844dd8292cd7f2290e3ff0`.

The worker owns only the additive verifier, tests, configuration, evidence and handoff listed by issue #783. The canonical Base, model/tokenizer/data/training/checkpoint/evaluation surfaces and shared workflow implementation are out of scope.

## Exact binding

Project main at qualification start: `5020afd671a3885c1b738c8b4eafe7525f630546`.

D09 claim authority: issue #783, exact lane `D09|EXECUTION-BOOTSTRAP|INDEPENDENT-VERIFY|STRESS-V1`.

ENV-151 incumbent: PR #313, branch `env151/universal-execution-bootstrap-20260826`, head `bbca2101ea9409b47d844dd8292cd7f2290e3ff0`.

The PR contains the reusable composite action, bootstrap implementation, capability registry, CPU lock, toolchain lock and focused tests. This worker binds their Git blob identities rather than copying their source into the new package.

## Environment result

The actual LOCAL_FREE host is Debian 13 / Linux x86_64 with CPython 3.13.5. CPython 3.11.16 is not installed. The exact interpreter acquisition was explicitly attempted through `uv python install 3.11.16` and failed because the required download was unavailable. Local package caches contain no wheel/sdist artifacts suitable to prove the exact ENV-151 stack. DNS resolution for both `pypi.org` and `github.com` is unavailable in the worker environment. No GPU is visible and no GPU compute was attempted.

Because ENV-151 source is not present on current main and the exact interpreter/artifacts are unavailable locally, the real ENV-151 dependency stack was **NOT EXECUTED**. No substituted package version is used.

## Stress coverage

The verifier checks immutable project/incumbent identities, SHA-shaped references, safe relative paths, the no-new-workflow boundary, Local_Free truth, canonical-surface avoidance, deterministic machine-readable evidence hashing, and explicit failure classification.

Adversarial cases cover absolute paths, parent-directory traversal, formatting-vs-identity stability, runtime-unavailable classification, and explicit absence of training/foreign-weight/paid-compute claims.

The verifier is intentionally independent of third-party packages and therefore runs even when the exact ENV-151 environment cannot be bootstrapped.

## Parity boundary

No runtime parity claim is made. Exact-input semantic/numerical parity requires a real CPython 3.11.16 environment with the exact PR #313 dependency artifacts, followed by execution of the real ENV-151 bootstrap and a clean repeat.

## Rights

The verifier package is project-owned additive qualification code. No foreign model weights or datasets are used. Third-party dependency rights are not represented as fully audited because the exact runtime artifacts could not be retrieved locally; adoption therefore remains blocked pending the real dependency qualification gate.

## Retest procedure

Use a clean Linux x86_64 environment with CPython 3.11.16 and the exact PR #313 source head. Obtain the exact hashed artifacts from the declared package indexes or an auditable local wheelhouse, create a fresh virtual environment, run the ENV-151 bootstrap without substituting versions, execute the incumbent focused suite, repeat the bootstrap in a second clean environment, and compare the two machine manifests. Only after those steps should latency/throughput/RSS and exact parity be recorded.
