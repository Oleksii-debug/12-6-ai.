# HF Tokenizers Bootstrap Stress V1

Worker: SWARM-793. Claim: #793. Protocol: SWARM-300-V2. Base: `5020afd671a3885c1b738c8b4eafe7525f630546`. Execution profile: LOCAL_FREE.

## Purpose

Qualify Hugging Face Tokenizers as an optional future tokenizer-training runtime without changing the canonical S0 byte tokenizer. This package owns bootstrap truth only; it does not select or freeze a tokenizer.

## Immutable upstream identity

Repository: `https://github.com/huggingface/tokenizers`
Release tag: `v0.23.1`
Resolved commit: `7f1623b90b5adfb9bc327d4c3468d2f70bbce262`
Software license: Apache-2.0
Pinned LICENSE blob: `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`
NOTICE at release root: absent in the checked path; absence is recorded rather than inferred.

Target Linux x86-64 wheel: `tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
Wheel SHA-256: `5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4`

The exact release and wheel identity are bound before installation. No `latest`, `main`, or floating dependency is used as qualification evidence.

## Local environment truth

Host probe at this run: CPython 3.13.5, Linux x86_64, 5 CPUs, no detected NVIDIA GPU, pip 25.1.1, uv 0.10.0, git 2.47.3. Poetry, PDM and Conda were unavailable. The pip cache was disabled/unwritable and contained no usable exact Tokenizers artifact. Outbound DNS to both PyPI and GitHub was unavailable.

An isolated virtual environment was created successfully. Its initial freeze was `pip==25.1.1`.

The exact install was attempted with `--require-hashes`, `--only-binary=:all:`, `--no-deps`, one retry-disabled request and no version substitution. The install failed because the exact artifact could not be reached. The source `git ls-remote` probe also failed for the same network condition.

Therefore real Tokenizers import, encode/decode runtime, throughput, RSS, deterministic training behavior and project parity are **NOT EXECUTED**. They are not credited as PASS or adoption evidence.

## Validator and adversarial coverage

The validator is fail-closed for floating tags, version drift, wheel hash drift, global installation intent, fabricated successful runtime status and canonical Base contamination. Evidence identity is deterministic for identical non-volatile inputs.

Focused tests execute without third-party test dependencies and cover the same adversarial contract plus isolated virtual-environment creation and environment discovery.

## Benchmark boundary

Two fresh local virtual environments were created as a host-bootstrap benchmark. Observed elapsed times are stored in the machine-readable evidence as host-specific telemetry. These timings are not an upstream Tokenizers speed claim.

Component runtime benchmark remains `NOT_EXECUTED` because the exact wheel was unavailable.

## Rights boundary

The checked software license is Apache-2.0. This does not grant training or redistribution rights to datasets, model weights, or third-party artifacts that may be used with Tokenizers. This worker makes no such rights inference.

## Canonical Base firewall

No foreign pretrained weights, instruction/alignment behavior, tokenizer replacement, model training, checkpoint mutation or final-test payload was used. Canonical S0 `s0-byte-v1` remains unchanged and remains the project tokenizer authority.

## Retest procedure

1. Use a LOCAL_FREE host with package-index access or the exact pinned wheel in a local cache.
2. Verify the wheel SHA-256 exactly before installation.
3. Create a fresh virtual environment.
4. Install `tokenizers==0.23.1` using the pinned wheel/hash and no dependency substitution.
5. Freeze the environment and bind its identity to the exact upstream commit and wheel.
6. Import Tokenizers in a fresh process and record `__version__`.
7. Run a bounded project-owned tokenizer fixture twice; compare token IDs and serialized artifact identity under the existing tokenizer experiment contract.
8. Measure host-local latency/RSS/throughput twice in clean environments.
9. Complete the project-vs-upstream semantic parity gate before any tokenizer freeze or training use.

## Final state for this run

`RETEST_RUNTIME_REQUIRED`.

This is the truthful state because the exact runtime was discovered and installation was genuinely attempted, but the required exact artifact could not be acquired on the local host. No alternate Tokenizers version was installed.
