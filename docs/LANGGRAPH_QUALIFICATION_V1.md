# LangGraph durable runtime qualification v1

This package qualifies LangGraph as a replaceable post-Base durable-execution reference for 12-6 AI. The project-owned task/evidence schema remains authoritative. LangGraph is not part of canonical Base and does not supply model weights, tokenizer behavior, instruction tuning, or alignment behavior.

## Pinned upstream

Repository: `https://github.com/langchain-ai/langgraph`  
Release tag: `1.2.11`  
Resolved commit: `644815f9e5bc52ad8f7a5227a456227e9c3e639b`  
Software license: MIT. The immutable upstream `libs/langgraph/LICENSE` was read at the pinned commit. No separate NOTICE/COPYING file exists under `libs/langgraph`; repository-level notices, if introduced by future integration, must be audited separately.

The immutable library metadata declares Python `>=3.10`, version `1.2.11`, and direct dependencies on `langchain-core`, `langgraph-checkpoint`, `langgraph-sdk`, `langgraph-prebuilt`, `xxhash`, and `pydantic`. The upstream library also carries a `uv.lock`; this qualification records its immutable blob identity but does not copy that monorepo lock into 12-6 because it contains local path sources for sibling packages.

## Installation and environment truth

The worker created a dedicated virtual environment and attempted an exact `langgraph==1.2.11` installation. The attempt could not contact the package index because DNS resolution for PyPI and GitHub was unavailable. No replacement version was installed and global packages were not modified. Local cache directories contained no usable exact LangGraph distribution.

The worker runtime is Debian 13 x86_64, Python 3.13.5, pip 25.1.1, uv 0.10.0, git 2.47.3, five visible CPU cores, and no NVIDIA GPU. Existing unrelated packages include CPU-only PyTorch 2.10.0, NumPy 2.3.5, pytest 9.0.2 and safetensors 0.7.0. LangGraph, ruff, transformers and tokenizers were absent before the attempt.

## What was executed

The real LangGraph graph probe is implemented and lazily imports `StateGraph`, `START` and `END`; when the exact package is installed it builds a one-node state transition and invokes it twice on the same input. This run did not execute that probe because the exact package was unavailable.

Project-owned mechanics were executed independently: strict task-state schema validation, atomic JSON checkpoint publication with fsync/replace, restart read-back, deterministic hashing, and adversarial rejection of unknown fields, duplicate/overlapping steps, unsafe task IDs and malformed state. These are orchestration mechanics, not proof of LangGraph runtime quality.

A small local I/O benchmark is available for the project-owned checkpoint path. No LangGraph runtime benchmark was recorded, and no upstream speed claim was copied into 12-6 evidence.

## Parity contract

When the exact runtime becomes available, the successor run must compare the same fixture transition against the project-owned transition function using identical inputs and expected state semantics. It must verify deterministic repeated invocation, checkpoint/restart behavior under the exact supported checkpointer path selected for integration, error propagation, and absence of project-schema drift. A mismatch blocks promotion.

## Adversarial gates

The validator must fail closed on dependency version drift, upstream commit/license drift, evidence tampering, fabricated runtime completion, unknown state keys, unsafe task IDs, duplicate steps, status drift, corrupt checkpoint bytes and canonical-Base contamination markers. Upstream-specific checkpoint/state fields must not become implicit project authority.

## Promotion state

This package is a **candidate qualification with runtime retest required**. It is not `PARITY_PROVEN` and not `ADOPTED`. The exact dependency must be installed and executed in a network-enabled, isolated environment before any runtime or performance conclusion is made.
