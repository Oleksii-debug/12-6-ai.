# COMPUTE-252 GPU campaign orchestration

Worker: `COMPUTE-252-GPU-CAMPAIGN-ORCHESTRATOR`

## Scope

This change is orchestration only. It does not provision hardware, purchase compute, authorize paid compute, embed credentials, simulate CUDA, or turn CPU results into GPU evidence.

The canonical campaign manifest is `configs/compute/compute252_gpu_campaign.json`. The CPU-only DAG validator is `tools/run_compute252_gpu_campaign.py`. The validator never imports Torch and never probes CUDA; its dry-run contract is exactly zero target-device measurements.

## Exact runner contract

The campaign inherits GPU-200's stricter self-hosted runner label set without weakening it:

`self-hosted, linux, x64, gpu, cuda, twelve-six-ai`

The target purpose environment remains `linux-x86_64-cuda-training`, bound to CPython 3.11.16, PyTorch 2.13.0 and Torch CUDA 13.0 by the parent environment contract.

## Dependency DAG

The intended manual execution order is:

1. `hardware_environment_preflight` — GPU-200.
2. `precision_decision` — GPU-199 executed target-device precision authority.
3. `ten_m_semantic_smoke` — GPU-201 exact 10M target-device campaign.
4. `ten_m_performance` — consume the same GPU-201 evidence artifact; do not rerun the campaign merely to relabel its performance measurements.
5. `checkpoint` — consume the same GPU-201 selected-precision checkpoint round-trip evidence.
6. `activation_checkpointing` — SCALE-205.
7. `compile` — PERF-206.
8. `hundred_m_qualification` — SCALE-202.
9. `two_gpu_fsdp` — DIST-203, only when at least two compatible CUDA devices and NCCL actually exist.
10. `two_gpu_dcp_recovery` — CHECKPOINT-204, only after the two-GPU FSDP gate passes.

Every stage descriptor is hashed from the stage id, authoritative parent, exact parent source SHA, artifact identity, dependencies and required terminal status. A dependent stage cannot become ready while any prerequisite is blocked.

## Current fail-closed result

The DAG is constructible, but the full target campaign is not yet executable from the listed parent authorities.

- GPU-199 currently publishes `NOT_RUN_NO_GPU` evidence but no dedicated target-device executor. The older TRAIN-15 single-GPU mechanics runner is not substituted for the missing GPU-199 precision-decision authority.
- PERF-206 is not present in the current repository as a discoverable PR, branch, issue or commit. PERF-59 is not substituted for it.
- DIST-203 currently publishes `NOT_RUN_INSUFFICIENT_GPU` evidence but no dedicated target-device executor. Merely attaching two GPUs therefore does not convert it into a runnable stage.

These are structural parent blockers, not GPU failures. GPU-201, SCALE-205, SCALE-202 and CHECKPOINT-204 remain bound to their existing exact prepared runners and are not copied into a new experiment implementation.

## Dry run

`.github/workflows/compute252-gpu-campaign.yml` always supports a CPU DAG dry run. The dry run:

- validates the acyclic dependency graph;
- validates exact self-hosted labels and CUDA purpose-environment name;
- validates parent SHA syntax and executable-parent presence;
- records deterministic per-stage artifact descriptor hashes;
- propagates dependency blockers;
- records `target_device_measurements_executed = 0`;
- records `paid_compute_authorized = false`.

It intentionally does not rerun the existing no-GPU hardware probes.

A manual `request_target_campaign=true` dispatch performs readiness evaluation on CPU first. While any structural blocker remains, it exits before scheduling any self-hosted GPU job. This avoids consuming a GPU runner merely to rediscover a known parent deficiency.

## Unlock rule

The target campaign may be unlocked only by binding real accepted successors for the missing parent capabilities in the manifest:

- a device-bound GPU-199 precision executor producing the precision handoff required by GPU-201;
- the actual PERF-206 compile authority and target executor;
- a DIST-203 two-GPU FSDP/NCCL executor or accepted successor.

Those bindings must carry exact source SHAs and artifacts. No compatibility alias, CPU extrapolation or older experiment may be silently promoted.

## Claim boundary

`DAG_PREPARED_BLOCKED_MISSING_PARENT_EXECUTORS`

This status means the campaign ordering, artifact-flow contract and fail-closed CPU dry run are prepared, while target-device execution remains blocked by missing authoritative parent executors. It is not GPU evidence and it is not paid-compute authorization.
