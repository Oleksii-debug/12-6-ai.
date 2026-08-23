# D09 Post-Training Runtime Compatibility Snapshot

Status: research compatibility snapshot and dry-run planning only. No runtime in this document is installed or executed by D09, and no behavioral training is authorized.

## Observed 2026-08-23 versions

- TRL latest observed stable release: `1.10.0` (PyPI release 2026-08-13).
- verl latest observed stable release: `0.9.0` (PyPI release 2026-08-14).
- vLLM latest observed stable release: `0.27.1` (PyPI release 2026-08-11).

Current TRL **main** vLLM-integration documentation states that TRL supports vLLM `0.16.0` through `0.23.0`. Current verl installation documentation states that vLLM `0.18.0` and later are supported for rollout. Therefore the highest version inside the documented intersection at this cutoff is **vLLM `0.23.0`**.

vLLM `0.27.1` is newer but outside TRL's declared range. Versions `0.24.0` through `0.27.1` are not selected for the joint TRL+verl planning path unless upstream compatibility documentation changes and the snapshot is deliberately revised.

This corrects an earlier same-day research snapshot that had recorded a wider TRL range and selected `0.26.0`. That earlier pin was never installed or executed, produced no weights, and is superseded by snapshot `posttraining-runtime-compat-2026-08-23-r2`.

Sources:

- https://pypi.org/project/trl/
- https://huggingface.co/docs/trl/main/vllm_integration
- https://pypi.org/project/verl/0.9.0/
- https://verl.readthedocs.io/en/latest/start/install.html
- https://pypi.org/project/vllm/

## Why the project does not use `latest` blindly

Post-training stacks have cross-version constraints. A newer release of one component can temporarily fall outside another component's declared compatibility range. The checked-in snapshot therefore distinguishes:

1. latest observed upstream release;
2. selected compatibility version for a future integration spike;
3. actual runtime-tested environment.

Only item 2 exists here. Item 3 remains NOT TESTED.

## Dry-run rollout planner

`src/twelve_six/posttraining/rollout_planning.py` converts the existing D09 `RolloutRequest` into a deterministic, content-addressed plan for one of three future targets:

- `vllm_offline`;
- `trl_vllm_server`;
- `verl_vllm`.

The planner normalizes `max_new_tokens`/`max_tokens`, temperature, top-p, top-k, seed and candidate count. It binds the request to an immutable `CheckpointRef`, records exact research runtime version pins, and emits deterministic request/plan SHA-256 identities.

It does **not** import TRL, verl or vLLM, open a network connection, allocate a GPU, generate a token, update a reward, compute a gradient or mutate a checkpoint. `execution_enabled=true` is rejected by construction.

## Cross-lane boundary

D07 owns actual inference/generation implementations. D09 owns future post-training rollout orchestration contracts. This package deliberately consumes only the generic D09 request shape and checkpoint identity; it does not copy D07 generation code or claim that vLLM can already load a 12-6 checkpoint.

D01/D05/D07 must first stabilize the exact model/export/inference surfaces needed by an external runtime. D06 retains evaluation/stage-gate authority. D10/Auditors retain promotion authority. An explicit owner decision remains required before behavioral-training weights are produced.

## NOT TESTED

- importing TRL `1.10.0` in the project environment;
- importing verl `0.9.0` in the project environment;
- importing vLLM `0.23.0` or `0.27.1`;
- CUDA/driver/PyTorch compatibility;
- a real vLLM load of 12-6 weights;
- TRL server or colocate mode;
- verl rollout worker execution;
- weight synchronization;
- SFT/DPO/GRPO/PPO/RL/reward-model training;
- external teacher or critic execution;
- paid compute.

Before any real runtime spike, re-check upstream docs, pin the exact dependency graph with hashes/container identity, and test against the exact D01/D05/D07 artifact identities. This snapshot is evidence for planning, not a production environment lock.
