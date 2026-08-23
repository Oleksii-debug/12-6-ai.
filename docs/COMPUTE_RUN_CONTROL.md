# Compute / Run Control

C01 owns the durable run queue and compute authorization boundary. This document defines the S0 execution matrix and launch policy; live queue state belongs in issue #16.

## Authorization classes

- `LOCAL_FREE`: may run on owner-controlled local CPU/GPU when no metered charge is created.
- `FREE_HOSTED`: may run only when the selected service/account is confirmed to create no charge and its policy permits the workload.
- `PAID_SMALL`: requires `COMPUTE_AUTHORIZED` or an owner-approved standing budget that explicitly covers the run.
- `PAID_LARGE`: requires explicit authorization for the exact candidate/config/budget.

`START`, `AUTOPULSE`, `continue`, or equivalent messages are never financial authorization.

## S0 local/free execution matrix

S0 targets an approximately 10K-parameter random-init causal model. The local/free path is CPU-first; GPU acceleration is optional, not a requirement.

| Work item | Preferred device | Minimum practical host class | Model/training tensor-memory class | Runtime class | Launch condition |
|---|---|---|---|---|---|
| import/config/parameter-count tests | CPU | Python 3.11+, 1 CPU core, 2 GiB RAM | << 1 MiB model state; process RSS dominated by Python/framework | seconds | candidate code exists |
| forward/loss/backward smoke | CPU | 1-2 CPU cores, 2 GiB RAM | << 10 MiB tensors at S0; framework dominates RSS | seconds-minutes | ModelSpec + tiny batch path resolved |
| checkpoint save/load/resume smoke | CPU | 1-2 CPU cores, 2 GiB RAM, local disk | checkpoint payload expected far below 10 MiB at S0; verify actual bytes | seconds-minutes | D05 serialization contract exists |
| tiny convergence smoke | CPU | 2 CPU cores recommended, 2-4 GiB RAM | model/AdamW state is negligible at ~10K params; activations/runtime dominate | minutes | dataset/tokenizer/trainer identities frozen for the run |
| S0 evidence training run | CPU first | 2-4 CPU cores recommended, 4 GiB RAM, sufficient local disk | expected to fit comfortably in host RAM; record measured peak RSS | minutes-hours depending on tokens/implementation | exact candidate SHA + all required hashes/configs + D06 eval plan |
| generation/eval probes | CPU | 1-2 CPU cores, 2 GiB RAM | negligible relative to framework | seconds-minutes | completed checkpoint + eval config |
| optional local CUDA acceleration | local GPU | CUDA-compatible GPU; >=2 GiB VRAM recommended for framework headroom | model itself is tiny; CUDA context/framework dominates | expected faster only if launch overhead does not dominate | only when local GPU is already available and unmetered |
| hosted GPU | none for S0 by default | N/A | N/A | N/A | do not use unless explicitly justified; paid use requires authorization |

The table gives engineering classes, not measured performance. Every completed run must replace estimates with actual wall time, peak RAM/VRAM, artifact sizes, tokens/steps, and cost.

## Memory accounting

For a ~10K-parameter FP32 model, raw weights are roughly 40 KiB. FP32 gradients add roughly another 40 KiB; Adam first/second moments add roughly 80 KiB. These are parameter-state payloads only and do not predict process RSS, allocator workspace, activations, dataloader buffers, Python overhead, or CUDA context overhead. C01 records measured peak host RAM/VRAM for every real run and does not extrapolate later stages from S0 framework overhead.

## Required identity before any integrated training launch

The launch manifest must bind all of the following:

- exact candidate Git SHA and branch/tag;
- ModelSpec hash and exact parameter count;
- dataset manifest hash;
- tokenizer hash/version and special-token identity;
- train/validation/test split identity;
- training config identity: seed, optimizer, scheduler, precision, context, global batch, steps/tokens, checkpoint interval;
- expected hardware profile;
- artifact/checkpoint destinations;
- retry, failure, cancellation, and NaN/Inf cutoffs;
- authorization class and cost ceiling when any charge is possible.

If any required identity is unresolved, state is `PREPARED_NOT_LAUNCHED`, not `RUNNING`.

## S0 run sequence

1. `S0-R00-STATIC`: install/import/lint/unit/parameter-count checks.
2. `S0-R01-FWD`: deterministic forward/loss/backward smoke.
3. `S0-R02-CKPT`: save/load/resume smoke.
4. `S0-R03-CONVERGE`: short deterministic convergence smoke on approved tiny train data.
5. `S0-R04-EVIDENCE`: integrated CPU-first S0 training run with checkpoints and measurements.
6. `S0-R05-EVAL`: D06-defined held-out evaluation/generation/regression probes.
7. `S0-R06-AUDIT`: manifest/hash/artifact handoff for AUDIT-A and AUDIT-B.

No later run in the sequence is launch-ready merely because an earlier run passed; each run must bind the exact candidate and inputs it actually uses.

## Retry / failure / cancellation policy

- Retry infrastructure/transient failures only after recording the original run ID, failure class, and evidence.
- Do not blindly retry deterministic code/data/config failures; return them to the owning lane.
- Abort on unexplained NaN/Inf, corrupted/unreadable checkpoint, dataset/tokenizer identity mismatch, candidate SHA drift, or authorization mismatch.
- Never silently resume from a checkpoint whose manifest does not match the intended candidate/config.
- For any paid run, stop before the spending ceiling can be exceeded and require a new authorization for scope/budget changes.

## Artifact convention

Preferred repository-neutral layout for local S0 runs:

`artifacts/runs/<run_id>/manifest.json`

`artifacts/runs/<run_id>/metrics.jsonl`

`artifacts/runs/<run_id>/hardware.json`

`artifacts/runs/<run_id>/checkpoints/`

`artifacts/runs/<run_id>/eval/`

Large/generated run artifacts should not be committed to Git unless D10/D05 explicitly define a small canonical evidence subset. Durable hashes and locations belong in the run manifest and issue #16.
