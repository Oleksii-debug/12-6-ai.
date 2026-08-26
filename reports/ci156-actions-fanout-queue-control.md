# CI-156 Actions Fan-out and Queue Control

Baseline authority: pull-request workflows observed on exact SHA `c6e8ac2784920cb96b9b34d26cd68cf9468bd5f0`.

## Baseline

A MILESTONE-150 research SHA triggered eight workflows: MILESTONE-150, CI, D02 S1 Numerical Preflight, D02 Real S0 Training, SCALE-02 S2 1M Executable Preflight, TRAIN-29 S1 Training Observability, D02 S0 Determinism Repeatability, and D08 Purpose Environments.

The five unrelated single-job experiment workflows consumed 402 runner-seconds before terminating. D08 consumed another 392 runner-seconds across Linux purpose runtimes, Windows bundle production, and Windows runtime validation. The CI ARM64 job consumed 130 runner-seconds although the MILESTONE-150-only change was not portability-sensitive. The resulting gross avoidable baseline is 924 runner-seconds, or 15.4 runner-minutes, for this representative SHA. This is a wall-duration estimate from completed jobs, not a billing estimate.

The environment fan-out also repeated an expensive installation pattern. `verify_locked_environment.py` constructs two independent clean environments and installs the locked runtime in each; several legacy experiment workflows then construct a third execution environment and install the runtime again. This remains scientifically authoritative today, so CI-156 does not weaken that evidence contract. Instead it prevents unrelated workflows from paying that cost at all and adds a cheap lock/source preflight before expensive work.

## After-policy

The repository integration spine remains `CI`. Its x86_64 path remains the normal full gate. ARM64 portability validation is conditional on dependency, lock, packaging, or portability-sensitive changes. D02, SCALE-02, TRAIN-29, and D08 workflows retain automatic PR execution only for their own scientific surface and retain `workflow_dispatch` for intentional evidence runs. MILESTONE-150 remains automatic on its milestone target branch only when ladder-relevant code, corpus, tokenizer, training, checkpoint, inference, lock, or test surfaces change.

Superseded pull-request SHA runs are canceled within the same workflow. Manually dispatched historical evidence runs are not assigned to the cancelable PR concurrency group.

## Representative trigger simulation

| Scenario | Before | After | Result |
|---|---:|---:|---|
| MILESTONE-150-only research SHA | 8 workflows | 2 workflows | CI + MILESTONE-150; 6 workflow runs avoided; ARM64 skipped |
| D02 real-training change | 7 workflows | 2 workflows | CI + D02 Real S0 Training |
| Dependency-lock change | 7 workflows | 2 workflows | CI + D08; ARM64 retained |
| Documentation-only change | 7 workflows | 0 compute workflows | no scientific compute gate is scheduled |

The executable simulation is `tools/ci156_trigger_simulation.py`; regression assertions are in `tests/test_ci156_trigger_simulation.py`.

## Migration rules

1. Never remove an integration-spine required gate merely to make a branch green.
2. Apply a path filter only when changes outside the declared scope cannot alter the workflow's scientific evidence.
3. Preserve `workflow_dispatch` for expensive or historical experimental campaigns.
4. Cancel only superseded PR SHA runs; never cancel an intentionally dispatched historical evidence run solely for queue reduction.
5. Dependency, packaging, lock, or portability changes must still trigger the appropriate portability matrix.
6. Branch-specific experiments must not fan out because a PR merely targets an experiment branch.
7. Run cheap source/dependency/tool preflight before tests and multi-gigabyte environment installation.
8. Refactoring the full locked-environment verifier to eliminate its internal second clean install requires a separate evidence-schema migration; CI-156 does not silently weaken it.
