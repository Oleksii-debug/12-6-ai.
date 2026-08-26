# CI Specialist Trigger Scoping V1

## Purpose

Reduce GitHub Actions runner fan-out on the S1 convergence spine without weakening the scientific checks owned by specialist workflows.

The live control-plane PR that motivated this package triggered shared CI plus seven specialist workflows even though none of their scientific input surfaces changed. Those specialist workflows previously used an unfiltered `pull_request` trigger.

## Routing contract

The following workflows now have conservative `pull_request.paths` ownership, PR-scoped cancellation, and `workflow_dispatch` for deliberate exact-head requalification:

- D02 S0 real training: S0 model/training/data/packing/tokenization/checkpoint inputs, S0 stage data/config, exact runner/validator tooling and locked Linux environment.
- D02 S0 repeatability: the same S0 scientific surface plus repeatability collection/validation tooling.
- D02 S1 numerical preflight: S1 model/training/data/packing/tokenization/checkpoint inputs, S1 stage config, exact runner/validator tooling and locked Linux environment.
- D08 purpose environments: purpose-environment verifier, lock/profile definitions and package metadata only.
- DATA-21-22 external source intake: D03 data code, the exact external-source candidate registry, owned intake tests/tooling and locked Linux environment.
- SCALE-02 S2 executable preflight: S2 model/training/checkpoint/inference/data/packing/tokenization inputs, S2 config and locked Linux environment.
- TRAIN-29 S1 observability: S1 model/training/data/packing/tokenization/checkpoint inputs, observability tooling/tests, S1 config and locked Linux environment.

Every workflow includes its own workflow file in `paths`, so a routing or scientific-workflow edit still requalifies that workflow automatically.

## Safety boundary

Generic CI remains the repository-wide regression authority. This package does not remove any scientific step, lower any PASS threshold, alter model/data/checkpoint semantics, authorize training, or authorize paid compute.

`workflow_dispatch` uses `github.sha` when there is no pull-request head, preserving explicit exact-head execution for cross-lane qualification.

The path lists deliberately avoid global `src/**`, `tests/**`, and `**` patterns. If a future dependency crosses an ownership boundary, add the exact dependency path rather than restoring unconditional pull-request execution.

## Verification

`tests/test_ci_specialist_trigger_scoping.py` fails if any of the seven workflows loses path scoping, manual dispatch, PR-scoped cancellation, self-trigger coverage, manual exact-head fallback, or reintroduces a global source/test glob.

This is queue-control engineering only. Queued or running workflow executions are not PASS evidence.
