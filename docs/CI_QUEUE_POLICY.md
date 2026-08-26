# CI queue convergence policy

## Problem

The 20M development lineage already has cancellation enabled in the shared `.github/workflows/ci.yml`, but the repository accumulated hundreds of queued GitHub Actions runs because many task-specific pull requests introduced separate automatic workflows. Concurrency groups cannot deduplicate different workflow files across different PRs.

The practical result is that exact-head evidence arrives too late to converge duplicated work. A queued run is never PASS authority.

## Policy

The shared `ci.yml` is the automatic pull-request gate.

A newly added task-specific workflow must not trigger automatically on `pull_request`, `pull_request_target`, or `push`. New dedicated workflows must instead expose `workflow_dispatch` or `workflow_call`. Runnable new workflows must declare a timeout.

Existing automatic workflows may be maintained, but any modified automatic workflow must keep top-level concurrency with `cancel-in-progress: true`.

The rule is fail-closed in `tools/check_workflow_queue_policy.py` and is executed by shared CI against the exact base/head diff.

## Worker behavior during queue saturation

Before creating a workflow or a remediation PR, search live open PRs/issues for the same defect and authority lineage. If another active owner already has the same scope, contribute to convergence or take a different residual blocker instead of creating a parallel implementation.

Use the shared CI for ordinary code, tests, lint, checkpoint, model, data-contract, and integration changes. Reserve `workflow_dispatch` for bounded expensive or authority-specific jobs that genuinely need a separate execution surface.

Do not infer success from `queued`, `pending`, `skipped`, or stale-head runs. Exact-head completed evidence remains required for terminal promotion.

## Scope

This policy controls queue growth and duplicate verification pressure. It does not weaken scientific gates, data rights, checkpoint integrity, test coverage, or paid-compute authorization. It also does not cancel already queued historical runs; those require separate Actions administration or natural queue drain.
