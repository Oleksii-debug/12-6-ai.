# Workflow queue-hygiene contract

## Why this exists

12-6 AI uses many scoped LOCAL_FREE scientific workflows. During the 20M readiness campaign the repository accumulated a large Actions queue, so a newly pushed workflow must not create unlimited superseded work.

This policy is preventive. It does not retroactively cancel historical queued runs and it does not turn a queued, cancelled, or stale run into scientific evidence.

## Required invariants

Every file under `.github/workflows/*.yml` or `.github/workflows/*.yaml` must satisfy all of the following:

1. A top-level `concurrency` block exists.
2. `concurrency.cancel-in-progress` is exactly `true`.
3. The concurrency group includes `github.workflow` and a pull-request/ref discriminator so unrelated workflows do not cancel one another while superseded executions of the same workflow/ref can be cancelled.
4. Top-level `permissions` is explicit. `write-all` is forbidden. Explicit write permissions remain possible when a workflow genuinely needs them and must be reviewed in that workflow's PR.
5. Every normal job has a positive bounded `timeout-minutes` value no greater than 360 minutes. A job implemented purely by `uses:` of a reusable workflow is exempt because its execution policy lives in the called workflow.
6. Tab indentation and duplicate top-level mapping keys fail closed in the repository validator.

The validator intentionally uses the Python standard library rather than PyYAML so it can execute immediately after checkout, before dependency installation.

## CI ordering

Base CI runs:

1. checkout;
2. `python tools/validate_workflow_hygiene.py`;
3. Python setup and dependency installation;
4. Ruff over `src`, `tests`, and `tools`;
5. Pytest.

The workflow-policy check therefore fails before normal dependency/setup cost whenever a repository workflow violates the contract.

## Scientific truth boundary

Workflow hygiene is infrastructure evidence only. It is not model-quality evidence, corpus readiness, tokenizer authorization, checkpoint integrity, stage promotion, or compute authorization. Exact-head scientific workflows still require their own terminal results, and stale or superseded runs must never be inherited as PASS evidence.
