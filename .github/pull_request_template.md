## Lane / issue

## Exact scope

## Base SHA / head SHA

## Tests and CI

## Training/eval run IDs

## Artifacts / manifests / hashes

## NOT TESTED

## Risks / blockers

## Ownership / overlap check

- Search live PRs/issues before implementation and name any overlapping owner or authority.
- Do not create a parallel remediation when an active PR already owns the same defect.

## CI queue discipline

- Ordinary PR validation uses the shared `.github/workflows/ci.yml`.
- New task-specific workflows are manual/reusable (`workflow_dispatch` / `workflow_call`), not automatic PR/push workflows.
- Queued or stale-head CI is not PASS evidence.
