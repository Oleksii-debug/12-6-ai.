# CI Swarm Policy

## Problem

12-6 AI uses many autonomous development lanes. Creating a dedicated GitHub Actions workflow for each temporary worker or experiment multiplies queued jobs and delays exact-head verification for higher-priority model, data, checkpoint and training gates.

## Default rule

Use `.github/workflows/ci.yml` as the shared repository CI surface. A normal development PR must not create, copy or rename a file into `.github/workflows/`.

Existing legacy specialized workflows may be modified or deleted while older development lineages are being converged. The canonical shared `ci.yml` may be modified but must not be deleted or renamed.

A genuinely permanent new workflow is an infrastructure-policy change, not a worker convenience. It requires an explicit separately reviewed change to this guard before the workflow is added.

## Worker guidance

Prefer adding focused pytest coverage, validators, deterministic probes and committed evidence to the repository, then let shared CI execute them. Do not create a one-off workflow merely to obtain a dedicated status check.

The workflow guard is diff-based. It can therefore be cherry-picked onto historical branches that already contain legacy workflows without forcing those files to be deleted in the same change; it blocks only new workflow proliferation and protection of the canonical shared CI path.

## Queue discipline

Shared CI uses per-PR/ref concurrency cancellation, a bounded job timeout and pip caching. Superseded runs for the same PR should be cancelled rather than allowed to consume runner capacity.

This policy does not authorize paid compute, model training, dataset promotion or stage promotion. It only protects verification capacity.
