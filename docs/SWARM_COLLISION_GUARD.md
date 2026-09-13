# Swarm collision guard

The project intentionally runs many autonomous workers, but ephemeral orchestration task IDs must have one active owner path. Duplicate ownership creates competing evidence, stale authority claims, duplicate pull requests, and avoidable GitHub Actions pressure.

`tools/swarm_collision_report.py` queries the repository's open GitHub issues endpoint and inspects only explicit ephemeral task claims found in issue/PR titles or `SWARM_WORKER_ID` lines. The initial guarded namespaces are `NEXT100-NNN` and `GNN-TNN`.

A normal lifecycle is exactly one open issue plus at most one open implementation PR for the same task key. The guard blocks multiple open issues or multiple open PRs claiming the same ephemeral key. Permanent lanes such as D01/D02/D10 are intentionally excluded because they can own multiple successive changes.

The workflow uses one repository-wide concurrency group with `cancel-in-progress: true`, so a burst of swarm events supersedes older guard runs instead of multiplying queue pressure. It uses only the standard library and read-only GitHub permissions.

Collision reports are machine-readable, self-hashed JSON artifacts. A collision is a coordination blocker, not evidence that either implementation is technically wrong. COORD must select or compose the stronger authority, close/supersede the duplicate ownership path, and only then allow downstream work to treat the task key as singular.

This guard does not merge code, promote a model/data authority, authorize compute, read evaluation payloads, or alter training behavior.
