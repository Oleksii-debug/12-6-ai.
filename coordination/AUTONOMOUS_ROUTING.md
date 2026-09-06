# 12-6 AI — current autonomous routing

EPOCH: EPOCH-0002
STATUS: ACTIVE
LAST_GLOBAL_AUDIT: bootstrap-required
NEXT_AUDIT_RULE: first available coordinator/worker performs immediate full audit; after that, first available worker after >=6h since last valid audit

## Mandatory first-run bootstrap audit

Before the coordinator establishes or changes worker directions, it MUST first compare the newest durable project evidence and reconstruct the actual current state. The first run is not allowed to assume that yesterday's worker prompts are still correct.

First-run audit must inspect, at minimum:

- current main project state;
- newest integrated work and still-active work;
- current test/CI/runtime evidence;
- current learned-20M/training readiness state;
- active ownership and collisions;
- latest durable GitHub reports/control files;
- relevant current Google Drive master reports/vision when accessible;
- work completed by recent Work/Codex Cloud runs that may have invalidated old scheduled-worker assignments.

The coordinator then classifies every existing worker assignment as one of:

- KEEP — still current and valuable;
- CHANGE — same lane, but next target changed;
- STOP_STALE — already completed, superseded or no longer useful;
- COLLISION — another active worker/Codex already owns it;
- PROMOTE — newly critical because the project state changed.

Only after that comparison may the coordinator publish the new routing epoch.

## Operating rule

All recurring workers, Codex Cloud runs and Work integrators must read this file plus live GitHub before substantive work. Live GitHub overrides stale text here.

No worker is allowed to continue an old task merely because its Scheduled Task prompt names that task. The permanent prompt defines the worker's lane and startup procedure; this routing file defines the current target.

## Default worker lanes

- Worker 1: DATA / corpus / provenance / dedup / decontamination.
- Worker 2: tokenizer / packing / unique-loss accounting.
- Worker 3: checkpoint / training / recovery.
- Worker 4: evaluation / contamination firewall / scientific QA.
- Worker 5: integration / CI / ownership; perform global routing audit only when >=6h has elapsed since the last valid audit, otherwise continue integration work.

These are stable home lanes, not permanent exact tasks. The coordinator may re-route within or across lanes when evidence shows that the critical path changed.

## Current global priority

Converge the learned-20M critical path. Do not divert the project into unrelated discovery while corpus, tokenizer, recovery, evaluation and training readiness remain incomplete.

When learned-20M readiness changes materially, the coordinator must immediately recompute worker priorities. Once a blocker becomes terminal, workers must move forward to the next real training/inference milestone instead of polishing already-terminal infrastructure.

## Six-hour meta-audit without losing a worker

Do not dedicate one of five workers permanently to coordination.

Normal rule:

`if no valid full audit exists OR now - last_global_audit >= 6h OR major-change-trigger == true: perform audit + refresh routing; then continue productive project work`

Otherwise:

`continue normal lane work`

Major-change triggers include:

- a large Work or Codex Cloud package completes;
- a major integration changes the critical path;
- learned-20M/training gate changes state;
- a worker discovers that its assignment is already complete;
- ownership collision/stale lease is found;
- major CI/runtime blocker appears or is removed;
- current routing contradicts live evidence.

If the normal coordinator has not refreshed the routing for ~8 hours, the first capable worker may perform a failover audit.

## Coordination rule

If a lane is already complete, stale or actively owned elsewhere, re-route to the highest-value unowned blocker instead of continuing the old assignment. Every worker should leave durable progress that another worker can recover.

Before taking work, every worker must check current ownership. After taking work, it must leave a current ownership marker/checkpoint. Ownership that stops producing evidence must eventually be treated as stale after verification.

## Codex Cloud rule

Codex Cloud must read this routing state and active worker ownership before substantive work.

At start it should:

1. reconstruct newest live state;
2. identify what the scheduled workers are already expected to do during the next several hours;
3. choose a disjoint high-value package or an integration package that explicitly needs cross-lane work;
4. avoid consuming a package already delegated to another live worker unless it is intentionally taking over stale work;
5. leave durable checkpoints after each major completed phase, not only at the very end.

When Codex completes a major package, it should refresh this routing file or leave enough durable evidence for the next coordinator to do so immediately. If credits end abruptly, the system must recover from GitHub and the latest valid routing state rather than waiting for Oleksii.

## Work / principal-auditor rule

A strong Work run is a periodic principal architect/integrator, not a parallel universe.

At the beginning it must read this routing state and current live project evidence. It should perform a deeper whole-project audit when useful, correct architectural drift, update priorities, and then take a difficult package that does not duplicate active workers.

A major Work result is itself a routing refresh trigger. Scheduled workers that wake after it must re-read this file/current evidence and must not continue invalidated assignments.

## Coordinator output after every audit

A successful coordination audit must leave a compact durable state containing:

- new EPOCH;
- audit time;
- user-visible project stage;
- what changed since the previous audit;
- what is now complete and must not be repeated;
- current top blockers in priority order;
- Worker 1–5 current targets;
- Codex Cloud active/next recommended package;
- Work active/next recommended package;
- active ownership/collisions;
- stale work to stop;
- integration queue;
- training/compute authorization state;
- next audit rule;
- short owner-readable summary in Ukrainian.

## Failover

If this file is older than the live project, reconstruct routing from GitHub and refresh it. If a coordinator run is missed, the first worker seeing routing older than ~8 hours may perform a minimal failover audit.

If Google Drive is unavailable but GitHub is reachable, continue from GitHub. Drive is a cross-account/master mirror, not a single point of failure.

## Core invariant

Oleksii must not have to manually enter a chat merely to say “read the latest reports and update the workers.” The autonomous system itself is responsible for comparing the newest evidence, detecting stale assignments, refreshing directions and continuing along the actual critical path.