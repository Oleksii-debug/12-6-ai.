# 12-6 AI — autonomous worker orchestration control plane

## Purpose

This document defines how recurring scheduled workers, long Codex Cloud runs and occasional high-capability Work runs cooperate without requiring Oleksii to manually rewrite worker prompts after every development wave.

The design goal is **five productive development workers plus coordination as an intermittent duty**, not four developers plus one permanently wasted coordinator.

## Core rule: stable prompts, live routing

Worker prompts should be stable and generic. They should not contain brittle task assignments such as “implement X until finished”.

Instead, every run MUST recover live state and read the current routing authority before substantive work.

Recommended live authority:

- `coordination/AUTONOMOUS_ROUTING.md` — human-readable current plan;
- live GitHub main/PR/issues/CI/ownership — operational truth;
- Drive mirror/report — owner-readable summary and cross-account continuity.

If routing conflicts with newer live GitHub state, live GitHub wins. If routing is stale, workers must not continue obsolete work merely because an old prompt named it.

## No dedicated 20% coordinator tax

All five recurring workers remain development workers.

One designated worker, preferably Worker 5 / Integrator, carries a **meta-coordination duty** only when due. On normal hourly runs it performs its development lane like the others.

Do not reserve one worker exclusively for coordination unless measured evidence later shows that coordination itself consumes most of a worker’s capacity.

## Audit cadence: elapsed-time trigger, not brittle wall-clock slots

Do NOT rely only on exact times such as 06:00, 12:00, 18:00 and 00:00.

Use this rule instead:

`if now - last_global_audit >= 6 hours: perform global audit and refresh routing; else: continue normal lane work`

This survives missed runs, quota exhaustion, timezone changes and delayed schedules.

The routing file must record:

- `last_global_audit`;
- current development epoch/version;
- current highest-value blockers;
- active ownership/leases;
- completed work since previous audit;
- stale assignments to stop;
- next recommended lane per worker;
- integration priority;
- unresolved human/compute approvals.

If the designated coordinator misses its run, the first worker that detects a routing age beyond a larger fallback threshold (recommended 8 hours) may perform a minimal failover audit and refresh the file.

## Worker startup algorithm

Every scheduled worker run should:

1. read `coordination/AUTONOMOUS_ROUTING.md`;
2. inspect relevant live GitHub state;
3. verify that its prior assignment is still unfinished and unowned by another active worker;
4. if assignment is obsolete/finished/colliding, take the current recommended safe lane or highest-value unowned blocker;
5. claim/refresh its ownership lease;
6. perform real development/integration/testing/training work;
7. persist durable progress;
8. update its short lane checkpoint for the next run.

Workers should not wait for a prompt rewrite when the live project has moved.

## Ownership leases

Use time-bounded ownership rather than permanent claims.

Each active lane should record:

- worker identity;
- scope;
- start/refresh time;
- expected next checkpoint;
- lease expiry;
- current state: ACTIVE / BLOCKED / READY_TO_INTEGRATE / TERMINAL / RELEASED.

A worker may reclaim an expired lane only after checking live activity. This prevents duplication while allowing recovery after quota exhaustion or abandoned sessions.

## Codex Cloud relationship

A long Codex Cloud run is a high-throughput implementer/integrator, not a separate universe.

At run start it should:

1. read current routing and live GitHub;
2. identify which lanes are already reserved for scheduled workers;
3. publish/refresh its own temporary ownership;
4. choose a disjoint high-value package;
5. avoid taking work already delegated to the next scheduled-worker window unless integration requires it.

Before a natural stopping point, Codex Cloud should:

1. write durable project progress;
2. update the shared routing file with what changed;
3. release or transfer ownership;
4. identify the next highest-value packages for scheduled workers.

This is much cheaper and more reliable than trying to directly rewrite the configuration/prompt of four separate scheduled tasks.

If Codex credits end abruptly before the final handoff, scheduled workers still recover from live GitHub and the last valid routing state.

## Work / high-capability run relationship

A strong Work run may act as periodic principal auditor/integrator. It should read the same control plane, resolve architectural conflicts, update routing, then take a disjoint hard package.

Work is not required for every routing refresh. The ordinary six-hour meta-audit must remain executable by the recurring worker system.

## Routing refresh triggers

Refresh routing when any of these occurs:

- six hours elapsed since the previous global audit;
- a major blocker becomes terminal;
- a large integration changes the critical path;
- a learned-model/training gate changes state;
- a worker collision or stale ownership is detected;
- Codex Cloud completes a major run;
- Work completes a major integration/audit;
- the current routing file is inconsistent with live state.

## Failover rules

If the routing file is missing or stale:

- do not stop development;
- reconstruct from live GitHub;
- perform a minimal audit;
- recreate/refresh routing;
- proceed with an unowned high-value task.

If API/Drive access is unavailable but GitHub is reachable, GitHub is sufficient to continue.

If all external state sources are unavailable, avoid irreversible integration and leave a local/durable checkpoint when possible.

## Recommended five-worker shape for 12-6 AI

The exact lanes may change dynamically, but a useful default is:

1. DATA / corpus / provenance / dedup / decontamination;
2. tokenizer / packing / unique-loss accounting;
3. checkpoint / training / recovery;
4. evaluation / contamination firewall / scientific QA;
5. integration / CI / ownership / routing meta-duty every ~6h.

Worker 5 is still an implementation/integration worker on ordinary runs.

## Development epoch

Every routing refresh increments a simple epoch label, for example `EPOCH-0007`.

Workers record which epoch they read. If they wake with an older epoch than the current file, they must re-evaluate their assignment before continuing.

## Shared routing schema

A human-readable file should contain at minimum:

- epoch;
- generated_at;
- last_global_audit;
- source-of-truth summary;
- top blockers;
- worker 1–5 assignments;
- active ownership;
- Codex/Work active package if any;
- integration queue;
- training/compute state;
- stale work to stop;
- next audit due after;
- one-paragraph owner summary.

A machine-readable JSON equivalent may be added later if automation benefits from it.

## Key invariant

**Do not solve coordination by continually rewriting worker prompts. Solve it by making all workers read a common live control plane and re-route from current evidence.**

This lets development accelerate even when Codex Cloud, Work and recurring workers advance the project at different speeds.