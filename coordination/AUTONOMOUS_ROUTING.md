# 12-6 AI — current autonomous routing

EPOCH: EPOCH-0001
STATUS: ACTIVE
LAST_GLOBAL_AUDIT: bootstrap-required
NEXT_AUDIT_RULE: first available worker after >=6h since last valid audit

## Operating rule

All recurring workers, Codex Cloud runs and Work integrators must read this file plus live GitHub before substantive work. Live GitHub overrides stale text here.

## Default worker lanes

- Worker 1: DATA / corpus / provenance / dedup / decontamination.
- Worker 2: tokenizer / packing / unique-loss accounting.
- Worker 3: checkpoint / training / recovery.
- Worker 4: evaluation / contamination firewall / scientific QA.
- Worker 5: integration / CI / ownership; perform global routing audit only when >=6h has elapsed since the last valid audit, otherwise continue integration work.

## Current global priority

Converge the learned-20M critical path. Do not divert the project into unrelated discovery while corpus, tokenizer, recovery, evaluation and training readiness remain incomplete.

## Coordination rule

If a lane is already complete, stale or actively owned elsewhere, re-route to the highest-value unowned blocker instead of continuing the old assignment. Every worker should leave durable progress that another worker can recover.

## Codex Cloud rule

Codex Cloud must read active worker ownership, claim a disjoint package, avoid duplicating the next scheduled-worker window, and refresh this routing document before a natural stopping point when possible.

## Failover

If this file is older than the live project, reconstruct routing from GitHub and refresh it. If a coordinator run is missed, the first worker seeing routing older than ~8 hours may perform a minimal failover audit.
