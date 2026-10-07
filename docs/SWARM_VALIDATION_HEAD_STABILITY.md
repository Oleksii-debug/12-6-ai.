# SWARM-300 validation-head stability

## Purpose

Protect scarce GitHub Actions capacity without weakening fail-closed integration or scientific promotion.

A live-main advance by itself is not a reason to mutate every open PR. Repeatedly merging `main` into an already-tested branch moves the PR head, invalidates exact-head CI authority, and can enqueue another full CI run even when the new main change is unrelated to the owned package. Under swarm load that creates validation churn instead of integration throughput.

## Non-negotiable truth

- Queued, running, pending, action-required, cancelled or absent CI is not PASS.
- CI success applies only to the exact PR head SHA that was tested.
- If the PR head changes for any reason, prior exact-head CI does not qualify the new head.
- This policy never turns a stale or untested head green.
- Scientific gates that bind an exact base, corpus, evaluation, model, checkpoint or other authority remain exact and fail closed.

## Stable-head rule

Once exact-head shared or scoped CI has started, keep that PR head stable while the run is queued or running. Do not merge or rebase live `main` into the branch merely because `main` advanced.

A bridge/rebase and fresh exact-head CI are required when any of these are true:

1. GitHub reports the PR non-mergeable or there is an actual conflict.
2. Main-since-tested-base changes overlap the package's owned surfaces.
3. Main-since-tested-base changes overlap declared dependency, integration, workflow, test or authority surfaces used by the package.
4. A scientific or release authority explicitly requires a newer exact base SHA.
5. Dependency semantics changed, or the integration owner cannot establish that the advance is safely disjoint.

If none applies, leave the branch head unchanged and let its exact-head CI finish.

## Late integration check

Before intake of a green stable head:

1. refresh live `main` and the PR head;
2. require the PR head to be unchanged from the successful CI head;
3. require current mergeability;
4. inspect main changes since the tested base for owned-surface overlap;
5. inspect declared dependency, test, workflow and scientific-authority overlap, not just filename overlap;
6. require no exact-base authority mismatch;
7. merge only with expected-head protection or an equivalent compare-and-swap guard.

If any check is uncertain or fails, do one late bridge to the current required base and rerun exact-head CI. Do not keep chasing additional disjoint main commits while that new CI is queued or running.

## CI pressure behavior

At GREEN pressure, required bridges may run normally. At AMBER or RED pressure, a bridge/requeue whose only justification is `main moved` is prohibited. Scarce validation slots are reserved for substantive P0/P1 packages and genuinely required integration evidence.

This rule separates development concurrency from validation concurrency. It does not authorize self-merge, paid compute, dataset promotion, tokenizer fitting, model training, evaluation leakage or stage promotion.

## Observed motivating pattern — 2026-09-07

The learned-20M data spine exposed the failure mode directly: critical quality/privacy/split/packing branches repeatedly accumulated current-main ancestry, moved from previously tested heads, and entered new shared-CI queues while their own narrow mechanics remained the same. The correct response is not to weaken exact-head truth; it is to stop gratuitously changing the exact head.

Refs: #11 #548 #723 #906.